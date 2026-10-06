from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor

from benchmark.config import (
    DATA_PROCESSED,
    DATASETS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_BENCHMARK,
)
from benchmark.metrics import (
    evaluate_classification,
    evaluate_regression,
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report,
)

TaskType = Literal["classification", "regression"]

DATASET_TASKS: dict[str, TaskType] = {k: v["task"] for k, v in DATASETS.items()}


@dataclass(slots=True)
class TabPFNTestResult:
    """Container for TabPFN test-time outputs.

    Attributes:
        y_test: True target values for the test set.
        y_pred: Predicted target values for the test set.
        y_proba: Predicted class probabilities (classification only), else None.
        train_time_s: Time taken to fit the model in seconds.
        predict_time_s: Time taken to make predictions on the test set in seconds.
        inference_time_per_sample_s: Average per-sample inference time in seconds.
    """

    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    train_time_s: float
    predict_time_s: float
    inference_time_per_sample_s: float


class TabPFNTableModel:
    """TabPFN wrapper for tabular classification and regression benchmarks.

    Handles data loading, preprocessing, model training, prediction, evaluation,
    and result saving. Supports optional downsampling to accommodate hardware
    limits and passes through extra kwargs to the underlying TabPFN estimator.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        processed_root: Path | None = None,
        models_root: Path | None = None,
        results_root: Path | None = None,
        random_state: int | None = None,
        ignore_pretraining_limits: bool = False,
        device: str = "auto",
        predict_batch_size: int | None = 8,
        downsample: bool = False,
        max_train_rows: int | None = None,
        max_test_rows: int | None = None,
        **tabpfn_kwargs: Any,
    ) -> None:
        """Initialize TabPFNTableModel for the given dataset.

        Args:
            dataset_id: Dataset key; must be present in DATASET_TASKS.
            processed_root: Root for processed CSV files; defaults to DATA_PROCESSED.
            models_root: Root for saved model artefacts; defaults to MODELS_DIR.
            results_root: Root for result JSONs; defaults to RESULTS_BENCHMARK.
            random_state: Random seed; defaults to RANDOM_STATE.
            ignore_pretraining_limits: Skip TabPFN's built-in row-count caps.
            device: Device string for TabPFN (``"auto"``, ``"cpu"``, ``"cuda"``, etc.).
            predict_batch_size: Chunk test rows into batches of this size; None disables batching.
            downsample: Enable downsampling to ``max_train_rows``/``max_test_rows``.
            max_train_rows: Upper bound on training rows (subject to TabPFN limits when downsampling).
            max_test_rows: Upper bound on test rows.
            **tabpfn_kwargs: Extra kwargs forwarded to the TabPFN estimator constructor.
        """
        if dataset_id not in DATASET_TASKS:
            raise ValueError(
                f"Unknown dataset_id {dataset_id!r}. Expected one of {sorted(DATASET_TASKS)}.",
            )
        self.dataset_id = dataset_id
        self.task: TaskType = DATASET_TASKS[dataset_id]
        self._processed_root = (
            Path(processed_root) if processed_root is not None else DATA_PROCESSED
        )
        self._models_root = Path(models_root) if models_root is not None else MODELS_DIR
        self._results_root = (
            Path(results_root) if results_root is not None else RESULTS_BENCHMARK
        )
        rs = RANDOM_STATE if random_state is None else random_state
        self._random_state = rs
        self._tabpfn_init: dict[str, Any] = {
            "device": device,
            "ignore_pretraining_limits": ignore_pretraining_limits,
            "random_state": rs,
            **tabpfn_kwargs,
        }
        self._device = device
        self._ignore_pretraining_limits = ignore_pretraining_limits
        self._predict_batch_size = predict_batch_size
        self._downsample = downsample
        self._max_train_rows = max_train_rows
        self._max_test_rows = max_test_rows
        meta = self._read_metadata()
        self._target_col: str = meta["target_col"]
        self._num_cols: list[str] = list(meta["num_cols"])
        self._cat_cols: list[str] = list(meta.get("cat_cols", []))
        self._bin_cols: list[str] = list(meta.get("bin_cols", []))
        self._feature_cols: list[str] = self._num_cols + self._cat_cols + self._bin_cols
        self._categorical_indices: list[int] | None
        if len(self._num_cols) < len(self._feature_cols):
            self._categorical_indices = list(
                range(len(self._num_cols), len(self._feature_cols)),
            )
        else:
            self._categorical_indices = None
        self._estimator: TabPFNClassifier | TabPFNRegressor | None = None
        self._label_encoder: LabelEncoder | None = None
        self._train_time_s: float = 0.0
        self._last_train_df: pd.DataFrame | None = None
        self._last_test_df: pd.DataFrame | None = None

    def _read_metadata(self) -> dict[str, Any]:
        path = self._processed_root / self.dataset_id / "metadata.json"
        with path.open() as f:
            return json.load(f)

    def load_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw train and test CSVs for this dataset.

        Returns:
            Tuple of (train DataFrame, test DataFrame).
        """
        base = self._processed_root / self.dataset_id
        train = pd.read_csv(base / "train.csv")
        test = pd.read_csv(base / "test.csv")
        return train, test

    def _downsample_frame(
        self, df: pd.DataFrame, *, max_rows: int | None, split_name: str
    ) -> pd.DataFrame:
        """Optionally downsample a DataFrame to at most ``max_rows`` rows.

        For classification tasks the sample is stratified on the target column.

        Args:
            df: DataFrame to potentially downsample.
            max_rows: Row cap; skipped if None or if downsampling is disabled.
            split_name: Label used in error messages (e.g. ``"train"`` or ``"test"``).

        Returns:
            Original or downsampled DataFrame with reset index.

        Raises:
            ValueError: If ``max_rows`` is non-positive.
        """
        if not self._downsample or max_rows is None or len(df) <= max_rows:
            return df
        if max_rows <= 0:
            raise ValueError(f"{split_name} max_rows must be positive, got {max_rows}.")

        stratify = None
        if self.task == "classification":
            target = df[self._target_col]
            if target.nunique(dropna=False) > 1:
                stratify = target

        sampled, _ = train_test_split(
            df,
            train_size=max_rows,
            random_state=self._random_state,
            stratify=stratify,
        )
        return sampled.reset_index(drop=True)

    def _has_hardware_acceleration(self) -> bool:
        """Check whether the configured device has GPU/MPS acceleration.

        Returns:
            True if CUDA or MPS is available for the configured device.
        """
        if self._device == "cpu":
            return False
        if self._device == "auto":
            return torch.cuda.is_available() or torch.backends.mps.is_available()
        return str(self._device).startswith(("cuda", "mps"))

    def _effective_max_train_rows(self) -> int | None:
        """Resolve the effective training-row cap after applying TabPFN limits.

        When downsampling is enabled and pretraining limits are active (no hardware
        acceleration), the cap is clamped to 1000 (TabPFN's default CPU limit).

        Returns:
            Effective row cap, or None for no limit.
        """
        if not self._downsample:
            return self._max_train_rows
        if self._ignore_pretraining_limits or self._has_hardware_acceleration():
            return self._max_train_rows
        if self._max_train_rows is None:
            return 1000
        return min(self._max_train_rows, 1000)

    def prepare_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and optionally downsample the train and test DataFrames.

        Returns:
            Tuple of (prepared train DataFrame, prepared test DataFrame).
        """
        train, test = self.load_train_test()
        train = self._downsample_frame(
            train, max_rows=self._effective_max_train_rows(), split_name="train"
        )
        test = self._downsample_frame(
            test, max_rows=self._max_test_rows, split_name="test"
        )
        return train, test

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from a DataFrame.

        Args:
            df: Source DataFrame containing at least the feature columns.

        Returns:
            DataFrame containing only the feature columns.

        Raises:
            ValueError: If any expected feature columns are absent.
        """
        missing = set(self._feature_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing feature columns {sorted(missing)} for dataset {self.dataset_id!r}"
            )
        return df[self._feature_cols].copy()

    def _raw_y(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the raw target column as a NumPy array.

        Args:
            df: Source DataFrame containing the target column.

        Returns:
            1-D NumPy array of raw target values.

        Raises:
            ValueError: If the target column is absent.
        """
        if self._target_col not in df.columns:
            raise ValueError(f"Missing target column {self._target_col!r}")
        return df[self._target_col].to_numpy()

    def _y(self, df: pd.DataFrame, *, fit_encoder: bool = False) -> np.ndarray:
        """Extract and encode target values for the current task.

        For classification, encodes string labels to integers using a LabelEncoder
        (fit when ``fit_encoder=True``, transform-only otherwise). For regression,
        casts to ``float32``.

        Args:
            df: Source DataFrame containing the target column.
            fit_encoder: If True, fit (and store) the LabelEncoder before transforming.

        Returns:
            Encoded integer labels (classification) or float32 values (regression).

        Raises:
            RuntimeError: If ``fit_encoder`` is False and the encoder has not been fit yet.
        """
        y = self._raw_y(df)
        if self.task == "classification":
            if fit_encoder:
                self._label_encoder = LabelEncoder()
                return self._label_encoder.fit_transform(y)
            if self._label_encoder is None:
                raise RuntimeError("Classification label encoder is not fitted yet.")
            return self._label_encoder.transform(y)
        return y.astype(np.float32, copy=False)

    def _make_estimator(self) -> TabPFNClassifier | TabPFNRegressor:
        if self.task == "classification":
            return TabPFNClassifier(
                categorical_features_indices=self._categorical_indices,
                **self._tabpfn_init,
            )
        return TabPFNRegressor(
            categorical_features_indices=self._categorical_indices,
            **self._tabpfn_init,
        )

    def fit(self, train: pd.DataFrame | None = None) -> float:
        """Fit the TabPFN estimator on the training data.

        Args:
            train: Training DataFrame; loaded via ``prepare_train_test`` if None.

        Returns:
            Training time in seconds.
        """
        if train is None:
            train, _ = self.prepare_train_test()
        self._last_train_df = train
        X = self._X(train)
        y = self._y(train, fit_encoder=self.task == "classification")
        self._estimator = self._make_estimator()
        t0 = time.perf_counter()
        self._estimator.fit(X, y)
        self._train_time_s = time.perf_counter() - t0
        return self._train_time_s

    def _iter_test_batches(self, X_test: pd.DataFrame) -> list[pd.DataFrame]:
        if self._predict_batch_size is None or self._predict_batch_size >= len(X_test):
            return [X_test]
        return [
            X_test.iloc[i : i + self._predict_batch_size]
            for i in range(0, len(X_test), self._predict_batch_size)
        ]

    def predict_test(self, test: pd.DataFrame | None = None) -> TabPFNTestResult:
        """Run inference on the test set, optionally in batches.

        Args:
            test: Test DataFrame; loaded via ``prepare_train_test`` if None.

        Returns:
            TabPFNTestResult with predictions, probabilities, and timing info.

        Raises:
            RuntimeError: If ``fit()`` has not been called first.
        """
        if self._estimator is None:
            raise RuntimeError("Call fit() before predict_test().")
        if test is None:
            _, test = self.prepare_train_test()
        self._last_test_df = test
        X_test = self._X(test)
        y_test = self._y(test)
        n = len(X_test)
        batches = self._iter_test_batches(X_test)
        t0 = time.perf_counter()
        y_proba: np.ndarray | None
        if self.task == "classification":
            proba_batches = [
                self._estimator.predict_proba(X_batch) for X_batch in batches
            ]
            y_proba = np.concatenate(proba_batches, axis=0)
            class_indices = np.argmax(y_proba, axis=1)
            y_pred = self._estimator.classes_[class_indices]
        else:
            pred_batches = [self._estimator.predict(X_batch) for X_batch in batches]
            y_pred = np.concatenate(pred_batches, axis=0)
            y_proba = None
        predict_time_s = time.perf_counter() - t0
        return TabPFNTestResult(
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            train_time_s=self._train_time_s,
            predict_time_s=predict_time_s,
            inference_time_per_sample_s=predict_time_s / max(n, 1),
        )

    def run(self) -> TabPFNTestResult:
        train, test = self.prepare_train_test()
        self.fit(train)
        return self.predict_test(test)

    @property
    def class_names(self) -> list[str] | None:
        if self._label_encoder is None:
            return None
        return list(self._label_encoder.classes_)

    @property
    def model_path(self) -> Path:
        return self._models_root / self.dataset_id / "tabpfn.joblib"

    @property
    def results_dir(self) -> Path:
        return self._results_root / self.dataset_id

    @property
    def results_path(self) -> Path:
        return self.results_dir / "tabpfn.json"

    def save_model(self, path: Path | None = None) -> Path:
        if self._estimator is None:
            raise RuntimeError("Call fit() before save_model().")
        target = path if path is not None else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load_model(cls, path: Path) -> "TabPFNTableModel":
        return joblib.load(path)

    def evaluate_test(
        self, test: pd.DataFrame | None = None
    ) -> tuple[TabPFNTestResult, dict[str, Any]]:
        """Predict on the test set and compute evaluation metrics.

        Args:
            test: Test DataFrame; loaded via ``prepare_train_test`` if None.

        Returns:
            Tuple of (TabPFNTestResult, metrics dict). The metrics dict includes
            fit/inference times, TabPFN config, and task-specific scores.

        Raises:
            RuntimeError: If classification probabilities are unavailable.
        """
        result = self.predict_test(test)
        output: dict[str, Any] = {
            "fit_time_s": result.train_time_s,
            "infer_time_per_sample_s": result.inference_time_per_sample_s,
            "tabpfn_params": dict(self._tabpfn_init),
            "downsample": self._downsample,
            "max_train_rows": self._max_train_rows,
            "max_test_rows": self._max_test_rows,
            "effective_max_train_rows": self._effective_max_train_rows(),
            "train_rows_used": len(self._last_train_df)
            if self._last_train_df is not None
            else None,
            "test_rows_used": len(self._last_test_df)
            if self._last_test_df is not None
            else None,
        }
        if self.task == "classification":
            if result.y_proba is None:
                raise RuntimeError("Classification benchmark requires probabilities.")
            output.update(evaluate_classification(result.y_test, result.y_proba))
        else:
            output.update(evaluate_regression(result.y_test, result.y_pred))
        return result, output

    def save_results(
        self, result: TabPFNTestResult, output: dict[str, Any], *, indent: int = 4
    ) -> Path:
        """Save evaluation results to JSON; write classification plot artefacts.

        Args:
            result: Raw test result used to generate classification plots.
            output: Metrics dict to serialise as JSON.
            indent: JSON indentation width.

        Returns:
            Path to the saved JSON file.

        Raises:
            RuntimeError: If classification probabilities are missing.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("w") as f:
            json.dump(output, f, indent=indent)

        if self.task == "classification":
            if result.y_proba is None:
                raise RuntimeError("Classification benchmark requires probabilities.")
            class_names = self.class_names
            plot_roc_curve(
                result.y_test,
                result.y_proba,
                save_path=self.results_dir / "roc_curve_tabpfn.png",
                class_names=class_names,
            )
            plot_confusion_matrix(
                result.y_test,
                result.y_proba,
                save_path=self.results_dir / "confusion_matrix_tabpfn.png",
                class_names=class_names,
            )
            save_classification_report(
                result.y_test,
                result.y_proba,
                save_path=self.results_dir / "classification_report_tabpfn.json",
                class_names=class_names,
            )

        return self.results_path

    def run_benchmark(self) -> dict[str, Any]:
        train, test = self.prepare_train_test()
        self.fit(train)
        self.save_model()
        result, output = self.evaluate_test(test)
        self.save_results(result, output)
        return output


def run_tabpfn_benchmark(dataset_id: str, **tabpfn_kwargs: Any) -> dict[str, Any]:
    model = TabPFNTableModel(dataset_id, **tabpfn_kwargs)
    output = model.run_benchmark()
    return {
        "dataset_id": dataset_id,
        "task": model.task,
        **output,
        "model_path": str(model.model_path),
        "results_path": str(model.results_path),
    }


if __name__ == "__main__":
    print(run_tabpfn_benchmark("abalone", n_estimators=2))
