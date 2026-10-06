import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xrfm import xRFM

from benchmark.config import N_TRIALS, NUM_FOLDS, RANDOM_STATE, TEST_SIZE
from benchmark.metrics import evaluate_classification, evaluate_regression
from benchmark.models.search_spaces import SEARCH_SPACES


def make_processor(metadata: dict[str, Any]) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric scaling and categorical OHE.

    Args:
        metadata: Dataset metadata dict with key ``num_cols``, and optionally
            ``cat_cols`` and ``bin_cols``.

    Returns:
        Unfitted ColumnTransformer with a StandardScaler step for numeric columns
        and, when categorical/binary columns are present, an OHE step.
    """
    num_cols = metadata["num_cols"]
    cat_cols = metadata.get("cat_cols", []) + metadata.get("bin_cols", [])
    transformers = [("scaler", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def build_categorical_info(
    fitted_pipeline: Pipeline, n_num_cols: int
) -> dict[str, Any]:
    """Build the xRFM categorical metadata dict from a fitted pipeline.

    Extracts per-category index ranges and identity vectors from the fitted OHE
    transformer so xRFM can handle categorical features via learned embeddings.

    Args:
        fitted_pipeline: Pipeline whose ``processor`` step contains a fitted OHE
            transformer under the key ``"ohe"``.
        n_num_cols: Number of numeric (scaled) columns; OHE columns start at this index.

    Returns:
        Dict with keys:
            ``numerical_indices``: 1-D LongTensor of numeric feature positions.
            ``categorical_indices``: List of 1-D LongTensors, one per categorical column.
            ``categorical_vectors``: List of identity matrices (float32), one per categorical column.
    """
    ohe = fitted_pipeline.named_steps["processor"].named_transformers_["ohe"]
    categorical_indices = []
    categorical_vectors = []
    start = n_num_cols
    for cats in ohe.categories_:
        cat_len = len(cats)
        idxs = torch.arange(start, start + cat_len, dtype=torch.long)
        categorical_indices.append(idxs)
        categorical_vectors.append(torch.eye(cat_len, dtype=torch.float32))
        start += cat_len
    numerical_indices = torch.arange(0, n_num_cols, dtype=torch.long)
    return dict(
        numerical_indices=numerical_indices,
        categorical_indices=categorical_indices,
        categorical_vectors=categorical_vectors,
    )


def fit_xrfm_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    processor: ColumnTransformer,
    n_num_cols: int,
    task: str,
    device: torch.device,
    n_trials: int = N_TRIALS,
) -> optuna.trial.FrozenTrial:
    """Run Optuna hyperparameter search for xRFM using cross-validation.

    Seeds PyTorch globally before the search for reproducibility.

    Args:
        X: Feature DataFrame (will be CV-split internally).
        y: Target array.
        processor: Unfitted ColumnTransformer; cloned fresh per fold.
        n_num_cols: Number of numeric columns, used to build categorical info.
        task: ``"regression"`` or ``"classification"``.
        device: PyTorch device for xRFM computations.
        n_trials: Number of Optuna trials to run.

    Returns:
        Best Optuna trial, including its params and CV score.
    """
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    def objective(trial: optuna.trial.Trial) -> float:
        """Evaluate one set of xRFM hyperparameters via cross-validation.

        Closes over ``X``, ``y``, ``processor``, ``n_num_cols``, ``task``,
        and ``device`` from the enclosing scope.

        Args:
            trial: Optuna trial supplying hyperparameter suggestions.

        Returns:
            Mean fold score (MSE for regression, macro F1 for classification).
        """
        s = SEARCH_SPACES["xrfm"]
        exponent = trial.suggest_distribution("exponent", s["exponent"])

        kernel_type = trial.suggest_distribution("kernel_type", s["kernel_type"])
        model_params: dict = {
            "bandwidth": trial.suggest_distribution("bandwidth", s["bandwidth"]),
            "exponent": exponent,
            "diag": trial.suggest_distribution("diag", s["diag"]),
            "bandwidth_mode": "constant",
        }
        if kernel_type == "kpq":
            model_params["kernel"] = "lpq"
            model_params["norm_p"] = trial.suggest_distribution(
                "norm_p", s["norm_p"](exponent)
            )
        else:
            model_params["kernel"] = "l2"

        rfm_params = {
            "model": model_params,
            "fit": {
                "reg": trial.suggest_distribution("reg", s["reg"]),
                "verbose": False,
                "early_stop_rfm": True,
            },
        }
        tuning_metric = "mse" if task == "regression" else "f1"
        kf = (
            StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            if task == "classification"
            else KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        )
        fold_scores = []

        for train_idx, val_idx in kf.split(X, y if task == "classification" else None):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline = Pipeline(steps=[("processor", clone(processor))])
            X_train = pipeline.fit_transform(X_train)
            X_val = pipeline.transform(X_val)

            cat_info = (
                build_categorical_info(pipeline, n_num_cols)
                if n_num_cols < X_train.shape[1]
                else None
            )

            model = xRFM(
                rfm_params=rfm_params,
                device=device,
                tuning_metric=tuning_metric,
                split_method="top_vector_agop_on_subset",
                categorical_info=cat_info,
            )
            model.fit(X_train, y_train, X_val, y_val)

            preds = model.predict(X_val)
            if task == "regression":
                score = mean_squared_error(y_val, preds)
            else:
                score = f1_score(y_val, preds, average="macro")
            fold_scores.append(score)

        return np.mean(fold_scores)

    direction = "minimize" if task == "regression" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial


def fit_best_xrfm(
    X: pd.DataFrame,
    y: np.ndarray,
    best_trial: optuna.trial.FrozenTrial,
    processor: ColumnTransformer,
    n_num_cols: int,
    task: str,
    device: torch.device,
) -> tuple[xRFM, Pipeline, float, dict[str, Any], dict[str, Any]]:
    """Fit xRFM with the best Optuna hyperparameters on a train/val split.

    Args:
        X: Full feature DataFrame.
        y: Full target array.
        best_trial: Optuna trial containing the best hyperparameters.
        processor: Unfitted ColumnTransformer to embed in the pipeline.
        n_num_cols: Number of numeric columns, used to build categorical info.
        task: ``"regression"`` or ``"classification"``.
        device: PyTorch device for xRFM computations.

    Returns:
        Tuple of (fitted xRFM model, fitted pipeline, fit time in seconds,
        train metrics dict, validation metrics dict).
    """
    p = best_trial.params
    model_params: dict = {
        "bandwidth": p["bandwidth"],
        "exponent": p["exponent"],
        "diag": p["diag"],
        "bandwidth_mode": "constant",
    }
    if p["kernel_type"] == "kpq":
        model_params["kernel"] = "lpq"
        model_params["norm_p"] = p["norm_p"]
    else:
        model_params["kernel"] = "l2"

    rfm_params = {
        "model": model_params,
        "fit": {
            "reg": p["reg"],
            "verbose": False,
            "early_stop_rfm": True,
        },
    }
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[("processor", clone(processor))])
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)

    cat_info = (
        build_categorical_info(pipeline, n_num_cols)
        if n_num_cols < X_train.shape[1]
        else None
    )

    tuning_metric = "mse" if task == "regression" else "f1"
    model = xRFM(
        rfm_params=rfm_params,
        device=device,
        tuning_metric=tuning_metric,
        split_method="top_vector_agop_on_subset",
        categorical_info=cat_info,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train, X_val, y_val)
    fit_time = time.perf_counter() - t0

    if task == "regression":
        train_metrics = evaluate_regression(y_train, model.predict(X_train))
        val_metrics = evaluate_regression(y_val, model.predict(X_val))
    else:
        train_metrics = evaluate_classification(y_train, model.predict_proba(X_train))
        val_metrics = evaluate_classification(y_val, model.predict_proba(X_val))

    return model, pipeline, fit_time, train_metrics, val_metrics
