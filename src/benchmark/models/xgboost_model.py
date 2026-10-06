"""
Shared XGBoost helpers for regression (_reg) and classification (_clf) benchmarks.

Usage:
    from benchmark.models.xgboost_model import (
        build_preprocessor,
        get_search_space,
        build_pipeline_reg,   build_pipeline_clf,
        run_search_reg,       run_search_clf,
        final_fit_reg,        final_fit_clf,
        evaluate_reg,         evaluate_clf,
        export_results_reg,   export_results_clf,
    )
"""

import json
import warnings
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.integration import OptunaSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from benchmark.config import (
    MODELS_DIR,
    N_TRIALS,
    NUM_FOLDS,
    RANDOM_STATE,
    RESULTS_BENCHMARK,
)
from benchmark.models.search_spaces import SEARCH_SPACES
from benchmark.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_preprocessor(metadata: dict[str, Any]) -> ColumnTransformer:
    """Build a ColumnTransformer from dataset metadata.

    Numeric columns pass through unchanged; binary and categorical columns are
    one-hot-encoded with ``drop="first"`` to avoid multicollinearity.

    Args:
        metadata: Dataset metadata dict with optional keys ``num_cols``,
            ``cat_cols``, and ``bin_cols``.

    Returns:
        Unfitted ColumnTransformer ready to be embedded in a Pipeline.
    """
    numeric_columns = metadata.get("num_cols", [])
    categorical_columns = metadata.get("cat_cols", [])
    binary_columns = metadata.get("bin_cols", [])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            # ("num", StandardScaler(), numeric_columns),
            (
                "cat_dummy",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                binary_columns + categorical_columns,
            ),
        ],
        remainder="passthrough",
    )
    return preprocessor


def get_search_space() -> dict[str, Any]:
    """Return the Optuna search space for XGBoost, prefixed for Pipeline use.

    Returns:
        Dict mapping ``"model__<param>"`` keys to Optuna distributions.
    """
    return {f"model__{k}": v for k, v in SEARCH_SPACES["xgboost"].items()}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def build_pipeline_reg(
    preprocessor: ColumnTransformer, device: str = "cpu", n_jobs: int = 1
) -> Pipeline:
    """Build a regression Pipeline with a preprocessor and XGBRegressor.

    Args:
        preprocessor: Fitted or unfitted ColumnTransformer to use as the first step.
        device: XGBoost device string (``"cpu"``, ``"cuda"``, etc.).
        n_jobs: Parallelism for XGBoost; pass ``-1`` to use all cores.

    Returns:
        Unfitted sklearn Pipeline with steps ``("processor", ...), ("model", ...)``.
    """
    return Pipeline(
        steps=[
            ("processor", preprocessor),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    device=device,
                    random_state=RANDOM_STATE,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def run_search_reg(
    pipeline: Pipeline,
    search_space: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    k_folds: int = NUM_FOLDS,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    """Run OptunaSearchCV for regression using KFold CV and neg-MSE scoring.

    Args:
        pipeline: Unfitted Pipeline to search over.
        search_space: Optuna parameter distributions keyed as ``"model__<param>"``.
        x: Feature DataFrame.
        y: Target array.
        k_folds: Number of cross-validation folds.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (best pipeline params, best model-only params,
        best CV MSE, search time in seconds).
    """
    cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    xgb_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        cv=cv_splitter,
        n_trials=n_trials,
        scoring="neg_mean_squared_error",
        # scoring="neg_root_mean_squared_error",
        refit=True,
        random_state=RANDOM_STATE,
        study=study,
        n_jobs=20,
        verbose=1,
        return_train_score=False,
    )

    start = perf_counter()
    xgb_search.fit(x, y)
    end = perf_counter()

    best_search_params = xgb_search.best_params_
    best_model_params = {
        k.replace("model__", ""): v for k, v in best_search_params.items()
    }
    best_cv_mse = -xgb_search.best_score_

    return best_search_params, best_model_params, best_cv_mse, end - start


def final_fit_reg(
    preprocessor: ColumnTransformer,
    best_search_params: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    device: str = "cpu",
) -> tuple[Pipeline, float]:
    """Rebuild and fit the regression Pipeline on full data with best params.

    Args:
        preprocessor: ColumnTransformer to embed in the new pipeline.
        best_search_params: Params from ``run_search_reg`` keyed as ``"model__<param>"``.
        x: Full feature DataFrame.
        y: Full target array.
        device: XGBoost device string.

    Returns:
        Tuple of (fitted Pipeline, fit time in seconds).
    """
    pipeline = build_pipeline_reg(preprocessor, device=device, n_jobs=-1)
    pipeline.set_params(**best_search_params)

    start = perf_counter()
    pipeline.fit(x, y)
    end = perf_counter()

    return pipeline, end - start


def evaluate_reg(
    pipeline: Pipeline, x: pd.DataFrame, y: np.ndarray
) -> dict[str, float]:
    """Predict with a fitted regression Pipeline and compute metrics.

    Args:
        pipeline: Fitted regression Pipeline.
        x: Feature DataFrame.
        y: True target array.

    Returns:
        Dict with keys ``"rmse"`` and ``"infer_time_per_sample_s"``.
    """
    start = perf_counter()
    y_pred = pipeline.predict(x)
    end = perf_counter()

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    total_time = end - start

    return {
        "rmse": rmse,
        "infer_time_per_sample_s": total_time / len(y),
    }


def export_results_reg(
    data_name: str,
    pipeline: Pipeline,
    best_model_params: dict[str, Any],
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    fit_time: float,
    search_time: float,
    best_cv_mse: float,
) -> None:
    """Save the XGBoost model, pipeline, and regression results to disk.

    Writes ``xgboost.joblib``, ``xgboost_pipeline.joblib``, and ``xgboost.json``
    under the appropriate dataset subdirectories.

    Args:
        data_name: Dataset name used as the output subdirectory.
        pipeline: Fitted regression Pipeline.
        best_model_params: Best hyperparameters without the ``"model__"`` prefix.
        train_metrics: Metrics dict from ``evaluate_reg`` on the training set.
        test_metrics: Metrics dict from ``evaluate_reg`` on the test set.
        fit_time: Final model fit time in seconds.
        search_time: Hyperparameter search time in seconds.
        best_cv_mse: Best cross-validation MSE from the search phase.
    """
    # --- model artefacts ---
    models_dir = MODELS_DIR / data_name
    models_dir.mkdir(parents=True, exist_ok=True)

    model = pipeline.named_steps["model"]
    joblib.dump(model, models_dir / "xgboost.joblib")
    joblib.dump(pipeline, models_dir / "xgboost_pipeline.joblib")

    print(f"\nModel saved to {models_dir / 'xgboost.joblib'}")
    print(f"Pipeline saved to {models_dir / 'xgboost_pipeline.joblib'}")

    # --- results JSON ---
    results_dir = RESULTS_BENCHMARK / data_name
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "rmse": test_metrics["rmse"],
        "fit_time_s": fit_time,
        "infer_time_per_sample_s": test_metrics["infer_time_per_sample_s"],
        "best_params": best_model_params,
        "add_train_metric": {
            "rmse": train_metrics["rmse"],
            "infer_time_per_sample_s": train_metrics["infer_time_per_sample_s"],
            "search_time_s": search_time,
            "best_cv_mse": float(best_cv_mse),
        },
    }

    with open(results_dir / "xgboost.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {results_dir / 'xgboost.json'}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def build_pipeline_clf(
    preprocessor: ColumnTransformer,
    n_classes: int,
    device: str = "cpu",
    n_jobs: int = 1,
) -> Pipeline:
    """Build a classification Pipeline with a preprocessor and XGBClassifier.

    Uses ``binary:logistic`` for binary tasks and ``multi:softprob`` for multiclass.

    Args:
        preprocessor: Fitted or unfitted ColumnTransformer to use as the first step.
        n_classes: Number of target classes; controls the XGBoost objective.
        device: XGBoost device string (``"cpu"``, ``"cuda"``, etc.).
        n_jobs: Parallelism for XGBoost; pass ``-1`` to use all cores.

    Returns:
        Unfitted sklearn Pipeline with steps ``("processor", ...), ("model", ...)``.
    """
    if n_classes == 2:
        return Pipeline(
            steps=[
                ("processor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        device=device,
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    else:
        return Pipeline(
            steps=[
                ("processor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=n_classes,
                        tree_method="hist",
                        device=device,
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )


def run_search_clf(
    pipeline: Pipeline,
    search_space: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    k_folds: int = NUM_FOLDS,
    n_trials: int = N_TRIALS,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    """Run OptunaSearchCV for classification with StratifiedKFold and F1.

    Args:
        pipeline: Unfitted Pipeline to search over.
        search_space: Optuna parameter distributions keyed as ``"model__<param>"``.
        x: Feature DataFrame.
        y: Target label array.
        n_classes: Number of target classes; selects ``"f1"`` or ``"f1_macro"`` scoring.
        k_folds: Number of cross-validation folds.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (best pipeline params, best model-only params,
        best CV F1, search time in seconds).
    """
    scoring_name = "f1" if n_classes == 2 else "f1_macro"
    # scoring_name = "roc_auc" if n_classes == 2 else "roc_auc_ovr"

    cv_splitter = StratifiedKFold(
        n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    xgb_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        cv=cv_splitter,
        n_trials=n_trials,
        scoring=scoring_name,
        refit=True,
        random_state=RANDOM_STATE,
        study=study,
        n_jobs=20,
        verbose=1,
        return_train_score=False,
    )

    start = perf_counter()
    xgb_search.fit(x, y)
    end = perf_counter()

    best_search_params = xgb_search.best_params_
    best_model_params = {
        k.replace("model__", ""): v for k, v in best_search_params.items()
    }
    best_cv_f1 = xgb_search.best_score_

    return best_search_params, best_model_params, best_cv_f1, end - start


def final_fit_clf(
    preprocessor: ColumnTransformer,
    best_search_params: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    device: str = "cpu",
) -> tuple[Pipeline, float]:
    """Rebuild and fit the classification Pipeline with best params.

    Args:
        preprocessor: ColumnTransformer to embed in the new pipeline.
        best_search_params: Params from ``run_search_clf`` keyed as ``"model__<param>"``.
        x: Full feature DataFrame.
        y: Full target label array.
        n_classes: Number of target classes.
        device: XGBoost device string.

    Returns:
        Tuple of (fitted Pipeline, fit time in seconds).
    """
    pipeline = build_pipeline_clf(preprocessor, n_classes, device=device, n_jobs=-1)
    pipeline.set_params(**best_search_params)

    start = perf_counter()
    pipeline.fit(x, y)
    end = perf_counter()

    return pipeline, end - start


def evaluate_clf(
    pipeline: Pipeline, x: pd.DataFrame, y: np.ndarray, n_classes: int
) -> dict[str, Any]:
    """Predict with a fitted classification Pipeline and compute metrics.

    Args:
        pipeline: Fitted classification Pipeline.
        x: Feature DataFrame.
        y: True label array.
        n_classes: Number of target classes; selects binary vs. macro averaging.

    Returns:
        Dict with keys ``"accuracy"``, ``"auc_roc"``, ``"f1"``, ``"precision"``,
        ``"recall"``, ``"infer_time_per_sample_s"``, and ``"y_prob"``.
    """
    start = perf_counter()
    y_pred = pipeline.predict(x)
    y_prob = pipeline.predict_proba(x)
    end = perf_counter()

    avg_method = "binary" if n_classes == 2 else "macro"

    accuracy = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average=avg_method))
    precision = float(precision_score(y, y_pred, average=avg_method))
    recall = float(recall_score(y, y_pred, average=avg_method))

    if n_classes == 2:
        auc = float(roc_auc_score(y, y_prob[:, 1]))
    else:
        auc = float(roc_auc_score(y, y_prob, multi_class="ovr", average="macro"))

    total_time = end - start

    return {
        "accuracy": accuracy,
        "auc_roc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "infer_time_per_sample_s": total_time / len(y),
        "y_prob": y_prob,
    }


def export_results_clf(
    data_name: str,
    pipeline: Pipeline,
    best_model_params: dict[str, Any],
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    fit_time: float,
    search_time: float,
    best_cv_f1: float,
    y_test: np.ndarray,
    class_names: list[str],
) -> None:
    """Save XGBoost classification model, results, and diagnostic plots to disk.

    Writes ``xgboost.joblib``, ``xgboost_pipeline.joblib``, ``xgboost.json``,
    ROC curve, confusion matrix, and classification report under the appropriate
    dataset subdirectories.

    Args:
        data_name: Dataset name used as the output subdirectory.
        pipeline: Fitted classification Pipeline.
        best_model_params: Best hyperparameters without the ``"model__"`` prefix.
        train_metrics: Metrics dict from ``evaluate_clf`` on the training set.
        test_metrics: Metrics dict from ``evaluate_clf`` on the test set.
        fit_time: Final model fit time in seconds.
        search_time: Hyperparameter search time in seconds.
        best_cv_f1: Best cross-validation F1 from the search phase.
        y_test: True test labels, used for diagnostic plots.
        class_names: Human-readable class names for plot labels.
    """
    # --- model artefacts ---
    models_dir = MODELS_DIR / data_name
    models_dir.mkdir(parents=True, exist_ok=True)

    model = pipeline.named_steps["model"]
    joblib.dump(model, models_dir / "xgboost.joblib")
    joblib.dump(pipeline, models_dir / "xgboost_pipeline.joblib")

    print(f"\nModel saved to {models_dir / 'xgboost.joblib'}")
    print(f"Pipeline saved to {models_dir / 'xgboost_pipeline.joblib'}")

    # --- results JSON ---
    results_dir = RESULTS_BENCHMARK / data_name
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "accuracy": test_metrics["accuracy"],
        "auc_roc": test_metrics["auc_roc"],
        "fit_time_s": fit_time,
        "infer_time_per_sample_s": test_metrics["infer_time_per_sample_s"],
        "best_params": best_model_params,
        "add_test_metric": {
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
        },
        "add_train_metric": {
            "accuracy": train_metrics["accuracy"],
            "auc_roc": train_metrics["auc_roc"],
            "f1": train_metrics["f1"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "infer_time_per_sample_s": train_metrics["infer_time_per_sample_s"],
            "search_time_s": search_time,
            "best_cv_f1": float(best_cv_f1),
        },
    }

    with open(results_dir / "xgboost.json", "w") as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {results_dir / 'xgboost.json'}")

    # --- classification-specific plots / reports ---
    y_test_prob = test_metrics["y_prob"]

    plot_roc_curve(
        y_test,
        y_test_prob,
        save_path=results_dir / "roc_curve_xgboost.png",
        class_names=list(class_names),
    )
    plot_confusion_matrix(
        y_test,
        y_test_prob,
        save_path=results_dir / "confusion_matrix_xgboost.png",
        class_names=list(class_names),
    )
    save_classification_report(
        y_test,
        y_test_prob,
        save_path=results_dir / "classification_report_xgboost.json",
        class_names=list(class_names),
    )
