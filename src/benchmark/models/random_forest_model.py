import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from benchmark.config import N_TRIALS, NUM_FOLDS, RANDOM_STATE, TEST_SIZE
from benchmark.models.search_spaces import SEARCH_SPACES


def build_rf_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample hyperparameters for a Random Forest from an Optuna trial.

    Args:
        trial: Optuna trial used to suggest each hyperparameter value.

    Returns:
        Dictionary of scikit-learn RandomForest constructor kwargs.
    """
    s = SEARCH_SPACES["random_forest"]
    return {
        "n_estimators": trial.suggest_distribution("n_estimators", s["n_estimators"]),
        "max_depth": trial.suggest_distribution("max_depth", s["max_depth"]),
        "min_samples_split": trial.suggest_distribution(
            "min_samples_split", s["min_samples_split"]
        ),
        "min_samples_leaf": trial.suggest_distribution(
            "min_samples_leaf", s["min_samples_leaf"]
        ),
        "max_features": trial.suggest_distribution("max_features", s["max_features"]),
        "bootstrap": trial.suggest_distribution("bootstrap", s["bootstrap"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def get_cv_splitter(task: str) -> StratifiedKFold | KFold:
    """Return the appropriate CV splitter for the given task.

    Args:
        task: ``"classification"`` uses StratifiedKFold; anything else uses KFold.

    Returns:
        Configured CV splitter instance.
    """
    if task == "classification":
        return StratifiedKFold(
            n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
    return KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_model(
    task: str, rf_params: dict[str, Any]
) -> RandomForestRegressor | RandomForestClassifier:
    if task == "regression":
        return RandomForestRegressor(**rf_params)
    return RandomForestClassifier(**rf_params)


def evaluate_fold(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute a single fold's evaluation metric.

    Args:
        task: ``"regression"`` returns MSE; ``"classification"`` returns macro F1.
        y_true: Ground-truth labels or values.
        y_pred: Predicted labels or values.

    Returns:
        Scalar score for the fold.
    """
    if task == "regression":
        return mean_squared_error(y_true, y_pred)
    return f1_score(y_true, y_pred, average="macro")


def run_rf_trial(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    processor: ColumnTransformer,
    task: str,
) -> float:
    """Run one Optuna trial of Random Forest cross-validation.

    Args:
        trial: Optuna trial supplying hyperparameter suggestions.
        X: Feature DataFrame (will be CV-split internally).
        y: Target array.
        processor: Unfitted ColumnTransformer; cloned fresh per fold.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        Mean fold score (MSE for regression, macro F1 for classification).
    """
    rf_params = build_rf_params(trial)
    cv = get_cv_splitter(task)
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y if task == "classification" else None):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline(steps=[("processor", clone(processor))])
        X_train = pipeline.fit_transform(X_train)
        X_val = pipeline.transform(X_val)

        model = build_model(task, rf_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = evaluate_fold(task, y_val, y_pred)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


def fit_random_forest_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    processor: ColumnTransformer,
    task: str,
    n_trials: int = N_TRIALS,
) -> optuna.trial.FrozenTrial:
    """Search Random Forest hyperparameters with Optuna cross-validation.

    Args:
        X: Feature DataFrame.
        y: Target array.
        processor: Unfitted ColumnTransformer applied inside each fold.
        task: ``"regression"`` or ``"classification"``.
        n_trials: Number of Optuna trials to run.

    Returns:
        Best Optuna trial, including its params and CV score.
    """
    direction = "minimize" if task == "regression" else "maximize"
    study = optuna.create_study(direction=direction)

    study.optimize(
        lambda trial: run_rf_trial(
            trial=trial,
            X=X,
            y=y,
            processor=processor,
            task=task,
        ),
        n_trials=n_trials,
    )

    return study.best_trial


def build_best_rf_params(best_trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Extract best-trial params as RandomForest constructor kwargs.

    Args:
        best_trial: Completed Optuna trial containing the winning hyperparameters.

    Returns:
        Dictionary of scikit-learn RandomForest constructor kwargs.
    """
    p = best_trial.params
    return {
        "n_estimators": p["n_estimators"],
        "max_depth": p["max_depth"],
        "min_samples_split": p["min_samples_split"],
        "min_samples_leaf": p["min_samples_leaf"],
        "max_features": p["max_features"],
        "bootstrap": p["bootstrap"],
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def fit_best_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    best_trial: optuna.trial.FrozenTrial,
    processor: ColumnTransformer,
    task: str,
) -> tuple[RandomForestRegressor | RandomForestClassifier, Pipeline, float]:
    """Fit a Random Forest using the best Optuna hyperparameters.

    Args:
        X: Full feature DataFrame.
        y: Full target array.
        best_trial: Optuna trial with the best hyperparameters.
        processor: Unfitted ColumnTransformer to embed in the pipeline.
        task: ``"regression"`` or ``"classification"``.

    Returns:
        Tuple of (fitted model, fitted pipeline, fit time in seconds).
    """
    rf_params = build_best_rf_params(best_trial)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if task == "classification" else None,
    )

    pipeline = Pipeline(steps=[("processor", clone(processor))])
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)

    model = build_model(task, rf_params)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    return model, pipeline, fit_time
