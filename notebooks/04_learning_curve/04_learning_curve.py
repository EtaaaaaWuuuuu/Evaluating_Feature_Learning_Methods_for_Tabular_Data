# %%
# Import
import json
import time
import traceback
import warnings
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from benchmark.config import (
    DATA_PROCESSED,
    RANDOM_STATE,
    RESULTS,
    RESULTS_BENCHMARK,
    TEST_SIZE,
)

# xRFM helpers
from benchmark.models.xrfm_model import make_processor, build_categorical_info

from sklearn.base import clone
from xrfm import xRFM

# XGBoost helpers (shared model module)
from benchmark.models.xgboost_model import (
    build_preprocessor as build_xgb_preprocessor,
    build_pipeline_clf as build_xgb_pipeline_clf,
)

# TabNet helpers (shared model module)
from tabpfn import TabPFNClassifier

# Random Forest helpers (shared model module)
from benchmark.models.random_forest_model import build_model as build_rf_model

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# %%
# ===================================================================================
# Split data into stratified subsamples of different sizes, and save to new CSV files
# ===================================================================================
# Paths
# Data source and used is sleep health
source_dir = DATA_PROCESSED / "sleep_health"
output_dir = DATA_PROCESSED / "subsampled_sleep_health"
output_dir.mkdir(parents=True, exist_ok=True)

# Load
train_df = pd.read_csv(source_dir / "train.csv")
test_df = pd.read_csv(source_dir / "test.csv")

with open(source_dir / "metadata.json") as f:
    metadata = json.load(f)

TARGET_COLUMN = metadata["target_col"]

print(f"Full train size : {len(train_df)}")
print(f"Test size       : {len(test_df)}")
print(f"Target column   : {TARGET_COLUMN}")
print(f"Class distribution:\n{train_df[TARGET_COLUMN].value_counts()}")

# %%
# Subsample
SUBSAMPLE_SIZES = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000]

for n in SUBSAMPLE_SIZES:
    if n >= len(train_df):
        # Full training set – just copy it
        sub_df = train_df.copy()
        print(f"[{n:>5}] full train set ({len(sub_df)} rows)")
    else:
        # Stratified subsample: keep n rows, discard the rest
        sub_df, _ = train_test_split(
            train_df,
            train_size=n,
            random_state=RANDOM_STATE,
            stratify=train_df[TARGET_COLUMN],
        )
        sub_df = sub_df.reset_index(drop=True)
        print(f"[{n:>5}] sampled {len(sub_df)} rows")

    out_path = output_dir / f"{n}sub_train.csv"
    sub_df.to_csv(out_path, index=False)

# %%
# Copy test and metadataa
shutil.copy2(source_dir / "test.csv", output_dir / "test.csv")
shutil.copy2(source_dir / "metadata.json", output_dir / "metadata.json")

print(f"\nAll files saved to {output_dir}")
print("Contents:")
for p in sorted(output_dir.iterdir()):
    print(f"  {p.name}")

# %%
# ===================================================================================
# Plot learning curves of xRFM, XGBoost, TabNet, and Random Forest on the subsamples
# ===================================================================================

# Path
SUBSAMPLE_DIR = DATA_PROCESSED / "subsampled_sleep_health"
RESULTS_DIR = RESULTS / "04_learning_curve"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Where the per-model optimal-hyperparameter JSONs live
HPARAM_DIR = RESULTS_BENCHMARK / "sleep_health"

SUBSAMPLE_SIZES = [500, 1000, 2000, 4000, 6000, 8000, 12000, 16000]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device       : {DEVICE}")
print(f"Subsample dir: {SUBSAMPLE_DIR}")
print(f"Results dir  : {RESULTS_DIR}")

# %%
# Metadata & test data
with open(SUBSAMPLE_DIR / "metadata.json") as f:
    metadata = json.load(f)

TARGET_COL = metadata["target_col"]
NUM_COLS = metadata["num_cols"]
CAT_COLS = metadata["cat_cols"]
BIN_COLS = metadata["bin_cols"]

test_df = pd.read_csv(SUBSAMPLE_DIR / "test.csv")
print(f"Test set size: {len(test_df)}")
print(f"Target       : {TARGET_COL}")

# Shared label encoder (fitted on test set classes – same across all splits)
le = LabelEncoder()
le.fit(test_df[TARGET_COL])
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {list(le.classes_)}")

# all results
all_results: dict[str, dict] = {}


# %%
# Helper functions for loading subsamples, computing metrics, and plotting learning curves
def load_subsample(n: int) -> pd.DataFrame:
    """Load the subsampled training CSV for size *n*."""
    path = SUBSAMPLE_DIR / f"{n}sub_train.csv"
    return pd.read_csv(path)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Return accuracy and macro AUC-ROC from integer labels + proba matrix."""
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {"test_acc": float(acc), "test_auc_roc": float(auc)}


def plot_learning_curves(
    all_results: dict,
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """Plot one learning-curve chart for a single metric across models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("deep", n_colors=len(all_results))
    for i, (model_name, size_dict) in enumerate(all_results.items()):
        sizes = sorted(int(k) for k in size_dict)
        values = [size_dict[str(s)][metric_key] for s in sizes]
        ax.plot(
            sizes, values, marker="o", linewidth=2, color=palette[i], label=model_name
        )
    ax.set_xlabel("Training subsample size", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(SUBSAMPLE_SIZES)
    ax.set_xticklabels([str(s) for s in SUBSAMPLE_SIZES], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# %%
# XRFM
try:
    with open(HPARAM_DIR / "xrfm.json") as f:
        xrfm_hparams = json.load(f)
    best_params = xrfm_hparams["best_params"]
    print(f"Best params: {best_params}")

    rfm_params = {
        "model": best_params["model"],
        "fit": best_params["fit"],
    }
    XRFM_MAX_LEAF_SIZE = 60_000
    n_num_cols = len(NUM_COLS)

    xrfm_results: dict[str, dict] = {}

    for n in SUBSAMPLE_SIZES:
        print(f"  n={n:>5} ... ", end="", flush=True)
        sub_df = load_subsample(n)

        X_sub = sub_df.drop(columns=TARGET_COL)
        y_sub = le.transform(sub_df[TARGET_COL]).astype(np.int32)

        # Internal train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_sub,
            y_sub,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Preprocessing
        processor = make_processor(metadata)
        pipeline = Pipeline(steps=[("processor", clone(processor))])
        X_train_t = pipeline.fit_transform(X_train)
        X_val_t = pipeline.transform(X_val)

        # Categorical feature info for xRFM
        cat_info = (
            build_categorical_info(pipeline, n_num_cols)
            if n_num_cols < X_train_t.shape[1]
            else None
        )

        # Construct and fit xRFM directly
        model = xRFM(
            rfm_params=rfm_params,
            device=DEVICE,
            tuning_metric="f1",
            split_method="top_vector_agop_on_subset",
            categorical_info=cat_info,
            max_leaf_size=XRFM_MAX_LEAF_SIZE,
        )
        t0 = time.perf_counter()
        model.fit(X_train_t, y_train, X_val_t, y_val)
        fit_time = time.perf_counter() - t0

        # Predict on test set
        X_test = test_df.drop(columns=TARGET_COL)
        y_test = le.transform(test_df[TARGET_COL]).astype(np.int32)
        X_test_t = pipeline.transform(X_test)
        y_proba = model.predict_proba(X_test_t)

        metrics = compute_metrics(y_test, y_proba)
        metrics["fit_time_s"] = float(fit_time)
        metrics["fit_time_persample_s"] = float(fit_time / n)
        xrfm_results[str(n)] = metrics
        print(
            f"acc={metrics['test_acc']:.4f}  auc={metrics['test_auc_roc']:.4f}  "
            f"fit={fit_time:.2f}s"
        )

    all_results["xRFM"] = xrfm_results
    print("xRFM done.\n")

except Exception as e:
    print(f"\n  [SKIP] xRFM failed: {e}")
    traceback.print_exc()

# %%
# XGBoost
try:
    # Load best hyperparameters
    with open(HPARAM_DIR / "xgboost.json") as f:
        xgb_hparams = json.load(f)
    best_params = xgb_hparams["best_params"]
    print(f"Best params: {best_params}")

    # Add "model" prefix so pipeline.set_params() can route them
    best_search_params = {f"model__{k}": v for k, v in best_params.items()}

    xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
    xgb_results: dict[str, dict] = {}

    for n in SUBSAMPLE_SIZES:
        print(f"  n={n:>5} ... ", end="", flush=True)
        sub_df = load_subsample(n)

        X_train = sub_df.drop(columns=TARGET_COL)
        y_train = le.transform(sub_df[TARGET_COL]).astype(np.int32)
        X_test = test_df.drop(columns=TARGET_COL)
        y_test = le.transform(test_df[TARGET_COL]).astype(np.int32)

        # Use shared helpers from xgboost_model.py
        preprocessor = build_xgb_preprocessor(metadata)
        pipe = build_xgb_pipeline_clf(
            preprocessor,
            n_classes,
            device=xgb_device,
            n_jobs=-1,
        )
        pipe.set_params(**best_search_params)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_proba = pipe.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_proba)
        metrics["fit_time_s"] = float(fit_time)
        metrics["fit_time_persample_s"] = float(fit_time / n)
        xgb_results[str(n)] = metrics
        print(
            f"acc={metrics['test_acc']:.4f}  auc={metrics['test_auc_roc']:.4f}  "
            f"fit={fit_time:.2f}s"
        )

    all_results["XGBoost"] = xgb_results
    print("XGBoost done.\n")

except Exception as e:
    print(f"\n  [SKIP] XGBoost failed: {e}")
    traceback.print_exc()

# %%
# TabPFN
try:
    # Load best hyperparameters
    with open(HPARAM_DIR / "tabpfn.json") as f:
        tabpfn_hparams = json.load(f)
    tabpfn_params = tabpfn_hparams.get("tabpfn_params", {})
    n_estimators = tabpfn_params.get("n_estimators", 2)
    print(f"TabPFN params: n_estimators={n_estimators}")

    # Feature columns in the order: num, cat, bin
    feature_cols = NUM_COLS + CAT_COLS + BIN_COLS
    if len(NUM_COLS) < len(feature_cols):
        cat_indices = list(range(len(NUM_COLS), len(feature_cols)))
    else:
        cat_indices = None

    tabpfn_results: dict[str, dict] = {}

    for n in SUBSAMPLE_SIZES:
        print(f"  n={n:>5} ... ", end="", flush=True)
        sub_df = load_subsample(n)

        X_train = sub_df[feature_cols].copy()
        y_train = le.transform(sub_df[TARGET_COL]).astype(np.int32)
        X_test = test_df[feature_cols].copy()
        y_test = le.transform(test_df[TARGET_COL]).astype(np.int32)

        clf = TabPFNClassifier(
            categorical_features_indices=cat_indices,
            device="auto",
            ignore_pretraining_limits=True,
            random_state=RANDOM_STATE,
            n_estimators=n_estimators,
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_proba = clf.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_proba)
        metrics["fit_time_s"] = float(fit_time)
        metrics["fit_time_persample_s"] = float(fit_time / n)
        tabpfn_results[str(n)] = metrics
        print(
            f"acc={metrics['test_acc']:.4f}  auc={metrics['test_auc_roc']:.4f}  "
            f"fit={fit_time:.2f}s"
        )

    all_results["TabPFN"] = tabpfn_results
    print("TabPFN done.\n")

except Exception as e:
    print(f"\n  [SKIP] TabPFN failed: {e}")
    traceback.print_exc()

# %%
# Random Forest
try:
    # Load best hyperparameters
    rf_json_path = HPARAM_DIR / "random_forest.json"
    if not rf_json_path.exists():
        raise FileNotFoundError(
            f"{rf_json_path} not found – Random Forest will be skipped."
        )

    with open(rf_json_path) as f:
        rf_hparams = json.load(f)
    best_params = rf_hparams["best_params"]
    print(f"Best params: {best_params}")

    # Add fixed params matching random_forest_model.py conventions
    rf_params = {
        **best_params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Preprocessor matching rf_sleep_health.py
    def build_rf_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("num", "passthrough", NUM_COLS),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
                ("bin", OrdinalEncoder(), BIN_COLS),
            ],
            remainder="passthrough",
        )

    rf_results: dict[str, dict] = {}

    for n in SUBSAMPLE_SIZES:
        print(f"  n={n:>5} ... ", end="", flush=True)
        sub_df = load_subsample(n)

        X_train = sub_df.drop(columns=TARGET_COL)
        y_train = le.transform(sub_df[TARGET_COL]).astype(np.int32)
        X_test = test_df.drop(columns=TARGET_COL)
        y_test = le.transform(test_df[TARGET_COL]).astype(np.int32)

        # Use build_model helper from random_forest_model.py
        rf_model = build_rf_model(task="classification", rf_params=rf_params)
        preprocessor = build_rf_preprocessor()
        pipe = Pipeline([("processor", preprocessor), ("model", rf_model)])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_proba = pipe.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_proba)
        metrics["fit_time_s"] = float(fit_time)
        metrics["fit_time_persample_s"] = float(fit_time / n)
        rf_results[str(n)] = metrics
        print(
            f"acc={metrics['test_acc']:.4f}  auc={metrics['test_auc_roc']:.4f}  "
            f"fit={fit_time:.2f}s"
        )

    all_results["Random Forest"] = rf_results
    print("Random Forest done.\n")

except Exception as e:
    print(f"\n  [SKIP] Random Forest failed: {e}")
    traceback.print_exc()

# %%
# Plot learning curves
plot_learning_curves(
    all_results,
    metric_key="test_acc",
    ylabel="Accuracy",
    title="Learning Curve – Accuracy vs Training Size",
    save_path=RESULTS_DIR / "learning_curve_sleep_health_accuracy.png",
)

plot_learning_curves(
    all_results,
    metric_key="test_auc_roc",
    ylabel="AUC-ROC",
    title="Learning Curve – AUC-ROC vs Training Size",
    save_path=RESULTS_DIR / "learning_curve_sleep_health_auc_roc.png",
)

plot_learning_curves(
    all_results,
    metric_key="fit_time_s",
    ylabel="Training Time (seconds)",
    title="Learning Curve – Training Time vs Training Size",
    save_path=RESULTS_DIR / "learning_curve_sleep_health_fit_time.png",
)

# %%
# Save all results to JSON for later analysis
json_path = RESULTS_DIR / "learning_curve_sleep_health.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nResults JSON saved to {json_path}")
print(f"Models included: {list(all_results.keys())}")
