# %%
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from benchmark.config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE, RESULTS
from benchmark.models.xrfm_model import build_categorical_info

torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

dataset = "abalone"
processed = DATA_PROCESSED / dataset
model = joblib.load(MODELS_DIR / dataset / "xrfm.joblib")
pipeline = joblib.load(MODELS_DIR / dataset / "xrfm_pipeline.joblib")

print(model.get_state_dict())

# %%
with open(processed / "metadata.json") as f:
    metadata = json.load(f)

n_num_cols = len(metadata["num_cols"])

print(metadata)


# %%
train_df = pd.read_csv(processed / "train.csv")
train_df.head()


# %%
X_train = train_df.drop(columns=metadata["target_col"])
X_train = pipeline.transform(X_train)

y_train = train_df[metadata["target_col"]].to_numpy().astype(np.float32)


# %%
test_df = pd.read_csv(processed / "test.csv")
test_df.head()


# %%
X_test = test_df.drop(columns=[metadata["target_col"]])
X_test = pipeline.transform(X_test)

y_test = test_df[metadata["target_col"]].to_numpy().astype(np.float32)


# %%
cat_info = (
    build_categorical_info(pipeline, n_num_cols)
    if n_num_cols < X_train.shape[1]
    else None
)
print(cat_info)


# %%
raw_names = pipeline.named_steps["processor"].get_feature_names_out()
feature_names = [n.replace("scaler__", "").replace("ohe__", "") for n in raw_names]
print(feature_names)

# %%
# AGOP
all_Ms = model.collect_Ms()

plt.figure(figsize=(8, 7))

sns.heatmap(
    all_Ms[0].cpu().numpy(),
    xticklabels=feature_names,
    yticklabels=feature_names,
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Interaction Strength"},
)

plt.title("AGOP")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
agop_diag = np.diag(all_Ms[0].cpu().numpy())
print(agop_diag)


# %%
print("AGOP diagonal (per-feature importance):")
for name, val in sorted(zip(feature_names, agop_diag), key=lambda x: -x[1]):
    print(f"{name:20s}: {val:.4f}")


# %%
# PCA
n_components = X_train.shape[1]
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
pca.fit(X_train)

# %%
print("Explained variance ratio per PC:")
for i, r in enumerate(pca.explained_variance_ratio_):
    print(
        f"PC{i + 1}: {r:.3f}  (cumulative: {pca.explained_variance_ratio_[: i + 1].sum():.3f})"
    )


# %%
print(pca.components_)  # loadings

# %%
pca_importance = (np.abs(pca.components_) * pca.explained_variance_ratio_[:, None]).sum(
    axis=0
)

print("PCA feature importance (variance-weighted loadings):")
for name, val in sorted(zip(feature_names, pca_importance), key=lambda x: -x[1]):
    print(f"{name:20s}: {val:.4f}")


# %%
# Mutual Information
mi_scores = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)

print("Mutual Information scores:")
for name, val in sorted(zip(feature_names, mi_scores), key=lambda x: -x[1]):
    print(f"  {name:20s}: {val:.4f}")


# %%
# Permutation Importance
class XRFMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X)


result = permutation_importance(
    XRFMWrapper(model),
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="neg_root_mean_squared_error",
)

perm_mean = result.importances_mean
perm_std = result.importances_std

# %%
print("Permutation Importance:")
for name, val, std in sorted(
    zip(feature_names, perm_mean, perm_std), key=lambda x: -x[1]
):
    print(f"{name:20s}: {val:.4f}  ±{std:.4f}")


# %%
# Report
def min_max_scaler(x: np.ndarray):
    min_val, max_val = x.min(), x.max()
    return (
        (x - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(x)
    )


comparison = pd.DataFrame(
    {
        "feature": feature_names,
        "agop": min_max_scaler(agop_diag),
        "pca": min_max_scaler(pca_importance),
        "mi": min_max_scaler(mi_scores),
        "perm": min_max_scaler(perm_mean),
    }
)

for col in ["agop", "pca", "mi", "perm"]:
    comparison[f"{col}_rank"] = comparison[col].rank(ascending=False).astype(int)

report_table = pd.DataFrame(
    {
        "Feature": feature_names,
        "AGOP": agop_diag.round(4),
        "AGOP Rank": comparison["agop_rank"],
        "PCA": pca_importance.round(4),
        "PCA Rank": comparison["pca_rank"],
        "MI": mi_scores.round(4),
        "MI Rank": comparison["mi_rank"],
        "Perm Imp": perm_mean.round(4),
        "Perm Rank": comparison["perm_rank"],
    }
).sort_values("AGOP Rank")

print(report_table.to_string(index=False))

# %%
report_table.to_csv(
    RESULTS / "03_interpretability" / "interpretability.csv", index=False
)


# %%
norm_df = comparison.sort_values("agop", ascending=True)

width = 0.2
method_colors = {
    "agop": "#2196F3",
    "pca": "#4CAF50",
    "mi": "#FF9800",
    "perm": "#E91E63",
}
method_labels = {
    "agop": "AGOP Diagonal",
    "pca": "PCA (variance-weighted)",
    "mi": "Mutual Information",
    "perm": "Permutation Importance",
}
methods = ["mi", "pca", "perm", "agop"]
colors = [method_colors[m] for m in methods]

fig, ax = plt.subplots(figsize=(9, 6))
for i, (col, color) in enumerate(zip(methods, colors)):
    ax.barh(
        np.arange(len(feature_names)) + i * width,
        norm_df[col],
        width,
        label=method_labels[col],
        color=color,
        alpha=0.85,
    )

ax.set_yticks(np.arange(len(feature_names)) + width * 1.5)
ax.set_yticklabels(norm_df["feature"])
ax.set_xlabel("Normalised Importance")
ax.set_title("Feature Importance (Abalone)")
ax.legend()
plt.tight_layout()
plt.savefig(
    RESULTS / "03_interpretability" / "importance_bar.png", dpi=150, bbox_inches="tight"
)
plt.show()


# %%
rank_df = comparison[
    ["feature", "agop_rank", "perm_rank", "mi_rank", "pca_rank"]
].copy()
rank_df.columns = ["feature", "AGOP", "Perm Imp", "MI", "PCA"]
methods_order = ["AGOP", "Perm Imp", "MI", "PCA"]

fig, ax = plt.subplots(figsize=(8, 6))

for _, row in rank_df.iterrows():
    ranks = [row[m] for m in methods_order]
    ax.plot(methods_order, ranks, marker="o", linewidth=2, markersize=18)

    for method, rank in zip(methods_order, ranks):
        ax.text(
            method,
            rank,
            str(rank),
            va="center",
            ha="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    ax.text(-0.25, row["AGOP"], row["feature"], va="center", ha="right", fontsize=9)
    ax.text(3.25, row["PCA"], row["feature"], va="center", ha="left", fontsize=9)

ax.set_ylim(len(feature_names) + 0.5, 0.5)
ax.set_yticks([])
ax.set_title("Feature Importance Rank – Bump Chart (Abalone)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(
    RESULTS / "03_interpretability" / "bump_chart.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %%
