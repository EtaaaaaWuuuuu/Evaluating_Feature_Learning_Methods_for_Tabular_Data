import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.preprocessing import label_binarize


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4)}


def evaluate_classification(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = y_proba.argmax(axis=1)
    n_classes = y_proba.shape[1]
    results = {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}

    if n_classes == 2:
        results["auc_roc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
    else:
        results["auc_roc"] = round(
            float(roc_auc_score(y_true, y_proba, multi_class="ovr")), 4
        )

    return results


def _get_labels(n_classes: int, class_names: list[str] | None) -> list[str]:
    return (
        class_names if class_names is not None else [str(i) for i in range(n_classes)]
    )


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    n_classes = y_proba.shape[1]
    labels = _get_labels(n_classes, class_names)

    fig, ax = plt.subplots(figsize=(7, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        ax.plot(fpr, tpr, label=f"{labels[1]} (AUC = {auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            ax.plot(
                fpr,
                tpr,
                alpha=0.4,
                lw=1,
                label=f"{labels[i]} (AUC = {roc_auc_score(y_bin[:, i], y_proba[:, i]):.3f})",
            )
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr = mean_tpr / n_classes
        macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
        ax.plot(
            all_fpr,
            mean_tpr,
            color="black",
            lw=2,
            linestyle="--",
            label=f"Macro avg (AUC = {macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    y_pred = y_proba.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = _get_labels(n_classes, class_names)

    fig, ax = plt.subplots(figsize=(max(5, n_classes), max(4, n_classes - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    thresh = (cm.min() + cm.max()) / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path,
    class_names: list[str] | None = None,
) -> None:
    y_pred = y_proba.argmax(axis=1)
    labels = _get_labels(y_proba.shape[1], class_names)

    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    rounded = {
        cls: (
            round(metrics, 4)
            if isinstance(metrics, float)
            else {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
            }
        )
        for cls, metrics in report.items()
    }
    with open(save_path, "w") as f:
        json.dump(rounded, f, indent=4)
