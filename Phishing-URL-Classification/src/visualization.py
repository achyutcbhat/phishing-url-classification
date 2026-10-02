"""
visualization.py
----------------
All 5 mandatory visualizations + helper plots.

Functions:
  1. plot_kernel_accuracy_bar()       - Cross-validated kernel accuracy comparison
  2. plot_rbf_gridsearch_heatmap()    - RBF C vs Gamma heatmap
  3. plot_train_time_vs_accuracy()    - Scatter: training time vs test accuracy
  4. plot_confusion_matrix()          - Annotated confusion matrix
  5. plot_feature_importance()        - Linear SVM |coef_| horizontal bar chart
  6. plot_class_distribution()        - Original + binary class bar charts
  7. plot_poly_degree_comparison()    - Polynomial degree sensitivity table/chart

All plots are saved as high-resolution PNGs and closed after saving.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.titlesize":  14,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PALETTE = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61", "#762a83",
           "#e66101", "#4dac26", "#b8e186"]


# ---------------------------------------------------------------------------
# 1. Kernel Accuracy Bar Chart
# ---------------------------------------------------------------------------

def plot_kernel_accuracy_bar(
    model_names: list[str],
    accuracies: list[float],
    figures_dir: Path,
    title: str = "Cross-Validated / Baseline Accuracy by Kernel / Model",
) -> Path:
    """
    Bar chart of classification accuracy for each kernel/model configuration.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "viz1_kernel_accuracy_bar.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(model_names))]
    bars = ax.bar(model_names, accuracies, color=colors, width=0.6, edgecolor="white")

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{acc:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_xlabel("Model / Kernel Configuration")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0, min(accuracies) - 0.03), min(1.02, max(accuracies) + 0.04))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 2. RBF GridSearchCV Heatmap
# ---------------------------------------------------------------------------

def plot_rbf_gridsearch_heatmap(
    cv_results_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Heatmap of GridSearchCV mean_test_score for RBF kernel (C x Gamma).

    cv_results_df must contain columns:
        param_svc__C, param_svc__gamma, mean_test_score, param_svc__kernel
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "viz2_rbf_gridsearch_heatmap.png"

    rbf = cv_results_df[cv_results_df["param_svc__kernel"] == "rbf"].copy()
    if rbf.empty:
        logger.warning("No RBF rows in cv_results for heatmap.")
        return out

    rbf["C"]     = rbf["param_svc__C"].astype(str)
    rbf["Gamma"] = rbf["param_svc__gamma"].astype(str)
    rbf["Score"] = rbf["mean_test_score"].astype(float)

    pivot = rbf.pivot_table(index="C", columns="Gamma", values="Score", aggfunc="max")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "CV Accuracy"},
        annot_kws={"size": 10},
    )
    ax.set_title(
        "RBF SVM -- GridSearchCV Validation Accuracy (C x Gamma)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Gamma")
    ax.set_ylabel("C")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Training Time vs Testing Accuracy Scatter
# ---------------------------------------------------------------------------

def plot_train_time_vs_accuracy(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Scatter plot: x = Train_Time_s, y = Accuracy.
    Each point is labelled with the model name.

    results_df must have columns: Model, Train_Time_s, Accuracy
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "viz3_train_time_vs_accuracy.png"

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(results_df))]

    for i, row in results_df.iterrows():
        ax.scatter(
            row["Train_Time_s"], row["Accuracy"],
            s=100, color=colors[i], zorder=5, edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            row["Model"],
            xy=(row["Train_Time_s"], row["Accuracy"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_title(
        "Training Time vs Testing Accuracy\n(Each point = one model / kernel)",
        fontweight="bold",
    )
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Test Accuracy")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    figures_dir: Path,
    fp_fn: Optional[dict] = None,
) -> Path:
    """
    Annotated confusion matrix with TN/FP/FN/TP labels.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    out = figures_dir / f"viz4_confusion_matrix_{safe_name}.png"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = ["Legitimate\n(Predicted)", "Phishing/Malicious\n(Predicted)"]
    row_labels = ["Legitimate\n(Actual)", "Phishing/Malicious\n(Actual)"]

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        fmt="d",
        linewidths=1,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    # Custom cell annotations with quadrant labels
    cell_labels = [
        ["TN\n(True Negative)", "FP\n(False Positive)"],
        ["FN\n(False Negative)", "TP\n(True Positive)"],
    ]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100.0 * val / cm.sum()
            ax.text(
                j + 0.5, i + 0.35,
                cell_labels[i][j],
                ha="center", va="center",
                fontsize=8.5, color="grey", style="italic",
            )
            ax.text(
                j + 0.5, i + 0.65,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if val > cm.max() * 0.5 else "black",
            )

    ax.set_xticklabels(["Legitimate", "Phishing/\nMalicious"], fontsize=10)
    ax.set_yticklabels(["Legitimate", "Phishing/\nMalicious"], fontsize=10, rotation=0)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    title = f"Confusion Matrix -- {model_name}"
    if fp_fn:
        title += f"\nFPR={fp_fn['FPR']:.4f}  |  FNR={fp_fn['FNR']:.4f}"
    ax.set_title(title, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Linear SVM Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    coef: np.ndarray,
    feature_names: list[str],
    figures_dir: Path,
    top_n: int = 15,
) -> Path:
    """
    Horizontal bar chart of top N features by absolute SVM coefficient magnitude.

    Positive coefficient direction:
        Features with large positive coefficients push the decision toward
        the POSITIVE class (Malicious/Phishing=1).

    Negative coefficient direction:
        Features with large negative coefficients push the decision toward
        the NEGATIVE class (Legitimate=0).

    Why absolute magnitude?
        We rank by |coef| to find the most influential features regardless
        of direction, since both positive and negative extremes indicate
        strong influence on the decision boundary.

    Caveat: Correlation != Causation. A high coefficient means the feature
    is strongly weighted by the linear SVM for THIS dataset -- it does not
    establish a causal relationship between the feature and phishing.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "viz5_feature_importance.png"

    abs_coef = np.abs(coef)
    indices = np.argsort(abs_coef)[-top_n:]
    top_names = [feature_names[i] for i in indices]
    top_vals  = [coef[i] for i in indices]
    top_abs   = [abs_coef[i] for i in indices]

    colors = ["#d7191c" if v > 0 else "#2c7bb6" for v in top_vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        top_names, top_abs,
        color=colors, edgecolor="white", height=0.7,
    )

    for bar, val, abs_val in zip(bars, top_vals, top_abs):
        direction = "-> Phishing" if val > 0 else "-> Legitimate"
        ax.text(
            abs_val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{abs_val:.4f}  {direction}",
            va="center", fontsize=8,
        )

    ax.set_title(
        f"Top {top_n} Phishing Indicators -- Linear SVM Feature Importance\n"
        "(Absolute Coefficient Magnitude)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("|Coefficient| -- Feature Importance")
    ax.set_ylabel("Feature")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d7191c", label="Positive coef -> Phishing/Malicious"),
        Patch(facecolor="#2c7bb6", label="Negative coef -> Legitimate"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Auxiliary: Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    original_dist: pd.Series,
    binary_dist: pd.Series,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "class_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    orig_labels = {-1: "Phishing (-1)", 0: "Suspicious (0)", 1: "Legitimate (1)"}
    orig_x = [orig_labels.get(k, str(k)) for k in original_dist.index]
    axes[0].bar(orig_x, original_dist.values, color=PALETTE[:len(orig_x)], edgecolor="white")
    axes[0].set_title("Original Class Distribution", fontweight="bold")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(original_dist.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=10)

    # Binary
    bin_labels = {0: "Legitimate (0)", 1: "Malicious/\nPhishing (1)"}
    bin_x = [bin_labels.get(k, str(k)) for k in binary_dist.index]
    axes[1].bar(bin_x, binary_dist.values, color=[PALETTE[2], PALETTE[1]], edgecolor="white")
    axes[1].set_title("Binary Class Distribution (After Conversion)", fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(binary_dist.values):
        axes[1].text(i, v + 30, str(v), ha="center", fontsize=10)

    fig.suptitle("Phishing Dataset -- Class Distributions", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Auxiliary: Polynomial degree comparison bar chart
# ---------------------------------------------------------------------------

def plot_poly_degree_comparison(
    poly_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Grouped bar chart for polynomial degree sensitivity.
    poly_df must have columns: Degree, Accuracy, F1, Train_Time_s
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "poly_degree_comparison.png"

    degrees = poly_df["Degree"].astype(str).tolist()
    x = np.arange(len(degrees))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width/2, poly_df["Accuracy"], width, label="Accuracy",
                 color=PALETTE[0], alpha=0.85, edgecolor="white")
    b2 = ax1.bar(x + width/2, poly_df["F1"], width, label="F1-Score",
                 color=PALETTE[2], alpha=0.85, edgecolor="white")
    ax2.plot(x, poly_df["Train_Time_s"], marker="o", color=PALETTE[1],
             linewidth=2, markersize=8, label="Train Time (s)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Degree {d}" for d in degrees])
    ax1.set_title("Polynomial Degree Sensitivity Analysis", fontweight="bold", pad=12)
    ax1.set_xlabel("Polynomial Degree")
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Training Time (s)", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Auxiliary: Latency comparison bar chart
# ---------------------------------------------------------------------------

def plot_latency_comparison(
    latency_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "inference_latency.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(latency_df))]
    bars = ax.bar(
        latency_df["Model"], latency_df["Avg_Time_ms_per_URL"],
        color=colors, edgecolor="white",
    )
    for bar, val in zip(bars, latency_df["Avg_Time_ms_per_URL"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + latency_df["Avg_Time_ms_per_URL"].max() * 0.01,
            f"{val:.4f}",
            ha="center", fontsize=8.5, fontweight="bold",
        )
    ax.set_title(
        f"Inference Latency per URL ({latency_df['N_Samples'].iloc[0]} samples)",
        fontweight="bold",
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg Time per URL (ms)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out}")
    return out
