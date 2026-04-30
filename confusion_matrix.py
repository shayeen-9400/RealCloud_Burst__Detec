"""
confusion_matrix.py
-------------------
Generates and saves a styled confusion matrix for the trained
cloudburst detection model on the TEST set.

Usage:
    python confusion_matrix.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset_loader import get_dataloaders
from model import CloudburstFusionModel

MODEL_PATH  = "cloudburst_model.pth"
OUTPUT_PATH = "confusion_matrix.png"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
#  Run inference on the test loader
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    for images, tabular, labels in test_loader:
        images  = images.to(DEVICE)
        tabular = tabular.to(DEVICE)
        logits  = model(images, tabular)
        probs   = torch.sigmoid(logits)
        preds   = (probs >= 0.5).long().cpu()
        all_preds.append(preds)
        all_labels.append(labels.long().cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ──────────────────────────────────────────────────────────────
#  Build 2x2 confusion matrix manually
# ──────────────────────────────────────────────────────────────
def build_cm(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return np.array([[tn, fp],
                     [fn, tp]])   # rows = actual, cols = predicted


# ──────────────────────────────────────────────────────────────
#  Compute classification metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total     = tn + fp + fn + tp
    accuracy  = (tp + tn) / (total + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return dict(Accuracy=accuracy, Precision=precision,
                Recall=recall, F1=f1, Specificity=specificity,
                TP=tp, TN=tn, FP=fp, FN=fn)


# ──────────────────────────────────────────────────────────────
#  Draw the confusion matrix
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, metrics):
    labels  = ["No Cloudburst\n(Label 0)", "Cloudburst\n(Label 1)"]
    n_total = cm.sum()

    # Colour map: deep blue for correct, coral for errors
    colours = np.array([
        ["#2176AE", "#E84855"],   # TN (correct), FP (error)
        ["#E84855", "#2176AE"],   # FN (error),   TP (correct)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#0F1117")

    # ── Left: Confusion matrix grid ────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0F1117")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

    for row in range(2):
        for col in range(2):
            val   = cm[row, col]
            pct   = val / n_total * 100
            color = colours[row][col]
            rect  = plt.Rectangle([col - 0.5, row - 0.5], 1, 1,
                                   facecolor=color, edgecolor="#0F1117",
                                   linewidth=3, alpha=0.90)
            ax.add_patch(rect)

            # Count
            ax.text(col, row - 0.10, str(val),
                    ha="center", va="center",
                    fontsize=36, fontweight="bold", color="white")
            # Percentage
            ax.text(col, row + 0.22, f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=13, color="white", alpha=0.85)

            # Cell label (TP/TN/FP/FN)
            cell_lbl = [["TN", "FP"], ["FN", "TP"]][row][col]
            ax.text(col, row + 0.42, f"({cell_lbl})",
                    ha="center", va="center",
                    fontsize=10, color="white", alpha=0.65)

    # Axes labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, color="white")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=11, color="white", rotation=90, va="center")
    ax.tick_params(colors="white", length=0)
    ax.set_xlabel("Predicted Label", fontsize=13, color="#A0AEC0", labelpad=12)
    ax.set_ylabel("Actual Label",    fontsize=13, color="#A0AEC0", labelpad=12)
    ax.set_title("Confusion Matrix\n(Test Set)", fontsize=15,
                 fontweight="bold", color="white", pad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Right: Metrics panel ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0F1117")
    ax2.axis("off")

    metric_rows = [
        ("Accuracy",    metrics["Accuracy"],    "#48BB78"),
        ("Precision",   metrics["Precision"],   "#63B3ED"),
        ("Recall",      metrics["Recall"],      "#F6E05E"),
        ("F1 Score",    metrics["F1"],          "#FC8181"),
        ("Specificity", metrics["Specificity"], "#B794F4"),
    ]

    ax2.text(0.5, 0.97, "Performance Metrics", ha="center", va="top",
             fontsize=14, fontweight="bold", color="white",
             transform=ax2.transAxes)

    for i, (name, val, color) in enumerate(metric_rows):
        y = 0.82 - i * 0.155
        # Background pill
        rect = mpatches.FancyBboxPatch((0.05, y - 0.055), 0.90, 0.105,
                                       boxstyle="round,pad=0.01",
                                       facecolor="#1A202C", edgecolor=color,
                                       linewidth=1.5,
                                       transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(rect)
        ax2.text(0.18, y, name, ha="left", va="center",
                 fontsize=12, color="#CBD5E0", transform=ax2.transAxes)
        ax2.text(0.85, y, f"{val:.4f}", ha="right", va="center",
                 fontsize=13, fontweight="bold", color=color,
                 transform=ax2.transAxes)

    # Raw counts summary
    y_counts = 0.05
    summary = (f"TP={metrics['TP']}  TN={metrics['TN']}  "
               f"FP={metrics['FP']}  FN={metrics['FN']}  |  Total={n_total}")
    ax2.text(0.5, y_counts, summary, ha="center", va="bottom",
             fontsize=9, color="#718096", transform=ax2.transAxes)

    fig.suptitle("Cloudburst Detection  |  CNN + FNN Fusion Model",
                 fontsize=16, fontweight="bold", color="white", y=1.01)

    plt.tight_layout(pad=2.0)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Saved] Confusion matrix saved to: {OUTPUT_PATH}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
def main():
    print("[Loading] data and model...")

    _, _, test_loader, _ = get_dataloaders(batch_size=32)

    model = CloudburstFusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("[Running] inference on test set...")
    y_pred, y_true = get_predictions(model, test_loader)

    cm      = build_cm(y_true, y_pred)
    metrics = compute_metrics(cm)

    print("\n=== Confusion Matrix ===")
    print(f"  {'':20s}  Pred: No CB   Pred: CB")
    print(f"  Actual: No CB       {cm[0,0]:>6}        {cm[0,1]:>6}")
    print(f"  Actual: CB          {cm[1,0]:>6}        {cm[1,1]:>6}")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<14}: {v:.4f}  ({v*100:.2f}%)")
        else:
            print(f"  {k:<14}: {v}")

    plot_confusion_matrix(cm, metrics)


if __name__ == "__main__":
    main()
