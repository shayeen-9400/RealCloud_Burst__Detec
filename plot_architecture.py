"""
plot_architecture.py
--------------------
Generates a professional model architecture diagram saved as:
    model_architecture.png

Usage:
    python plot_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─── Canvas Setup ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 13))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")

# ─── Colours ─────────────────────────────────────────────────────
BG       = "#0D1117"
CNN_COL  = "#1B4F8A"      # CNN branch — blue
CNN_ACC  = "#4A90E2"
FNN_COL  = "#1B5E38"      # FNN branch — green
FNN_ACC  = "#48BB78"
FUS_COL  = "#3B1F6B"      # Fusion head — purple
FUS_ACC  = "#9F7AEA"
OUT_COL  = "#7B2020"      # Output — red
OUT_ACC  = "#FC8181"
TITLE_C  = "white"
TEXT_C   = "#E2E8F0"
DIM_C    = "#A0AEC0"
ARROW_C  = "#718096"


# ─── Helper functions ─────────────────────────────────────────────
def box(ax, x, y, w, h, color, accent, label, sublabel="", radius=0.25):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor=accent,
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                fontsize=8.2, color=TEXT_C, fontweight="bold", zorder=4)
        ax.text(x, y - 0.18, sublabel, ha="center", va="center",
                fontsize=7.0, color=accent, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.2, color=TEXT_C, fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=ARROW_C):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=14), zorder=2)


def dim_label(ax, x, y, text, color=DIM_C):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="#1A202C", ec=color,
                      lw=0.8, alpha=0.85), zorder=5)


def section_header(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=color, lw=1.5),
            zorder=4)


# ═══════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(10, 12.55, "Cloudburst Detection  —  CNN + FNN Fusion Architecture",
        ha="center", va="center", fontsize=15, color="white",
        fontweight="bold", zorder=5)
ax.text(10, 12.15, "Binary Classification  |  716,961 Trainable Parameters  |  Test Accuracy: 100.00%",
        ha="center", va="center", fontsize=9.5, color=DIM_C, zorder=5)

# Divider
ax.plot([0.5, 19.5], [11.88, 11.88], color="#2D3748", lw=1.2)

# ═══════════════════════════════════════════════════════════════
#  CNN BRANCH  (left side, x=4.5)
# ═══════════════════════════════════════════════════════════════
CX = 4.5   # center-x of CNN column

section_header(ax, CX, 11.55, "  CNN Branch  ", CNN_ACC)

# Input
box(ax, CX, 11.0, 3.4, 0.52, "#1A202C", CNN_ACC,
    "Satellite Image Input", "224 × 224 × 3 (RGB)")
arrow(ax, CX, 10.74, CX, 10.38)

# Block 1
box(ax, CX, 10.12, 3.4, 0.48, CNN_COL, CNN_ACC,
    "Conv2D(32) + BN + ReLU  ×2", "MaxPool2D  |  Dropout(0.1)  →  112×112×32")
arrow(ax, CX, 9.86, CX, 9.50)

# Block 2
box(ax, CX, 9.24, 3.4, 0.48, CNN_COL, CNN_ACC,
    "Conv2D(64) + BN + ReLU  ×2", "MaxPool2D  |  Dropout(0.2)  →  56×56×64")
arrow(ax, CX, 8.98, CX, 8.62)

# Block 3
box(ax, CX, 8.36, 3.4, 0.48, CNN_COL, CNN_ACC,
    "Conv2D(128) + BN + ReLU  ×2", "MaxPool2D  |  Dropout(0.2)  →  28×28×128")
arrow(ax, CX, 8.10, CX, 7.74)

# Block 4
box(ax, CX, 7.48, 3.4, 0.48, CNN_COL, CNN_ACC,
    "Conv2D(256) + BN + ReLU", "MaxPool2D  →  14×14×256")
arrow(ax, CX, 7.22, CX, 6.86)

# GAP
box(ax, CX, 6.60, 3.4, 0.48, "#1A202C", CNN_ACC,
    "Global Average Pooling", "14×14×256  →  (256,)")
arrow(ax, CX, 6.34, CX, 5.98)

# Dense head
box(ax, CX, 5.72, 3.4, 0.48, CNN_COL, CNN_ACC,
    "Dense(256) + BN + ReLU", "Dropout(0.4)")
arrow(ax, CX, 5.46, CX, 5.10)

# CNN output
dim_label(ax, CX, 4.90, "CNN Feature Vector  →  (256,)", CNN_ACC)


# ═══════════════════════════════════════════════════════════════
#  FNN BRANCH  (right side, x=15.5)
# ═══════════════════════════════════════════════════════════════
FX = 15.5   # center-x of FNN column

section_header(ax, FX, 11.55, "  FNN Branch  ", FNN_ACC)

# Input
box(ax, FX, 11.0, 3.6, 0.52, "#1A202C", FNN_ACC,
    "ERA5 Tabular Input", "Latitude · Longitude · Temperature · Precipitation")
arrow(ax, FX, 10.74, FX, 10.38)

# Layer 1
box(ax, FX, 10.12, 3.2, 0.48, FNN_COL, FNN_ACC,
    "Dense(64) + BN + ReLU", "Dropout(0.3)  →  4 → 64")
arrow(ax, FX, 9.86, FX, 9.50)

# Layer 2
box(ax, FX, 9.24, 3.2, 0.48, FNN_COL, FNN_ACC,
    "Dense(128) + BN + ReLU", "Dropout(0.3)  →  64 → 128")
arrow(ax, FX, 8.98, FX, 8.62)

# Layer 3
box(ax, FX, 8.36, 3.2, 0.48, FNN_COL, FNN_ACC,
    "Dense(64) + BN + ReLU", "128 → 64")

# Spacer lines (visual alignment)
arrow(ax, FX, 8.10, FX, 4.90)
dim_label(ax, FX, 4.90, "FNN Feature Vector  →  (64,)", FNN_ACC)


# ═══════════════════════════════════════════════════════════════
#  FUSION ARROWS  (both branches → center)
# ═══════════════════════════════════════════════════════════════
# CNN arrow → fusion point
ax.annotate("", xy=(9.55, 3.90), xytext=(CX, 4.65),
            arrowprops=dict(arrowstyle="-|>", color=CNN_ACC, lw=2.0,
                            connectionstyle="arc3,rad=-0.25",
                            mutation_scale=15), zorder=2)
# FNN arrow → fusion point
ax.annotate("", xy=(10.45, 3.90), xytext=(FX, 4.65),
            arrowprops=dict(arrowstyle="-|>", color=FNN_ACC, lw=2.0,
                            connectionstyle="arc3,rad=0.25",
                            mutation_scale=15), zorder=2)


# ═══════════════════════════════════════════════════════════════
#  FUSION HEAD  (center, x=10)
# ═══════════════════════════════════════════════════════════════
FUX = 10.0

section_header(ax, FUX, 4.30, "  Fusion Head  ", FUS_ACC)

# Concat
box(ax, FUX, 3.75, 4.0, 0.50, "#1A202C", FUS_ACC,
    "Concatenate  [ CNN(256) ⊕ FNN(64) ]", "→  (320-dim fused vector)")
arrow(ax, FUX, 3.50, FUX, 3.14)

# Dense 128
box(ax, FUX, 2.88, 4.0, 0.48, FUS_COL, FUS_ACC,
    "Dense(128) + BN + ReLU", "Dropout(0.4)  →  320 → 128")
arrow(ax, FUX, 2.62, FUX, 2.26)

# Dense 64
box(ax, FUX, 2.00, 4.0, 0.48, FUS_COL, FUS_ACC,
    "Dense(64) + ReLU", "Dropout(0.3)  →  128 → 64")
arrow(ax, FUX, 1.74, FUX, 1.38)

# Output
box(ax, FUX, 1.12, 4.0, 0.48, OUT_COL, OUT_ACC,
    "Dense(1) + Sigmoid", "64 → 1  (probability score)")
arrow(ax, FUX, 0.86, FUX, 0.50)

# Final prediction
ax.text(FUX, 0.32, "CLOUDBURST  /  NO CLOUDBURST",
        ha="center", va="center", fontsize=10.5,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1A202C",
                  ec=OUT_ACC, lw=2.0), zorder=5)


# ═══════════════════════════════════════════════════════════════
#  VERTICAL SEPARATORS
# ═══════════════════════════════════════════════════════════════
ax.plot([7.2, 7.2], [4.3, 11.75], color="#2D3748", lw=1.0, ls="--")
ax.plot([12.8, 12.8], [4.3, 11.75], color="#2D3748", lw=1.0, ls="--")


# ═══════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=CNN_COL, edgecolor=CNN_ACC, label="CNN Branch (Image)"),
    mpatches.Patch(facecolor=FNN_COL, edgecolor=FNN_ACC, label="FNN Branch (Tabular)"),
    mpatches.Patch(facecolor=FUS_COL, edgecolor=FUS_ACC, label="Fusion Head"),
    mpatches.Patch(facecolor=OUT_COL, edgecolor=OUT_ACC, label="Output"),
]
leg = ax.legend(handles=legend_items, loc="lower right",
                bbox_to_anchor=(0.99, 0.01),
                framealpha=0.85, facecolor="#1A202C",
                edgecolor="#4A5568", labelcolor="white",
                fontsize=8.5)


# ─── Save ────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
plt.savefig("model_architecture.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("[Done] Saved: model_architecture.png")
