"""Regenerate the confusion matrix and training-curve charts using the
portfolio site's own pink/blue color scheme instead of matplotlib
defaults, at a larger size, for embedding in the project page."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "runs" / "grocery_detect"

PINK = "#e8187a"
BLUE = "#1A4480"
BLACK = "#14131A"
GRAY = "#6B7280"
BORDER = "#ECECEA"

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["text.color"] = BLACK
plt.rcParams["axes.edgecolor"] = BORDER
plt.rcParams["axes.labelcolor"] = BLACK
plt.rcParams["xtick.color"] = GRAY
plt.rcParams["ytick.color"] = GRAY

pink_cmap = LinearSegmentedColormap.from_list("pink_scale", ["#F4F4F3", "#FDE7EF", PINK])

# ---------- Confusion matrix (5 real classes, background merged as a note) ----------
classes = ["barilla", "corn_flakes", "indomie", "keya_piri_piri", "nut_bars"]
labels = ["Barilla", "Corn Flakes", "Indomie", "Keya Piri Piri", "Nut Bars"]
cm_full = np.load(OUT / "cm_raw.npy")  # 6x6, index 5 = background
cm = cm_full[:5, :5].astype(int)

fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(cm, cmap=pink_cmap, vmin=0)

ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel("True class", fontsize=13, labelpad=10)
ax.set_ylabel("Predicted class", fontsize=13, labelpad=10)
ax.set_title("Confusion Matrix — Held-Out Test Set (47 images)", fontsize=15, fontweight="bold", pad=16, color=BLACK)

for i in range(5):
    for j in range(5):
        v = cm[i, j]
        if v == 0:
            continue
        color = "white" if v > cm.max() * 0.55 else BLACK
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13, fontweight="bold", color=color)

for spine in ax.spines.values():
    spine.set_visible(False)

# draw an explicit box border around every cell (including empty ones)
for i in range(5):
    for j in range(5):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#D9D9D9", linewidth=1.5))

ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=10, color=GRAY)

plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_styled.png", dpi=200, facecolor="white")
plt.close()
print("saved confusion_matrix_styled.png")

# ---------- Training curves ----------
df = pd.read_csv(OUT / "results.csv")
df.columns = [c.strip() for c in df.columns]

fig, axes2d = plt.subplots(2, 2, figsize=(12, 9))
axes = axes2d.flatten()

def style_ax(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", color=BLACK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("epoch", fontsize=10, color=GRAY)
    ax.grid(axis="y", color=BORDER, linewidth=1)
    ax.set_axisbelow(True)

ax = axes[0]
ax.plot(df["epoch"], df["train/box_loss"], color=PINK, linewidth=2, label="train")
ax.plot(df["epoch"], df["val/box_loss"], color=BLUE, linewidth=2, label="val")
style_ax(ax, "Box Loss")
ax.legend(frameon=False, fontsize=9, labelcolor=GRAY)

ax = axes[1]
ax.plot(df["epoch"], df["train/cls_loss"], color=PINK, linewidth=2, label="train")
ax.plot(df["epoch"], df["val/cls_loss"], color=BLUE, linewidth=2, label="val")
style_ax(ax, "Classification Loss")
ax.legend(frameon=False, fontsize=9, labelcolor=GRAY)

ax = axes[2]
ax.plot(df["epoch"], df["metrics/precision(B)"], color=PINK, linewidth=2, label="precision")
ax.plot(df["epoch"], df["metrics/recall(B)"], color=BLUE, linewidth=2, label="recall")
style_ax(ax, "Precision & Recall")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=9, labelcolor=GRAY, loc="lower right")

ax = axes[3]
ax.plot(df["epoch"], df["metrics/mAP50(B)"], color=PINK, linewidth=2, label="mAP@0.5")
ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], color=BLUE, linewidth=2, label="mAP@0.5:0.95")
style_ax(ax, "mAP")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=9, labelcolor=GRAY, loc="lower right")

plt.tight_layout()
plt.savefig(OUT / "results_styled.png", dpi=200, facecolor="white")
plt.close()
print("saved results_styled.png")
