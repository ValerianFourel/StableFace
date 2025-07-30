#!/usr/bin/env python3
# ------------------------------------------------------------
# emotion_metrics_plots.py
#
#   1) Text report (overall & per-class)
#   2) radar_gt.png   – Base / Fine vs ground-truth (polygonal)
#   3) radar_orig.png – Base / Fine vs EmoNet(real) (polygonal)
#   4) overall_bars.png – grouped bar chart with Top-1 & Top-3
# ------------------------------------------------------------

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. helpers
# ------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[ ,.;:\t\r\n]+$")

def clean(series: pd.Series) -> pd.Series:
    """lower-case, strip blanks and trailing punctuation."""
    return (series.astype(str)
                  .str.lower()
                  .str.strip()
                  .str.replace(_PUNCT_RE, "", regex=True))

def load_csv(path: str) -> pd.DataFrame:
    needed = [
        "gt",
        "orig_top1", "orig_top2", "orig_top3",
        "base_top1", "base_top2", "base_top3",
        "fine_top1", "fine_top2", "fine_top3",
    ]
    df = pd.read_csv(
        path,
        usecols=lambda c: c in needed,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip",
        escapechar="\\"
    )
    for col in needed:
        df[col] = clean(df[col])
    return df

def per_class_recall(df, target_col, pred_col, classes):
    rec = []
    for c in classes:
        m = df[target_col] == c
        tot = m.sum()
        rec.append((df.loc[m, pred_col] == c).sum() / tot if tot else 0.0)
    return np.asarray(rec)

def per_class_recall_top3(df, target_col, cols_pred_top3, classes):
    rec = []
    for c in classes:
        m = df[target_col] == c
        tot = m.sum()
        if tot == 0:
            rec.append(0.0)
            continue
        correct = (
            (df.loc[m, cols_pred_top3[0]] == c) |
            (df.loc[m, cols_pred_top3[1]] == c) |
            (df.loc[m, cols_pred_top3[2]] == c)
        ).sum()
        rec.append(correct / tot)
    return np.asarray(rec)

# ------------------------------------------------------------------
# 2. plotting utilities
# ------------------------------------------------------------------
def setup_polygonal_radar(ax):
    """Configure the radar plot with straight lines and no degree markings."""
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"])
    # Remove degree markings
    ax.set_thetagrids([])
    # Custom grid with straight lines
    ax.grid(True, ls="--", lw=0.5, alpha=0.7)

def radar(ax, values, angles, label, color):
    """Plot radar with straight lines connecting points."""
    vals = values.tolist() + [values[0]]
    # Plot with straight lines (no curve fitting) - Fixed: removed duplicate linewidth
    ax.plot(angles, vals, lw=3, label=label, color=color, marker='o', 
            markersize=6, linestyle='-')
    ax.fill(angles, vals, alpha=0.2, color=color)

def label_axes(ax, angles, classes):
    """Place class labels outside the plot area."""
    for ang, c in zip(angles, classes):
        # Calculate position for labels outside the plot
        x_pos = 1.15 * np.cos(ang)
        y_pos = 1.15 * np.sin(ang)

        ax.text(ang, 1.15, c,
                ha="center", va="center",
                rotation=0,
                size=11, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='gray', alpha=0.8))

# ------------------------------------------------------------------
# 3. main
# ------------------------------------------------------------------
def main(cfg):
    df = load_csv(cfg.csv)

    classes = sorted(df["gt"].unique())
    n_cls   = len(classes)
    angles  = np.linspace(0, 2*np.pi, n_cls, endpoint=False).tolist() + [0]

    # ------------------------------------------------------------------
    # Metrics dictionaries
    # ------------------------------------------------------------------
    metrics = {}

    # --- Accuracy against Ground-Truth ---------------------------------
    metrics["GT_top1"] = {
        "orig":  (df["orig_top1"] == df["gt"]).mean(),
        "base":  (df["base_top1"] == df["gt"]).mean(),
        "fine":  (df["fine_top1"] == df["gt"]).mean(),
    }
    metrics["GT_top3"] = {
        "base":  ((df["base_top1"] == df["gt"]) |
                  (df["base_top2"] == df["gt"]) |
                  (df["base_top3"] == df["gt"])).mean(),
        "fine":  ((df["fine_top1"] == df["gt"]) |
                  (df["fine_top2"] == df["gt"]) |
                  (df["fine_top3"] == df["gt"])).mean(),
    }

    # --- Accuracy against EmoNet label of real photo -------------------
    metrics["EMO_top1"] = {
        "base": (df["base_top1"] == df["orig_top1"]).mean(),
        "fine": (df["fine_top1"] == df["orig_top1"]).mean(),
    }
    metrics["EMO_top3"] = {
        "base": ((df["base_top1"] == df["orig_top1"]) |
                 (df["base_top2"] == df["orig_top1"]) |
                 (df["base_top3"] == df["orig_top1"])).mean(),
        "fine": ((df["fine_top1"] == df["orig_top1"]) |
                 (df["fine_top2"] == df["orig_top1"]) |
                 (df["fine_top3"] == df["orig_top1"])).mean(),
    }

    # --- Per-class recalls (Top-1) -------------------------------------
    per_cls_gt = {
        "base": per_class_recall(df, "gt",   "base_top1", classes),
        "fine": per_class_recall(df, "gt",   "fine_top1", classes),
    }
    per_cls_emo = {
        "base": per_class_recall(df, "orig_top1", "base_top1", classes),
        "fine": per_class_recall(df, "orig_top1", "fine_top1", classes),
    }

    # ------------------------------------------------------------------
    # 4. Console report
    # ------------------------------------------------------------------
    def pct(x): return f"{x*100:6.2f}%"

    print("\n============== GLOBAL ACCURACIES ==============")
    print("Against Ground-Truth")
    print(f"  Orig  Top-1 {pct(metrics['GT_top1']['orig'])}")
    print(f"  Base  Top-1 {pct(metrics['GT_top1']['base'])} | "
          f"Top-3 {pct(metrics['GT_top3']['base'])}")
    print(f"  Fine  Top-1 {pct(metrics['GT_top1']['fine'])} | "
          f"Top-3 {pct(metrics['GT_top3']['fine'])}")

    print("\nAgainst EmoNet label of real image")
    print(f"  Base  Top-1 {pct(metrics['EMO_top1']['base'])} | "
          f"Top-3 {pct(metrics['EMO_top3']['base'])}")
    print(f"  Fine  Top-1 {pct(metrics['EMO_top1']['fine'])} | "
          f"Top-3 {pct(metrics['EMO_top3']['fine'])}")
    print("===============================================\n")

    # ------------------------------------------------------------------
    # 5. Polygonal Radar plot ① – versus Ground-Truth (Base & Fine only)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    setup_polygonal_radar(ax1)

    palette = {"base": "#ff7f0e", "fine": "#2ca02c"}
    radar(ax1, per_cls_gt["base"], angles, "Baseline → EmoNet", palette["base"])
    radar(ax1, per_cls_gt["fine"], angles, "Finetuned → EmoNet", palette["fine"])
    label_axes(ax1, angles[:-1], classes)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=12)
    # ax1.set_title("Per-class Top-1 Recall vs Ground-Truth", pad=40, size=16, weight='bold')

    # ------------------------------------------------------------------
    # 6. Polygonal Radar plot ② – versus EmoNet(real)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    setup_polygonal_radar(ax2)

    radar(ax2, per_cls_emo["base"], angles, "Baseline → EmoNet", palette["base"])
    radar(ax2, per_cls_emo["fine"], angles, "Finetuned → EmoNet", palette["fine"])
    label_axes(ax2, angles[:-1], classes)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=12)
   # ax2.set_title("Per-class Top-1 Recall vs EmoNet(real)", pad=40, size=16, weight='bold')

    # ------------------------------------------------------------------
    # 7. Global bar chart (Top-1 & Top-3)
    # ------------------------------------------------------------------
    labels = [
        "Base-GT",  "Fine-GT",
        "Base-EMO", "Fine-EMO",
    ]
    top1 = [
        metrics["GT_top1"]["base"],  metrics["GT_top1"]["fine"],
        metrics["EMO_top1"]["base"], metrics["EMO_top1"]["fine"],
    ]
    top3 = [
        metrics["GT_top3"]["base"],  metrics["GT_top3"]["fine"],
        metrics["EMO_top3"]["base"], metrics["EMO_top3"]["fine"],
    ]
    x = np.arange(len(labels))
    width = 0.35

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars1 = ax3.bar(x - width/2, np.array(top1)*100, width, label="Top-1", alpha=0.8)
    bars2 = ax3.bar(x + width/2, np.array(top3)*100, width, label="Top-3", alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', size=9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15)
    ax3.set_ylabel("Accuracy [%]", size=12)
    ax3.set_ylim(0, 110)
    ax3.set_title("Global Top-1 / Top-3 Accuracy", size=14, weight='bold')
    ax3.grid(axis="y", ls="--", lw=0.5, alpha=0.6)
    ax3.legend()

    # ------------------------------------------------------------------
    # 8. Save figures
    # ------------------------------------------------------------------
    out_dir = Path(cfg.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1.tight_layout(); fig2.tight_layout(); fig3.tight_layout()
    fig1.savefig(out_dir / "radar_gt.png",   dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "radar_orig.png", dpi=300, bbox_inches='tight')
    fig3.savefig(out_dir / "overall_bars.png", dpi=300, bbox_inches='tight')
    print(f"Figures saved to {out_dir.resolve()}\n")

    if cfg.show:
        plt.show()

# ------------------------------------------------------------------
# 9. CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute emotion metrics and generate polygonal radar / bar plots."
    )
    parser.add_argument("--csv",    required=True, help="Path to emotion_results.csv")
    parser.add_argument("--outdir", default="plots", help="Directory for the PNGs")
    parser.add_argument("--show",   action="store_true", help="Display figures")
    main(parser.parse_args())
