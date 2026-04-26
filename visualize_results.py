"""
visualize_results.py
Visualizes pipeline summary results as charts.

Usage:
    python visualize_results.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────

results = {
    "Small Only":            {"ap50": 67.48, "time": 1487.01, "escalation": 0.0},
    "Large Only":            {"ap50": 74.63, "time": 4745.53, "escalation": 100.0},
    "Collab (t=0.4)":        {"ap50": 74.45, "time": 2186.23, "escalation": 26.2},
    "Collab (t=0.5)":        {"ap50": 74.14, "time": 4189.28, "escalation": 68.5},
    "Collab (t=0.6)":        {"ap50": 74.43, "time": 5438.13, "escalation": 94.6},
    "Collab (t=0.7)":        {"ap50": 74.63, "time": 5709.06, "escalation": 100.0},
}

thresholds   = [0.4, 0.5, 0.6, 0.7]
collab_ap50  = [74.45, 74.14, 74.43, 74.63]
collab_time  = [2186.23, 4189.28, 5438.13, 5709.06]
collab_escal = [26.2, 68.5, 94.6, 100.0]

hist_bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
hist_data = {
    "Small Only": [13.2, 13.0, 68.3, 5.4, 0.0],
    "Large Only": [12.9,  0.0,  6.3, 52.2, 28.6],
}

BLUE   = "#4C9BE8"
RED    = "#E86B4C"
GREEN  = "#2ECC71"
GRAY   = "#95A5A6"
PURPLE = "#9B59B6"

collab_colors = ["#27AE60", "#F39C12", "#E67E22", "#E74C3C"]


# ── Figure 1: AP50 + Time for all pipelines ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pipeline Comparison — All Strategies (mean_score routing)", fontsize=13, fontweight="bold")

names  = list(results.keys())
ap50s  = [results[n]["ap50"] for n in names]
times  = [results[n]["time"]  for n in names]
colors = [BLUE, RED, GREEN, "#F39C12", "#E67E22", "#E74C3C"]

# AP50
ax = axes[0]
bars = ax.bar(names, ap50s, color=colors, width=0.6)
ax.set_title("Mean AP@50 (%)")
ax.set_ylim(64, 77)
ax.set_ylabel("%")
ax.tick_params(axis="x", rotation=20)
for bar, v in zip(bars, ap50s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
ax.axhline(y=74.63, color=RED, linestyle="--", linewidth=1, alpha=0.5, label="Large Only baseline")
ax.legend(fontsize=8)

# Time
ax = axes[1]
bars = ax.bar(names, times, color=colors, width=0.6)
ax.set_title("Total Inference Time (s)")
ax.set_ylabel("Seconds")
ax.tick_params(axis="x", rotation=20)
for bar, v in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{v:.0f}s", ha="center", va="bottom", fontsize=9)
ax.axhline(y=4745.53, color=RED, linestyle="--", linewidth=1, alpha=0.5, label="Large Only baseline")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_pipeline_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig1_pipeline_comparison.png")


# ── Figure 2: Threshold Sweep — AP50 / Time / Escalation ───────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Collaborative Pipeline — Threshold Sweep (mean_score routing)", fontsize=13, fontweight="bold")

# AP50 vs threshold
ax = axes[0]
ax.plot(thresholds, collab_ap50, "o-", color=GREEN, linewidth=2, markersize=8, label="Collaborative")
ax.axhline(y=74.63, color=RED, linestyle="--", linewidth=1.5, label=f"Large Only (74.63%)")
ax.axhline(y=67.48, color=BLUE, linestyle="--", linewidth=1.5, label=f"Small Only (67.48%)")
for x, y in zip(thresholds, collab_ap50):
    offset = -15 if y > 74.5 else 8
    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, offset), ha="center", fontsize=9)
ax.set_title("Mean AP@50 vs Threshold")
ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("%")
ax.set_ylim(64, 78)
ax.set_xticks(thresholds)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Time vs threshold
ax = axes[1]
ax.plot(thresholds, collab_time, "s-", color=GREEN, linewidth=2, markersize=8, label="Collaborative")
ax.axhline(y=4745.53, color=RED, linestyle="--", linewidth=1.5, label=f"Large Only (4,746s)")
ax.axhline(y=1487.01, color=BLUE, linestyle="--", linewidth=1.5, label=f"Small Only (1,487s)")
for x, y in zip(thresholds, collab_time):
    offset = -18 if y > 5000 else 10
    ax.annotate(f"{y:.0f}s", (x, y), textcoords="offset points", xytext=(0, offset), ha="center", fontsize=9)
ax.set_title("Inference Time vs Threshold")
ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("Seconds")
ax.set_ylim(1000, 6500)
ax.set_xticks(thresholds)
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Escalation vs threshold
ax = axes[2]
bars = ax.bar(thresholds, collab_escal, color=collab_colors, width=0.07)
for bar, v in zip(bars, collab_escal):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("Escalation Rate vs Threshold")
ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("% escalated to Large Model")
ax.set_ylim(0, 110)
ax.set_xticks(thresholds)
ax.grid(True, linestyle="--", alpha=0.4, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig2_threshold_sweep.png")


# ── Figure 3: AP50 vs Time Tradeoff Scatter ─────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Accuracy–Efficiency Tradeoff (mean_score routing)", fontsize=13, fontweight="bold")

# baselines
ax.scatter(1487.01, 67.48, s=200, color=BLUE, zorder=5, marker="^", label="Small Only")
ax.scatter(4745.53, 74.63, s=200, color=RED,  zorder=5, marker="^", label="Large Only")
ax.annotate("Small Only\n67.48%", (1487.01, 67.48), textcoords="offset points", xytext=(10, -20), fontsize=9, color=BLUE)
ax.annotate("Large Only\n74.63%", (4745.53, 74.63), textcoords="offset points", xytext=(10, 5),   fontsize=9, color=RED)

# collaborative points
for t, ap, tm, c in zip(thresholds, collab_ap50, collab_time, collab_colors):
    ax.scatter(tm, ap, s=180, color=c, zorder=5, marker="o")
    ax.annotate(f"t={t}\n{ap:.2f}%", (tm, ap), textcoords="offset points", xytext=(8, 5), fontsize=9, color=c)

# connect collaborative points
ax.plot(collab_time, collab_ap50, "--", color=GRAY, linewidth=1.2, zorder=3)

ax.set_xlabel("Total Inference Time (s)", fontsize=11)
ax.set_ylabel("Mean AP@50 (%)", fontsize=11)
ax.set_ylim(65, 76.5)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

# highlight best tradeoff
ax.annotate("Best tradeoff", (2186.23, 74.45),
            textcoords="offset points", xytext=(-80, -30),
            fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green"))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_tradeoff_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig3_tradeoff_scatter.png")


# ── Figure 4: Confidence Histograms (Small vs Large) ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confidence Score Distribution — Small Only vs Large Only", fontsize=13, fontweight="bold")

x = np.arange(len(hist_bins))
for ax, key, color, title in zip(axes, ["Small Only", "Large Only"], [BLUE, RED],
                                  ["Small Only (NanoDet-Plus)", "Large Only (YOLOv11L)"]):
    vals = hist_data[key]
    bars = ax.bar(x, vals, color=color, width=0.6, alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(hist_bins, fontsize=9)
    ax.set_ylabel("% of images")
    ax.set_ylim(0, 80)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_confidence_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig4_confidence_histograms.png")

print(f"\nAll figures saved to ./{OUTPUT_DIR}/")
