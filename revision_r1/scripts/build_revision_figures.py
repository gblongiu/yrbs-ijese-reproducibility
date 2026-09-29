from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

R1 = Path(__file__).resolve().parents[1]
OUT = R1 / "outputs" / "figures" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)


def workflow_figure():
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 4.3)
    ax.axis('off')
    boxes = [
        (0.15, 3.22, 2.15, .75, '2023 National YRBS\nFixed analytic sample'),
        (2.62, 3.22, 2.15, .75, 'Frozen split\nTraining / evaluation'),
        (5.09, 3.22, 2.15, .75, 'Training-side CV\nSelection + calibration'),
        (2.62, 1.75, 2.15, .75, 'Locked fitted pipelines\nBaseline and expanded'),
        (.15, .30, 3.30, .75, '2023 internal evaluation\nFixed respondent partition'),
        (3.95, .30, 3.30, .75, '2021 cross-wave validation\nNo refitting or updating'),
    ]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.04',facecolor='white',edgecolor='black',linewidth=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for start,end in [((2.30,3.60),(2.62,3.60)),((4.77,3.60),(5.09,3.60)),((6.16,3.22),(4.45,2.50)),((3.25,1.75),(1.80,1.05)),((4.10,1.75),(5.60,1.05))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.1))
    ax.text(.2,2.45,'Baseline: demographics\nExpanded: demographics\n+ two bullying indicators',ha='left',va='top',fontsize=9)
    fig.tight_layout(pad=.2)
    fig.savefig(OUT/'Figure_1_R1_workflow.png',dpi=300,bbox_inches='tight',metadata={})
    plt.close(fig)


def improvement_figure():
    internal = pd.read_csv(R1 / "outputs" / "tables" / "paired_improvement_ci_2023_seed20260915.csv")
    internal = internal[internal["comparison"] == "platt"].copy()
    external = pd.read_csv(R1 / "outputs" / "external" / "yrbs2021_external_incremental_value_ci.csv")
    labels = ["ROC AUC change", "Average precision change", "Brier improvement"]
    keys = ["roc_auc_change", "average_precision_change", "brier_improvement"]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    y = np.arange(3)
    offset = 0.12
    for data, dy, marker, label in [(internal, -offset, "o", "2023 internal"), (external, offset, "s", "2021 external")]:
        values, lows, highs = [], [], []
        for key in keys:
            row = data[data["metric"] == key].iloc[0]
            values.append(row["point_difference"])
            lows.append(row["ci95_low"])
            highs.append(row["ci95_high"])
        values = np.array(values)
        lows = np.array(lows)
        highs = np.array(highs)
        ax.errorbar(values, y + dy, xerr=np.vstack([values - lows, highs - values]), fmt=marker, capsize=3, linewidth=1.2, markersize=5, label=label)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Incremental improvement")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_2_R1_incremental_improvements.png", dpi=300, bbox_inches="tight", metadata={})
    plt.close(fig)


def reliability_figure():
    bins = pd.read_csv(R1 / "outputs" / "tables" / "reliability_bins_final_model_2023_seed2026.csv")
    fig = plt.figure(figsize=(7.3, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Ideal")
    ax.plot(bins["mean_predicted_probability"], bins["observed_event_frequency"], marker="o", linewidth=1.5, label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed event frequency")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(axis="x", labelbottom=False)
    ax_hist.bar(bins["mean_predicted_probability"], bins["n"], width=0.055, edgecolor="black", linewidth=0.6)
    ax_hist.set_ylim(0, 430)
    ax_hist.set_ylabel("n")
    ax_hist.set_xlabel("Mean predicted probability")
    ax_hist.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_R1_reliability.png", dpi=300, bbox_inches="tight", metadata={})
    plt.close(fig)


if __name__ == "__main__":
    workflow_figure()
    improvement_figure()
    reliability_figure()
