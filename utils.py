import pandas as pd
import os
import numpy as np

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt

import sklearn.metrics as skl_metrics

## Plot settings (adapted from Kiran and Thijmen's repos)
sns.set_style("white")
sns.set_theme(
    "talk",
    "whitegrid",
    "dark",
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)
color_palette = sns.color_palette("colorblind")

def plot_rocs(rocs, title=None, imgpath=None, plot_ci=False, figsize=(6,6)):
    plt.figure(figsize=figsize)
    plt.plot([0.0, 1.0], [0.0, 1.0], "--", color="k", alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)  # X axis ticks in steps of 0.1
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)  # Y axis ticks in steps of 0.1
    plt.grid(lw=1)
    plt.xlim(-0.005,1)
    plt.ylim(0,1.005)

    for i, label in enumerate(rocs):
        roc = rocs[label]
        # roc = get_bootstrapped_roc_ci_curves(df[MODEL_TO_COL[m]].values, df[true_col].values)
        auc = skl_metrics.auc(roc.fpr_vals, roc.mean_tpr_vals)
        plt.plot(
            roc.fpr_vals, roc.mean_tpr_vals, color=color_palette[i],
            label=f"{label}: AUC = {auc:.3f} (95% CI: {roc.low_az_val:.3f} - {roc.high_az_val:.3f})",
        )
        if plot_ci:
            plt.fill_between(roc.fpr_vals, roc.low_tpr_vals, roc.high_tpr_vals, color=color_palette[i], alpha=.1)

    if title:
        plt.title(title, fontsize=14)

    leg = plt.legend(loc='lower right', fontsize=12)
    # shift = max([t.get_window_extent().width for t in leg.get_texts()])
    # for t in leg.get_texts():
    #     t.set_ha('right') # ha is alias for horizontalalignment
    #     t.set_position((shift - t.get_window_extent().width,0))

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)
    plt.show()

def ax_rocs(ax, rocs, title=None, plot_ci=False):
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="k", alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_xticks(np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12)  # X axis ticks in steps of 0.1
    ax.set_yticks(np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12)  # Y axis ticks in steps of 0.1
    ax.grid(lw=1)
    ax.set_xlim(-0.005,1)
    ax.set_ylim(0,1.005)

    for i, label in enumerate(rocs):
        roc = rocs[label]
        # roc = get_bootstrapped_roc_ci_curves(df[MODEL_TO_COL[m]].values, df[true_col].values)
        auc = skl_metrics.auc(roc.fpr_vals, roc.mean_tpr_vals)

        ax.plot(
            roc.fpr_vals, roc.mean_tpr_vals, color=color_palette[i],
            label=f"{label}: AUC = {auc:.3f} ({roc.low_az_val:.3f} - {roc.high_az_val:.3f})",
        )
        if plot_ci:
            ax.fill_between(roc.fpr_vals, roc.low_tpr_vals, roc.high_tpr_vals, color=color_palette[i], alpha=.1)

    if title:
        ax.set_title(title, fontsize=14)

    leg = ax.legend(loc='lower right', fontsize=12)
    # shift = max([t.get_window_extent().width for t in leg.get_texts()])
    # for t in leg.get_texts():
    #     t.set_ha('right') # ha is alias for horizontalalignment
    #     t.set_position((shift - t.get_window_extent().width,0))

