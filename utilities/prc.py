import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

import sklearn.metrics as skl_metrics
from .data import *
from .info import MODEL_TO_COL

## Plot settings (adapted from Kiran and Thijmen's repos)
sns.set_style("white")
sns.set_theme(
    "talk",
    "whitegrid",
    "dark",
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)
color_palette = sns.color_palette("colorblind")


## Return precision recall curves for one model on one group.
def calc_prc(df, pred_col, true_col="label"):
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    precision, recall, thresholds = skl_metrics.precision_recall_curve(y_true, y_pred)
    prc = {"precision": precision, "recall": recall, "thresholds": thresholds}
    auc = skl_metrics.auc(prc["recall"], prc["precision"])
    return prc, auc


## Check if the label has all true or all false.
## If this is the case, then the prc will not compute (and useless to do so).
def check_if_prc_ok(df, true_col="label", min_mal=1):
    total = len(df[true_col])
    mal_count = len(df.query(f"{true_col} == 1")[true_col])
    if mal_count == total:
        return False
    if mal_count < min_mal:
        return False
    return True


## prc test for different subgroups for a single model.
def calc_prcs_subgroups(
    df,
    cat,
    pred_col,
    include_all=False,
    true_col="label",
):
    prcs, aucs = {}, {}
    ## If we want to include the overall result for comparison.
    if include_all:
        prcs["ALL"], aucs["ALL"] = calc_prc(
            df,
            pred_col,
            true_col=true_col,
        )

    ## Get prc and AUC for subgroups.
    subgroups = df.groupby(cat)
    for subg, dfg in subgroups:
        is_prc_ok = check_if_prc_ok(dfg)
        if not is_prc_ok:
            continue

        prcs[subg], aucs[subg] = calc_prc(
            dfg,
            pred_col,
            true_col=true_col,
        )

    aucs
    return prcs, aucs


## Calculate prcs for different models (no subgroups).
## Note: models should be dictionary of format {'label': 'model_column"}
def calc_prcs_models(df, models=MODEL_TO_COL, true_col="label"):
    prcs, aucs = {}, {}
    for m in models:
        prcs[m], aucs[m] = calc_prc(
            df,
            models[m],
            true_col=true_col,
        )

    return prcs, aucs


## Calculate prcs for models for subgroups.
def calc_prcs_subgroups_models(
    df,
    cat,
    models=MODEL_TO_COL,
    include_all=False,
    true_col="label",
):
    prcs, aucs = {}, {}
    for m in models:
        prcs[m], aucs[m] = calc_prcs_subgroups(
            df,
            cat,
            models[m],
            include_all=include_all,
            true_col=true_col,
        )

    return prcs, aucs


def binary_group_prc_table(aucs, p, subgroups):
    assert len(subgroups) == 2
    tablerow = {}
    for m in aucs:
        tablerow[m] = {"p": p[m].loc[subgroups[0], subgroups[1]]}

        for i in range(2):
            g = subgroups[i]
            tablerow[m][f"Group_{i+1}"] = g
            tablerow[m][f"AUC_{i+1}"] = aucs[m].loc[g, "score"]

    return pd.DataFrame(tablerow).T


## General plotting function for multiple prc curves. Need to make figure separately.
def ax_prcs(ax, prcs, title=None, catinfo=None):
    ax.plot([0.0, 1.0], [0.5, 0.5], "--", color="k", alpha=0.5)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_xticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # X axis ticks in steps of 0.1
    ax.set_yticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # Y axis ticks in steps of 0.1
    # ax.grid(lw=1)
    ax.grid(visible=False)
    ax.set_xlim(-0.005, 1)
    ax.set_ylim(0, 1.005)

    for i, label in enumerate(prcs):
        prc = prcs[label]
        auc = skl_metrics.auc(prc["recall"], prc["precision"])
        legend_label = f"{label}: AUC = {auc:.3f}"
        if catinfo is not None:
            legend_label = f"{label} ({catinfo.loc[label, 'mal']} mal, {catinfo.loc[label, 'ben']} ben): \nAUC = {auc:.3f}"

        ax.plot(
            prc["recall"],
            prc["precision"],
            color=color_palette[i],
            label=legend_label,
        )

    if title:
        ax.set_title(title, fontsize=14)

    leg = ax.legend(loc="lower right", fontsize=12)
    return


## Plot prcs between models.
def plot_prcs_models(
    df,
    models=MODEL_TO_COL,
    prcs=None,
    aucs=None,
    dataset_name="NLST",
    title=None,
    imgpath=None,
    figsize=(6, 6),
    true_col="label",
):
    if prcs is None:
        prcs, aucs = calc_prcs_models(df, models, true_col)
    if title is None:
        title = f"{dataset_name} (n={len(df)}) Precision-Recall Curves Across Models"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax_prcs(ax=ax, prcs=prcs, title=title)

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)

    plt.show()
    return prcs, aucs


def plot_prcs_subgroups(
    df,
    cat,
    models=MODEL_TO_COL,
    prc=None,
    auc=None,
    dataset_name="NLST Scans",
    figheight=5,
    true_col="label",
    imgpath=None,
):
    df_catinfo = catinfo(df, cat)
    display(df_catinfo)

    if (prc is None) or (auc is None):
        prc, auc = calc_prcs_subgroups_models(
            df,
            cat,
            models,
            true_col=true_col,
            include_all=False,
        )

    first_prc = prc[list(prc.keys())[0]]
    if len(first_prc) < 2:
        print("Less than 2 valid groups. SKIP")
        return prc, auc

    if len(models) <= 4:
        fig, ax = plt.subplots(
            1, len(models), figsize=(figheight * len(models) - 0.5, figheight)
        )
    else:
        lm = len(models)
        if lm % 2 == 1:
            lm += 1
        fig, ax = plt.subplots(
            2,
            lm // 2,
            figsize=(figheight * (lm // 2) - 0.5, figheight * 2),
            squeeze=False,
        )
        ax = ax.flatten()

    fig.suptitle(f"{dataset_name} (n={len(df)}) Model prc Curves Split By {cat}")

    for i, m in enumerate(models):
        title_str = f"{m} on {dataset_name} (n={len(df)}) \nprc by {cat}"
        ax_prcs(ax[i], prc[m], title=title_str, catinfo=df_catinfo)

    plt.tight_layout()
    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)
    plt.show()

    return prc, auc
