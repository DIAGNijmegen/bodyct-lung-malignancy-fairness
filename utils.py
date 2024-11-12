import pandas as pd
import os
import numpy as np

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt

import sklearn.metrics as skl_metrics

ILST_THRESHOLD = 0.0151
MODEL_TO_COL = {
    "Venkadesh": "DL",
    "de Haas Combined": "Thijmen_mean",
    "de Haas Local": "Thijmen_local",
    "de Haas Global (hidden nodule)": "Thijmen_global_hidden",
    "de Haas Global (shown nodule)": "Thijmen_global_show",
    "Sybil": "sybil_year1",
    "PanCan2b": "PanCan2b",
}

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

def rocs_models(df, true_col='label', 
                models=MODEL_TO_COL, 
                dataset_label="DLCST", subset_label="all", imgpath=None, plot_ci=False):
    rocs = {}
    for m in models:
        rocs[m] = get_bootstrapped_roc_ci_curves(df[models[m]].values, df[true_col].values)
    
    plot_rocs(rocs, f'{dataset_label} ({subset_label} patients, n={len(df)}) ROC Curves Across Models ', imgpath, plot_ci)

def stats_from_cm(tp, tn, fp, fn):
    metrics = {
        "num": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tp / (tp + fn), ## Recall, sensitivity, hit rate 
        "fpr": fp / (fp + tn), ## Overdiagnosis: incorrect malignant classification 
        "fnr": fn / (tp + fn), ## Underdiagnosis: malignant classification missed 
        "tnr": tn / (tn + fp), ## Specificity
        "ppv": tp / (tp + fp), ## Precision: positive predictive value
        "npv": tn / (tn + fn), ## negative predictive value
        "fdr": fp / (fp + tp), ## False discovery rate
        "for": fn / (fn + tn), ## False omission rate,
        "acc": (tp + tn) / (tp + fp + fn + tn)
    }
    return metrics

def cm_with_thres(df, threshold=ILST_THRESHOLD, pred_col='DL', true_col='label'):
    y_true = df[true_col].to_numpy()
    y_pred = (df[pred_col] > threshold).astype(int).to_numpy()
    tn, fp, fn, tp = skl_metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn

def info_by_splits(groups):
    cat_info = {'num': [], 'pct': [], 'num_mal': [], 'pct_mal': []}
    cat_vals = []
    n = sum(len(df) for _, df in groups)

    plot_roc = True
    for val, df_group in groups:
        cat_vals.append(val)
        cat_info['num'].append(len(df_group))
        cat_info['pct'].append(100 * len(df_group) / n)
        cat_info['num_mal'].append(len(df_group.query('label == 1')))
        cat_info['pct_mal'].append(100 * len(df_group.query('label == 1')) / len(df_group))

        if len(df_group.query('label == 1')) == 0 or len(df_group.query('label == 1')) == len(df_group):
            plot_roc = False
    
    df_catinfo = pd.DataFrame(cat_info, index=cat_vals)
    return df_catinfo, plot_roc

def perf_by_splits(groups, pred_col='DL', true_col='label', threshold=ILST_THRESHOLD):
    rocs = {}
    stats = []
    vals = []

    for val, df_group in groups:
        y_true = df_group[true_col].values
        y_pred = df_group[pred_col].values
        rocs[val] = get_bootstrapped_roc_ci_curves(y_pred, y_true)

        vals.append(val)
        stats.append(stats_from_cm(*cm_with_thres(df_group, threshold=threshold, pred_col=pred_col, true_col=true_col)))

    df_modelperf = pd.DataFrame(stats, index=vals)
    return rocs, df_modelperf

def roc_cm_by_category(df, cat, models=MODEL_TO_COL, min_mal=2, threshold=ILST_THRESHOLD):
    groups = df.groupby(cat) 
    df_catinfo, plot_roc = info_by_splits(groups)

    if (plot_roc == False) or (min(df_catinfo['num_mal']) < min_mal):
        print("Not plotting ROC since there are not enough malignant or benign nodules in a category :(")
        return df_catinfo, {}

    rocs = {}
    perfs = {}
    for m in models:
        rocs[m], perfs[m] = perf_by_splits(groups, pred_col=models[m], threshold=threshold)

    fig, ax = plt.subplots(1, len(models), figsize=(6.5*len(models) - 0.5, 6))
    fig.suptitle(f"Model Performance Split By {cat}")
    for i, m in enumerate(models):
        ax_rocs(ax[i], rocs[m], title=m)
    plt.show()

    return df_catinfo, perfs

def prep_nlst_preds(df, scanlevel=True, sybil=True, tijmen=True):
    if scanlevel:
        nodule_drop_cols = ['NoduleType', 'CoordX', 'CoordY', 'CoordZ', 'NoduleID', 'AnnotationID']
        nodule_agg_cols = ['Spiculation', 'Diameter [mm]', 'NoduleCounts', 'NoduleInUpperLung']
        model_cols = ['DL', 'PanCan2b', 'label',
                    'Thijmen_mean', 'Thijmen_global_hidden', 'Thijmen_global_show', 'Thijmen_local']
        # Not including Sybil here because it's already scan-level of course.
        
        df = df.drop(nodule_drop_cols, axis=1)
        dfgb = df.groupby('SeriesInstanceUID')

        for c in nodule_agg_cols + model_cols:
            df[c] = dfgb[c].transform(max)
        
        df = df.drop_duplicates(['SeriesInstanceUID'], ignore_index=True)
    
    if tijmen: df = df[(~df['Thijmen_mean'].isna())]
    if sybil: df = df[(~df['sybil_year1'].isna())]
    return df
