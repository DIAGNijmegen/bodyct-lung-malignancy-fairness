import pandas as pd
import os
import numpy as np

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt
import scipy.stats

import sklearn.metrics as skl_metrics
import warnings

ILST_THRESHOLD = 0.06
MODEL_TO_COL = {
    "Venkadesh": "DL",
    "de Haas Combined": "Thijmen_mean",
    "de Haas Local": "Thijmen_local",
    "de Haas Global (hidden nodule)": "Thijmen_global_hidden",
    "de Haas Global (shown nodule)": "Thijmen_global_show",
    "Sybil year 1": "sybil_year1",
    "Sybil year 2": "sybil_year2",
    "Sybil year 3": "sybil_year3",
    "Sybil year 4": "sybil_year4",
    "Sybil year 5": "sybil_year5",
    "Sybil year 6": "sybil_year6",
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


def ax_rocs(ax, curves, title=None, plot_ci=False, catinfo=None):
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="k", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
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

    aucs = {}
    for i, label in enumerate(curves):
        roc = curves[label]
        # roc = get_bootstrapped_roc_ci_curves(df[MODEL_TO_COL[m]].values, df[true_col].values)
        auc = skl_metrics.auc(roc.fpr_vals, roc.mean_tpr_vals)
        aucs[label] = {"score": auc, "ci-hi": roc.high_az_val, "ci-lo": roc.low_az_val}

        legend_label = (
            f"{label}: AUC = {auc:.3f} ({roc.low_az_val:.3f} - {roc.high_az_val:.3f})"
        )
        if catinfo is not None:
            legend_label = f"{label} ({catinfo.loc[label, 'num_mal']} mal, {catinfo.loc[label, 'num'] - catinfo.loc[label, 'num_mal']} ben): \nAUC = {auc:.3f} ({roc.low_az_val:.3f} - {roc.high_az_val:.3f})"

        ax.plot(
            roc.fpr_vals,
            roc.mean_tpr_vals,
            color=color_palette[i],
            label=legend_label,
        )
        if plot_ci:
            ax.fill_between(
                roc.fpr_vals,
                roc.low_tpr_vals,
                roc.high_tpr_vals,
                color=color_palette[i],
                alpha=0.1,
            )

    if title:
        ax.set_title(title, fontsize=14)

    leg = ax.legend(loc="lower right", fontsize=12)
    return pd.DataFrame(aucs)


def ax_prcs(ax, curves, title=None, plot_ci=False):
    ax.plot([0.0, 1.0], [0.5, 0.5], "--", color="k", alpha=0.5)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_xticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # X axis ticks in steps of 0.1
    ax.set_yticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # Y axis ticks in steps of 0.1
    ax.grid(lw=1)
    ax.set_xlim(-0.005, 1)
    ax.set_ylim(0, 1.005)

    aucs = {}
    for i, label in enumerate(curves):
        prc = curves[label]
        # roc = get_bootstrapped_roc_ci_curves(df[MODEL_TO_COL[m]].values, df[true_col].values)
        auc = skl_metrics.auc(prc["recall"], prc["precision"])
        aucs[label] = {"score": auc}

        ax.plot(
            prc["recall"],
            prc["precision"],
            color=color_palette[i],
            label=f"{label}: AUC = {auc:.3f}",
        )

    if title:
        ax.set_title(title, fontsize=14)

    leg = ax.legend(loc="lower right", fontsize=12)
    return pd.DataFrame(aucs)


def rocs_models(
    df,
    true_col="label",
    models=MODEL_TO_COL,
    dataset_label="NLST",
    subset_label="all",
    title=None,
    imgpath=None,
    plot_ci=False,
    figsize=(6, 6),
):
    rocs = {}
    for m in models:
        rocs[m] = get_bootstrapped_roc_ci_curves(
            df[models[m]].values, df[true_col].values
        )

    if title is None:
        title = f"{dataset_label} (n={len(df)}) ROC Curves Across Models "

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    auc_data = ax_rocs(ax=ax, curves=rocs, title=title, plot_ci=plot_ci)

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)
    plt.show()

    return auc_data


def prcs_models(
    df,
    true_col="label",
    models=MODEL_TO_COL,
    dataset_label="NLST",
    subset_label="all",
    title=None,
    imgpath=None,
    plot_ci=False,
    figsize=(6, 6),
):
    prcs = {}
    for m in models:
        y_true = df[true_col].values
        y_pred = df[models[m]].values
        precision, recall, thresholds = skl_metrics.precision_recall_curve(
            y_true, y_pred
        )
        prcs[m] = {"precision": precision, "recall": recall, "thresholds": thresholds}

    if title is None:
        title = f"{dataset_label} (n={len(df)}) Precision Recall Curves Across Models "

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    auc_data = ax_prcs(ax=ax, curves=prcs, title=title, plot_ci=False)

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)
    plt.show()
    return auc_data


def stats_from_cm(tp, tn, fp, fn):
    metrics = {}
    metrics["num"] = tp + fp + fn + tn
    metrics["mal"] = tp + fn
    metrics["ben"] = fp + tn
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        metrics["tpr"] = tp / (tp + fn)  ## Recall, sensitivity, hit rate
        metrics["fpr"] = fp / (fp + tn)  ## Overdiagnosis
        metrics["fnr"] = fn / (tp + fn)  ## Underdiagnosis
        metrics["tnr"] = tn / (tn + fp)  ## Specificity
        metrics["ppv"] = tp / (tp + fp)  ## Precision: positive predictive value
        metrics["npv"] = tn / (tn + fn)  ## negative predictive value
        metrics["fdr"] = fp / (fp + tp)  ## False discovery rate
        metrics["for"] = fn / (fn + tn)  ## False omission rate
        metrics["acc"] = (tp + tn) / (tp + fp + fn + tn)  ## Accuracy
        metrics["j"] = (
            metrics["tpr"] - metrics["fpr"]
        )  ## Youden's J statistic (seen in some papers)
        metrics["f1"] = (2 * tp) / (2 * tp + fp + fn)  ## f1 Score
        metrics["mcc"] = np.sqrt(  ## Matthews Correlation Coefficient
            metrics["tpr"] * metrics["tnr"] * metrics["ppv"] * metrics["npv"]
        ) - np.sqrt(metrics["fpr"] * metrics["fnr"] * metrics["for"] * metrics["fdr"])

    return metrics


def cm_with_thres(df, threshold=ILST_THRESHOLD, pred_col="DL", true_col="label"):
    y_true = df[true_col].to_numpy()
    y_pred = (df[pred_col] > threshold).astype(int).to_numpy()
    tn, fp, fn, tp = skl_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp, tn, fp, fn


def info_by_splits(groups, min_mal):
    cat_info = {"num": [], "pct": [], "num_mal": [], "pct_mal": []}
    cat_vals = []
    n = sum(len(df) for _, df in groups)
    skips = []

    for val, df_group in groups:
        cat_vals.append(val)
        cat_info["num"].append(len(df_group))
        cat_info["pct"].append(100 * len(df_group) / n)
        cat_info["num_mal"].append(len(df_group.query("label == 1")))
        cat_info["pct_mal"].append(
            100 * len(df_group.query("label == 1")) / len(df_group)
        )

        if len(df_group.query("label == 1")) < min_mal or len(
            df_group.query("label == 1")
        ) == len(df_group):
            skips.append(val)

    df_catinfo = pd.DataFrame(cat_info, index=cat_vals)
    return df_catinfo, skips


def roc_by_splits(groups, pred_col="DL", true_col="label", skips=[]):
    rocs = {}

    for val, df_group in groups:
        if val not in skips:
            y_true = df_group[true_col].values
            y_pred = df_group[pred_col].values
            rocs[val] = get_bootstrapped_roc_ci_curves(y_pred, y_true)

    return rocs


def prc_by_splits(groups, pred_col="DL", true_col="label", skips=[]):
    prcs = {}

    for val, df_group in groups:
        if val not in skips:
            y_true = df_group[true_col].values
            y_pred = df_group[pred_col].values
            precision, recall, thresholds = skl_metrics.precision_recall_curve(
                y_true, y_pred
            )
            prcs[val] = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
            }

    return prcs


def threshold_metrics_by_splits(
    groups, pred_col="DL", true_col="label", threshold=ILST_THRESHOLD, include_all=False
):
    stats = []
    vals = []

    for val, df_group in groups:
        vals.append(val)
        stats.append(
            stats_from_cm(
                *cm_with_thres(
                    df_group,
                    threshold=threshold,
                    pred_col=pred_col,
                    true_col=true_col,
                )
            )
        )

    df_modelperf = pd.DataFrame(stats, index=vals)
    return df_modelperf


def perf_by_splits(
    groups, pred_col="DL", true_col="label", threshold=ILST_THRESHOLD, skips=[]
):
    rocs = roc_by_splits(groups, pred_col, true_col, skips)
    threshold_metrics = threshold_metrics_by_splits(
        groups, pred_col, true_col, threshold
    )
    return rocs, threshold_metrics


def roc_by_category(
    df, cat, models=MODEL_TO_COL, min_mal=2, dataset_name="NLST", figheight=5
):
    groups = df.groupby(cat)
    df_catinfo, skips = info_by_splits(groups, min_mal)

    if len(df_catinfo) - len(skips) < 2:
        print("Less than two groups. SKIP")
        return df_catinfo, None, None, None

    rocs = {}
    z = {}
    p = {}
    for m in models:
        rocs[m] = roc_by_splits(groups, pred_col=models[m], skips=skips)
        z[m], p[m] = hanley_mcneil_sigtest(df_catinfo, skips, rocs[m])

    sigtest_on_plot = (len(df_catinfo) - len(skips)) == 2

    if len(models) <= 5:
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
        # ax = ax[0] + ax[1]

    aucs = {}
    fig.suptitle(f"{dataset_name} (n={len(df)}) Model ROC Curves Split By {cat}")
    for i, m in enumerate(models):
        title_str = f"{m} on {dataset_name} (n={len(df)}) \nROC by {cat}"
        if sigtest_on_plot:
            z_show, p_show = z[m].iloc[0, 1], p[m].iloc[0, 1]
            title_str = f"{m} on {dataset_name} (n={len(df)}) \nROC by {cat} (z={z_show:.2f}, p={p_show:.3f})"
            if p_show < 0.001:
                title_str = f"{m} on {dataset_name} (n={len(df)}) \nROC by {cat} (z={z_show:.2f}, p<0.001)"

        aucs[m] = ax_rocs(
            ax[i], rocs[m], title=title_str, catinfo=df_catinfo, plot_ci=True
        )

    plt.tight_layout()
    plt.show()

    return df_catinfo, aucs, z, p


def prc_by_category(df, cat, models=MODEL_TO_COL, min_mal=2):
    groups = df.groupby(cat)
    df_catinfo, skips = info_by_splits(groups, min_mal)

    if len(df_catinfo) - len(skips) < 2:
        print("Less than two groups. SKIP")
        return df_catinfo, None, None

    prcs = {}
    for m in models:
        prcs[m] = prc_by_splits(groups, pred_col=models[m], skips=skips)

    # do_sigtest = (len(df_catinfo) - len(skips)) == 2
    # bin_sigtest_results = {}

    if len(models) <= 5:
        fig, ax = plt.subplots(1, len(models), figsize=(6 * len(models) - 0.5, 6))
    else:
        lm = len(models)
        if lm % 2 == 1:
            lm += 1
        fig, ax = plt.subplots(
            2, lm // 2, figsize=(6 * (lm // 2) - 0.5, 6 * 2), squeeze=False
        )
        ax = ax.flatten()
        # ax = ax[0] + ax[1]

    aucs = {}
    fig.suptitle(f"Model Precision-Recall Curves Split By {cat}")
    for i, m in enumerate(models):
        title_str = m
        # if do_sigtest:
        #     z, p = hanley_mcneil_sigtest(df_catinfo, skips, prcs[m])
        #     title_str = f"{m}\n(z={z:.6f}, p={p:.6f})"
        #     bin_sigtest_results[m] = {"z": z, "p": p}

        aucs[m] = ax_prcs(ax[i], prcs[m], title=title_str)

    plt.tight_layout()
    plt.show()

    df_sigtest_results = None
    # if do_sigtest:
    #     df_sigtest_results = pd.DataFrame(bin_sigtest_results)

    return df_catinfo, aucs, df_sigtest_results


def roc_cm_by_category(
    df, cat, models=MODEL_TO_COL, min_mal=2, threshold=ILST_THRESHOLD
):
    df_catinfo, aucs, df_sigtest_results = roc_by_category(
        df, cat, models=MODEL_TO_COL, min_mal=2
    )

    groups = df.groupby(cat)
    perfs = {}
    for m in models:
        perfs[m] = threshold_metrics_by_splits(
            groups, pred_col=models[m], threshold=threshold
        )

    return df_catinfo, perfs, df_sigtest_results


## Hanley-McNeil (1982) significance test for comparing two independent AUCs
## From http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
def hanley_mcneil_sigtest(df_catinfo, skips, rocs):
    groups = list(set(df_catinfo.index.values) - set(skips))
    # if len(groups) != 2:
    #     return np.nan, np.nan

    aucs = {}
    ses = {}
    for g in groups:
        # print(g)
        auc = skl_metrics.auc(rocs[g].fpr_vals, rocs[g].mean_tpr_vals)
        # print("auc:", auc)
        q1 = auc / (2 - auc)
        q2 = (2 * auc * auc) / (1 + auc)
        # print("q1:", q1, "q2:", q2)
        n_mal = df_catinfo.loc[g]["num_mal"]
        n_ben = df_catinfo.loc[g]["num"] - df_catinfo.loc[g]["num_mal"]
        # print("mal:", n_mal, "ben:", n_ben)

        se = np.sqrt(
            (
                auc * (1 - auc)
                - (n_mal - 1) * (q1 - auc**2)
                + (n_ben - 1) * (q2 - auc**2)
            )
            / (n_mal * n_ben)
        )
        # print("se:", se)
        aucs[g] = auc
        ses[g] = se

    z = {g: {g: 0 for g in groups} for g in groups}
    p = {g: {g: 1 for g in groups} for g in groups}
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                # group1, group2 = groups[0], groups[1]

                auc_diff = aucs[group1] - aucs[group2]
                # print("aucdiff:", auc_diff)
                se_diff = np.sqrt(ses[group1] ** 2 + ses[group2] ** 2)
                # print("sediff:", se_diff)
                z[group1][group2] = auc_diff / se_diff
                # print("z:", z)
                p[group1][group2] = (
                    scipy.stats.norm.sf(abs(z[group1][group2])) * 2
                )  ## two-tailed p-value (Normal distribution)
                # print("p:", p)

    return pd.DataFrame(z), pd.DataFrame(p)


def prep_nlst_preds(df, scanlevel=True, sybil=True, tijmen=True):
    if scanlevel:
        nodule_drop_cols = [
            "CoordX",
            "CoordY",
            "CoordZ",
            "NoduleID",
            "AnnotationID",
        ]
        nodule_agg_cols = [
            "Spiculation",
            "Diameter [mm]",
            "NoduleCounts",
            "NoduleInUpperLung",
            "Solid",  # Nodule Types
            "GroundGlassOpacity",
            "Perifissural",
            "NonSolid",
            "PartSolid",
            "SemiSolid",
            "Calcified",
        ]
        model_cols = [
            "DL",
            "PanCan2b",
            "label",
            "Thijmen_mean",
            "Thijmen_global_hidden",
            "Thijmen_global_show",
            "Thijmen_local",
            "Thijmen_mean_cal",
            "Thijmen_global_hidden_cal",
            "Thijmen_global_show_cal",
            "Thijmen_local_cal",
            "DL_cal",
        ]
        # Not including Sybil here because it's already scan-level of course.

        df = df.drop(nodule_drop_cols, axis=1)
        dfgb = df.groupby("SeriesInstanceUID")

        for c in nodule_agg_cols + model_cols:
            df[c] = dfgb[c].transform("max")

        df = df.drop_duplicates(["SeriesInstanceUID"], ignore_index=True)

    if tijmen:
        df = df[(~df["Thijmen_mean"].isna())]
    if sybil:
        df = df[(~df["sybil_year1"].isna())]
    return df


def bmi_calc(height, weight):
    return (weight * 703) / (height * height)


def corrmat(df, rows, cols, method="kendall", vmin=-1, vmax=1, cmap="RdYlGn"):
    cols_list = list(set(rows).union(set(cols)))
    corrmat = df[cols_list].corr(method=method)

    plt.figure(figsize=(len(cols) * 0.6, len(rows) * 0.5))
    sns.heatmap(corrmat.loc[rows, cols], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.show()

    return corrmat


DEFAULT_POLICIES = (
    ("Sensitivity", 0.9),
    ("Sensitivity", 1.0),
    ("Specificity", 0.9),
    ("Specificity", 1.0),
    ("Youden J", 1.0),  ## Max J statistic
)


def get_threshold_policies(
    df, models=MODEL_TO_COL, policies=DEFAULT_POLICIES, brock=True, precision=3
):
    threshold_perfs = {}
    threshold_cands = np.arange(0, 1, 10 ** (-1 * precision))

    for m in models:
        stats = {}
        for t in threshold_cands:
            stats[np.around(t, precision)] = stats_from_cm(
                *cm_with_thres(df, threshold=t, pred_col=models[m], true_col="label")
            )

        statdf = pd.DataFrame(stats).T
        statdf["Sensitivity"] = statdf["tpr"]
        statdf["Specificity"] = statdf["tnr"]
        statdf["Youden J"] = statdf["j"]

        threshold_perfs[m] = statdf

    policy_thresholds = {}
    for col, val in policies:
        other_col = "Specificity" if col == "Sensitivity" else "Sensitivity"
        policy_thresholds[f"{col}={val}"] = {}

        for m in models:
            df = threshold_perfs[m]
            df[f"abs_diff"] = abs(df[col] - val)
            df = df.sort_values(by=["abs_diff", other_col], ascending=[True, False])
            df = df.drop(columns=["abs_diff"])
            policy_thresholds[f"{col}={val}"][m] = list(df.index.values)[0]

    policy_threshold_df = pd.DataFrame(policy_thresholds)
    if brock:
        policy_threshold_df["Brock"] = [ILST_THRESHOLD] * len(policy_threshold_df)

    return policy_threshold_df, threshold_perfs
