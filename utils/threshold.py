import pandas as pd
import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

import sklearn.metrics as skl_metrics
from sklearn.utils import resample
import warnings
from IPython.display import display, Markdown
from info import *
from data import catinfo


def confmat(df, threshold=ILST_THRESHOLD, pred_col="DL", true_col="label"):
    y_true = df[true_col].to_numpy()
    y_pred = (df[pred_col] > threshold).astype(int).to_numpy()
    tn, fp, fn, tp = skl_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp, tn, fp, fn


def threshold_stats(df, threshold=ILST_THRESHOLD, pred_col="DL", true_col="label"):
    tp, tn, fp, fn = confmat(df, threshold, pred_col, true_col)
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


def perfs_by_threshold_models(df, models=MODEL_TO_COL, precision=3):
    threshold_perfs = {}
    threshold_cands = np.arange(0, 1, 10 ** (-1 * precision))

    for m in models:
        stats = {}
        for t in threshold_cands:
            stats[np.around(t, precision)] = threshold_stats(
                df, threshold=t, pred_col=models[m], true_col="label"
            )

        statdf = pd.DataFrame(stats).T
        statdf["Sensitivity"] = statdf["tpr"]
        statdf["Specificity"] = statdf["tnr"]
        statdf["Youden J"] = statdf["j"]

        threshold_perfs[m] = statdf

    return threshold_perfs


def threshold_policies_models(
    perfs=None,
    policies=THRESHOLD_POLICIES,
    brock=True,
):
    policy_thresholds = {}
    for col, val in policies:
        other_col = "Specificity" if col == "Sensitivity" else "Sensitivity"
        policy_thresholds[f"{col}={val}"] = {}

        for m in perfs:
            df = perfs[m]
            df[f"abs_diff"] = abs(df[col] - val)
            df = df.sort_values(by=["abs_diff", other_col], ascending=[True, False])
            df = df.drop(columns=["abs_diff"])
            policy_thresholds[f"{col}={val}"][m] = list(df.index.values)[0]

    policy_threshold_df = pd.DataFrame(policy_thresholds)
    if brock:
        policy_threshold_df["Brock"] = [ILST_THRESHOLD] * len(policy_threshold_df)

    return policy_threshold_df


def threshold_stats_models(df, policies, models=MODEL_TO_COL, true_col="label"):
    dfs_by_policy = []

    for p in policies.items():
        metrics_by_model = {}
        for m in policies.index.values():
            threshold = policies.loc[m, p]
            metrics_by_model[m] = threshold_stats(
                df, threshold, pred_col=models[m], true_col=true_col
            )

        dfm = pd.DataFrame(metrics_by_model)
        dfm["model"] = policies.index.values()
        dfm["policy"] = [p] * len(dfm)
        dfm["threshold"] = [threshold] * len(dfm)
        dfs_by_policy.append(dfm)

    mega_stats_df = pd.concat(dfs_by_policy, axis=0, ignore_index=True)
    return mega_stats_df


## Threshold stats for: multiple subgroups, multiple models, multiple threshold policies.
def calc_threshold_stats_subgroups(
    df,
    cat,
    policies,
    models=MODEL_TO_COL,
    include_all=False,
    true_col="label",
    csvpath=None,
    bootstrap_ci=True,
    ci_to_use=0.95,
    num_bootstraps=100,
    bootstrap_sample_size=None,
):
    stat_dfs = []
    ## If we want to include the overall result for comparison.
    if include_all:
        stats = threshold_stats_models(df, policies, models=models, true_col=true_col)
        stats["Group"] = ["ALL"] * len(stats)
        stat_dfs.append(stats)

    ## Get threshold statistics for subgroups.
    subgroups = df.groupby(cat)
    for subg, dfg in subgroups:
        stats = threshold_stats_models(dfg, policies, models=models, true_col=true_col)
        stats["Group"] = [subg] * len(stats)
        stat_dfs.append(stats)

    allstats = pd.concat(stat_dfs, axis=0, ignore_index=True)

    if bootstrap_ci:
        df0 = df.dropna(axis=0, subset=[cat])[list(models.values()) + ["label", cat]]
        all_bootstraps = []
        for it in range(num_bootstraps):
            bootstrap_df = resample(
                df0,
                replace=True,
                n_samples=bootstrap_sample_size,
                stratify=df0[cat],
                random_state=None,
            )
            bootstrap_stats = calc_threshold_stats_subgroups(
                bootstrap_df,
                cat,
                policies,
                models,
                include_all,
                true_col,
                csvpath=None,
                bootstrap_ci=False,  #### MUST BE FALSE TO STOP INFINITE RECURSION
            )
            bootstrap_stats["iter"] = [it] * len(bootstrap_stats)

        df_all_bootstraps = pd.concat(all_bootstraps, axis=0, ignore_index=False)
        aggperfs = df_all_bootstraps.groupby(level=0)
        ci_lo = aggperfs.quantile((1 - ci_to_use) / 2)
        ci_hi = aggperfs.quantile(ci_to_use + ((1 - ci_to_use) / 2))
        ci_df = pd.merge(
            ci_lo, ci_hi, left_index=True, right_index=True, suffixes=("_lo", "_hi")
        )

        allstats = pd.merge(
            allstats, ci_df, left_index=True, right_index=True, suffixes=("", "")
        )
        allstats = pd.concat(
            [allstats.loc[["ALL"], :], allstats.drop("ALL", axis=0)], axis=0
        )

    if csvpath:
        pd.to_csv(csvpath, index=False)

    return allstats


def plot_threshold_stats_subgroups(
    df,
    cat,
    policies,
    stats=None,
    models=MODEL_TO_COL,
    plot_metrics=["fpr", "fnr"],
    show_all=False,
    diff=True,
    dataset_name="NLST",
    title=None,
    imgpath=None,
    include_all=False,
    true_col="label",
    csvpath=None,
    bootstrap_ci=True,
    ci_to_use=0.95,
    num_bootstraps=100,
    bootstrap_sample_size=None,
):
    if diff:
        show_all = False
    if show_all:
        diff = False
    if show_all or diff:
        include_all = True

    if stats is None:
        stats = calc_threshold_stats_subgroups(
            df,
            cat,
            policies,
            models,
            include_all,
            true_col,
            csvpath,
            bootstrap_ci,
            ci_to_use,
            num_bootstraps,
            bootstrap_sample_size,
        )

    df_catinfo = catinfo(df, cat)
    display(df_catinfo)

    subgroups = pd.unique()
    num_models = None
    num_policies = None
    num_metrics = len(plot_metrics)

    # if temp_perfs is None:
    #     print("Not enough malignant samples from multiple groups. SKIP")
    #     return

    figheight = 1 + len(policies.columns) * 5
    figwidth = (
        4
        * (
            len(list(temp_perfs.values())[0])
            + (0 if (diff or (show_all == False)) else 1)
        )
        * len(plot_metrics)
    )
    fig, ax = plt.subplots(
        len(policies.columns),
        len(plot_metrics),
        figsize=(figwidth, figheight),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    for j, policy in enumerate(list(policies.columns)):
        # _, perfs = modelstats_by_cat(
        #     df,
        #     c,
        #     policies=policies,
        #     threshold_policy=policy,
        #     bootstrap=bootstrap,
        #     models=models,
        # )
        first_df = list(perfs.values())[0]

        for i, metric in enumerate(plot_metrics):
            category = list(first_df.index.values)

            if diff:
                category.remove("ALL")
                perf_metrics = {
                    m: list(perfs[m][metric] - perfs[m].loc["ALL", metric])[1:]
                    for m in perfs
                }

                if bootstrap:
                    ci_metrics = {
                        m: [
                            list(perfs[m][f"{metric}"] - perfs[m][f"{metric}_lo"])[1:],
                            list(perfs[m][f"{metric}_hi"] - perfs[m][f"{metric}"])[1:],
                        ]
                        for m in perfs
                    }
                else:
                    ci_metrics = {m: None for m in perfs}
            else:
                start_idx = 0
                if show_all == False:
                    category.remove("ALL")
                    start_idx = 1

                perf_metrics = {m: list(perfs[m][metric])[start_idx:] for m in perfs}
                if bootstrap:
                    ci_metrics = {
                        m: [
                            list(perfs[m][f"{metric}"] - perfs[m][f"{metric}_lo"])[
                                start_idx:
                            ],
                            list(perfs[m][f"{metric}_hi"] - perfs[m][f"{metric}"])[
                                start_idx:
                            ],
                        ]
                        for m in perfs
                    }
                else:
                    ci_metrics = {m: None for m in perfs}

            x = np.arange(len(category))  # the label locations
            width = 0.12  # the width of the bars
            multiplier = 0

            for model, score in perf_metrics.items():
                offset = (width) * ((multiplier))
                rects = ax[j][i].bar(
                    x + offset, score, width, label=model, yerr=ci_metrics[model]
                )
                ax[j][i].bar_label(rects, padding=3, fmt="%.2f", fontsize="x-small")
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax[j][i].set_ylabel(metric)
            ax[j][i].set_title(
                f"{DATASET_NAME} (n={len(df)}) {metric} by {c} (thresholds: {policy})"
            )
            ax[j][i].set_xticks(
                x + width,
                [
                    f"{val}\n({first_df.loc[val, 'mal']} mal, {first_df.loc[val, 'ben']} ben)"
                    for val in category
                ],
            )
            # ax[j][i].set_xticks(x + width, category)
            # ax[j][i].legend(loc='upper left', bbox_to_anchor=(1, 1))

            if diff:
                ax[j][i].set_ylim(-0.5, 0.5)
            else:
                ax[j][i].set_ylim(0, 1)

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.suptitle(" \n ")
    fig.legend(handles, labels, loc="upper center", ncol=1 + (len(handles) // 2))
    plt.tight_layout()
    plt.show()
