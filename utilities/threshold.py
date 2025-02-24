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

from .info import *
from .data import catinfo


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


## Plot function for multiple threshold performance curves for a single model. Need to make figure separately.
def ax_threshold_perfs(
    ax,
    perfs,
    metrics=["Sensitivity", "Specificity"],
    policy_df=None,
    model="Venkadesh",
    title=None,
):
    ax.set_xlabel("Threshold", fontsize=14)
    ax.set_ylabel("Rate", fontsize=14)
    ax.set_xticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # X axis ticks in steps of 0.1
    ax.set_yticks(
        np.arange(0, 1.1, 0.1), np.around(np.arange(0, 1.1, 0.1), 1), fontsize=12
    )  # Y axis ticks in steps of 0.1
    ax.grid(lw=1)
    ax.grid(visible=False)
    ax.set_xlim(-0.005, 1)
    ax.set_ylim(0, 1.005)

    perfs = perfs.sort_index(ascending=True)

    thresholds = list(perfs.index.values)
    for i, metric in enumerate(metrics):
        perf = perfs[metric]
        ax.plot(
            thresholds,
            perf,
            color=color_palette[i],
            label=metric,
        )

    for j, p in enumerate(list(policy_df.columns)):
        ax.axvline(
            policy_df.loc[model, p],
            linestyle="--",
            color=color_palette[j],
            label=f"{p} (threshold = {policy_df.loc[model, p]})",
        )

    if title:
        ax.set_title(title, fontsize=14)

    leg = ax.legend(loc="center right", fontsize=12)
    return


## Plot ROCs between models.
def plot_threshold_perfs_models(
    df,
    models=MODEL_TO_COL,
    perfs=None,
    policy_df=None,
    metrics=["Sensitivity", "Specificity"],
    precision=3,
    dataset_name="NLST",
    imgpath=None,
):
    if perfs is None:
        perfs = perfs_by_threshold_models(df, models=models, precision=precision)

    fig, ax = plt.subplots(len(models), 1, figsize=(8, 6 * len(models)))
    fig.suptitle(" ")

    for i, m in enumerate(models):
        subtitle = f"{m} Performance by Threshold ({dataset_name}, n={len(df)})"
        ax_threshold_perfs(
            ax=ax[i],
            perfs=perfs[m],
            metrics=metrics,
            policy_df=policy_df,
            title=subtitle,
        )

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)

    plt.tight_layout()
    plt.show()
    return perfs, policy_df


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
            df0 = perfs[m]
            df0["diff"] = df0[col] - val

            if val == 1.0:
                df = df0.query(f'diff == {max(df0["diff"])}')
            else:
                df = df0.query(f"diff >= 0")

            df = df.sort_values(by=["diff", other_col], ascending=[True, False])
            df0 = df0.drop(columns=["diff"])
            perfs[m] = df0
            policy_thresholds[f"{col}={val}"][m] = list(df.index.values)[0]

    policy_threshold_df = pd.DataFrame(policy_thresholds)
    if brock:
        policy_threshold_df["Brock"] = [ILST_THRESHOLD] * len(policy_threshold_df)

    return policy_threshold_df


def get_threshold_policies(
    df, models=MODEL_TO_COL, policies=THRESHOLD_POLICIES, brock=True, precision=3
):
    perfs = perfs_by_threshold_models(df, models, precision)
    policies = threshold_policies_models(perfs, policies, brock)
    return policies, perfs


def threshold_stats_models(df, policies, models=MODEL_TO_COL, true_col="label"):
    dfs_by_policy = []

    for p in list(policies.columns):
        metrics_by_model = {}
        for m in list(policies.index.values):
            threshold = policies.loc[m, p]
            metrics_by_model[m] = threshold_stats(
                df, threshold, pred_col=models[m], true_col=true_col
            )

        dfm = pd.DataFrame(metrics_by_model).T
        dfm["model"] = list(policies.index.values)
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
        stats["group"] = ["ALL"] * len(stats)
        stat_dfs.append(stats)

    ## Get threshold statistics for subgroups.
    subgroups = df.groupby(cat, observed=True)
    for subg, dfg in subgroups:
        stats = threshold_stats_models(dfg, policies, models=models, true_col=true_col)
        stats["group"] = [subg] * len(stats)
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
            all_bootstraps.append(bootstrap_stats)

        df_all_bootstraps = pd.concat(all_bootstraps, axis=0, ignore_index=False)
        aggperfs = df_all_bootstraps.groupby(level=0)
        ci_lo = aggperfs.quantile((1 - ci_to_use) / 2, numeric_only=True)
        ci_hi = aggperfs.quantile(ci_to_use + ((1 - ci_to_use) / 2), numeric_only=True)
        ci_df = pd.merge(
            ci_lo, ci_hi, left_index=True, right_index=True, suffixes=("_lo", "_hi")
        )

        allstats = pd.merge(
            allstats, ci_df, left_index=True, right_index=True, suffixes=("", "")
        )

    if csvpath:
        allstats.to_csv(csvpath, index=False)

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
    min_mal=10,
    dataset_name="NLST",
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

    subgroups = []
    for val, row in df_catinfo.iterrows():
        if row["mal"] >= min_mal:
            subgroups.append(val)

    if len(subgroups) < 2:
        "Not enough malignant samples from multiple groups. SKIP"
        return stats

    if show_all:
        subgroups.insert(0, "ALL")

    figheight = 1 + len(policies.columns) * 5
    figwidth = (
        (len(models) * 0.7 + 0.5)
        * (len(subgroups) + (1 if show_all else 0))
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

    color_palette = sns.color_palette("colorblind", len(models))
    for j, p in enumerate(list(policies.columns)):
        for i, s in enumerate(plot_metrics):
            x = np.arange(len(subgroups))  # the label locations
            width = 0.12  # the width of the bars
            multiplier = 0

            for k, m in enumerate(models):
                modelstats = stats.query(f'(policy == "{p}") & (model == "{m}")')
                scores, ci_lo, ci_hi, labels = [], [], [], []

                for g in subgroups:
                    subgroup_stats = modelstats[modelstats["group"] == g].iloc[0]

                    if diff:
                        allgroup_stats = modelstats[modelstats["group"] == "ALL"]
                        scores.append(subgroup_stats[s] - allgroup_stats.iloc[0][s])
                    else:
                        scores.append(subgroup_stats[s])

                    if bootstrap_ci:
                        ci_lo.append(subgroup_stats[s] - subgroup_stats[f"{s}_lo"])
                        ci_hi.append(subgroup_stats[f"{s}_hi"] - subgroup_stats[s])
                    else:
                        ci_lo.append(None)
                        ci_hi.append(None)

                    labels.append(
                        f"{g}\n({subgroup_stats['mal']} mal, {subgroup_stats['ben']} ben)"
                    )

                offset = width * multiplier
                rects = ax[j][i].bar(
                    x + offset,
                    scores,
                    width,
                    label=m,
                    yerr=[ci_lo, ci_hi],
                    color=color_palette[k],
                )
                ax[j][i].bar_label(rects, padding=3, fmt="%.2f", fontsize="x-small")
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax[j][i].set_ylabel(s)
            ax[j][i].set_title(f"{dataset_name} (n={len(df)}) {s} by {cat} ({p})")
            ax[j][i].set_xticks(x + width + 0.2, labels)
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
    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)
    plt.show()

    return stats


def all_results_subgroups_models(
    df,
    democols,  ### Shape: {'category1': ['attribute1', 'attribute2', ...], ...}
    policies,
    models=MODEL_TO_COL,
    true_col="label",
    metrics=["fpr", "fnr"],
    ci_to_use=0.95,
    num_bootstraps=100,
    plot=False,
    dirpath=None,
):
    all_perfs_metric = {}
    for met in metrics:
        category_perfs = []
        for category in democols:
            attribute_perfs = []
            if plot:
                display(Markdown(f"## {category}"))

            for attribute in democols[category]:
                df_catinfo = catinfo(df, attribute)
                top2 = df_catinfo.sort_values(by="mal", ascending=False).index.values[
                    0:2
                ]

                if plot:
                    display(Markdown(f"### {attribute}"))

                analysis_func = (
                    plot_threshold_stats_subgroups
                    if plot
                    else calc_threshold_stats_subgroups
                )

                stats = analysis_func(
                    df,
                    attribute,
                    policies,
                    models,
                    include_all=False,
                    true_col=true_col,
                    csvpath=None,
                    bootstrap_ci=True,
                    ci_to_use=ci_to_use,
                    num_bootstraps=num_bootstraps,
                    bootstrap_sample_size=None,
                )

                if stats is None:
                    continue

                filtered_stats = [
                    stats[stats["group"] == r][
                        ["model", "policy", "threshold", "group"]
                        + [met, f"{met}_lo", f"{met}_hi"]
                    ].sort_values(by="policy")
                    for r in top2
                ]

                bintable = pd.merge(
                    filtered_stats[0],
                    filtered_stats[1],
                    on=["model", "policy", "threshold"],
                    left_index=False,
                    right_index=False,
                    suffixes=("_1", "_2"),
                )

                for i, g in enumerate(top2):
                    bintable[f"Group_{i+1}_mal"] = df_catinfo.loc[g, "mal"]
                    bintable[f"Group_{i+1}_ben"] = df_catinfo.loc[g, "ben"]
                    bintable[f"Group_{i+1}_pct"] = df_catinfo.loc[g, "pct"]
                    bintable[f"Group_{i+1}_pct_mal"] = df_catinfo.loc[g, "pct_mal"]

                bintable["col"] = [models[m] for m in bintable["model"]]
                bintable["attribute"] = [attribute] * len(bintable)
                attribute_perfs.append(bintable)

            if len(attribute_perfs) > 0:
                df_attribute_perfs = pd.concat(attribute_perfs, axis=0)
                df_attribute_perfs["category"] = [category] * len(df_attribute_perfs)
                category_perfs.append(df_attribute_perfs)

        if len(category_perfs) > 0:
            all_perfs = pd.concat(category_perfs, axis=0)

            if dirpath is not None:
                os.makedirs(dirpath, exist_ok=True)
                all_perfs.to_csv(f"{dirpath}/")

            all_perfs_metric[met] = all_perfs
        else:
            all_perfs_metric[met] = None

    return all_perfs_metric
