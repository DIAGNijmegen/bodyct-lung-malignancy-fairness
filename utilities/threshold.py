import pandas as pd
import os
import numpy as np
import itertools

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
        ci_hi = aggperfs.quantile((1 + ci_to_use) / 2, numeric_only=True)
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
    stats=None,
    show_mb_count=True,
):
    # df_catinfo = catinfo(df, cat)
    # if len(df_catinfo) == 0:
    #     return None
    # display(df_catinfo)

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

    subgroups = []
    category_info = (
        stats.groupby("group")[["mal", "ben"]]
        .mean()
        .sort_values(by="mal", ascending=False)
    )
    for val, row in category_info.iterrows():
        if val == "ALL":
            continue
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
        # print(p)
        for i, s in enumerate(plot_metrics):
            # print(s)
            x = np.arange(len(subgroups))  # the label locations
            width = 0.12  # the width of the bars
            multiplier = 0

            for k, m in enumerate(models):
                # print(m, f'(policy == "{p}") & (model == "{m}")')
                # display(stats)
                modelstats = stats.query(f'(policy == "{p}") & (model == "{m}")')
                # print(len(modelstats))
                scores, ci_lo, ci_hi, labels = [], [], [], []

                for g in subgroups:
                    # print(g)
                    # print(modelstats[modelstats["group"] == g])
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

                    group_label = g
                    if show_mb_count:
                        group_label += f"\n({subgroup_stats['mal']} mal, {subgroup_stats['ben']} ben)"
                    labels.append(group_label)

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
            ax[j][i].set_xticks(x + width, labels)
            # ax[j][i].legend(loc='upper left', bbox_to_anchor=(1, 1))

            if diff:
                ax[j][i].set_ylim(-0.55, 0.55)
            else:
                ax[j][i].set_ylim(0, 1.1)

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.suptitle(" \n ")
    fig.legend(handles, labels, loc="upper center", ncol=((len(handles) + 1) // 2))

    plt.tight_layout()
    if imgpath is not None:
        plt.savefig(imgpath, dpi=300)
    plt.show()

    return stats


## Check if subgroup performances are outside the confidence intervals of other subgroups.
## Take as input performances for one subgroup (stats from above).
def pairwise_comparisons_subgroups(stats, metric="fpr"):
    if "model" not in stats.columns:
        stats["model"] = list(stats.index)
    modelstats = stats.groupby("model")
    subgroups = list(pd.unique(stats["group"]))

    output_dfs = []
    for m, mdf in modelstats:
        policystats = mdf.groupby("policy")
        for p, pdf in policystats:
            df_array = []
            for sg_idx_1, sg_idx_2 in itertools.permutations(
                list(range(len(subgroups))), 2
            ):
                sg1, sg2 = subgroups[sg_idx_1], subgroups[sg_idx_2]
                row1 = pdf[pdf["group"] == sg1].iloc[0]
                row2 = pdf[pdf["group"] == sg2].iloc[0]
                info_dict = {
                    "comp_id": f"{metric}_{m}_{p}_{sg1 if (sg_idx_1 < sg_idx_2) else sg2}_{sg1 if (sg_idx_1 > sg_idx_2) else sg2}",
                    "model": m,
                    "policy": p,
                    "Group_1": sg1,
                    "Group_2": sg2,
                    f"{metric}_1": row1[metric],
                    f"{metric}-CI-lo_1": row1[f"{metric}_lo"],
                    f"{metric}-CI-hi_1": row1[f"{metric}_hi"],
                    f"{metric}_2": row2[metric],
                    f"{metric}-CI-lo_2": row2[f"{metric}_lo"],
                    f"{metric}-CI-hi_2": row2[f"{metric}_hi"],
                    f"{metric}_diff": row2[metric] - row1[metric],
                    f"{metric}_outside_CI": (
                        (row2[metric] > row1[f"{metric}_hi"])
                        or (row2[metric] < row1[f"{metric}_lo"])
                    ),
                    f"{metric}_CI_notouch": (
                        (row2[f"{metric}_lo"] > row1[f"{metric}_hi"])
                        or (row2[f"{metric}_hi"] < row1[f"{metric}_lo"])
                    ),
                    f"Group_1_mal": row1["mal"],
                    f"Group_2_mal": row2["mal"],
                    f"Group_1_ben": row1["ben"],
                    f"Group_2_ben": row2["ben"],
                    f"tp_1": row1["tp"],
                    f"tp_2": row2["tp"],
                    f"tn_1": row1["tn"],
                    f"tn_2": row2["tn"],
                    f"fp_1": row1["fp"],
                    f"fp_2": row2["fp"],
                    f"fn_1": row1["fn"],
                    f"fn_2": row2["fn"],
                }
                df_array.append(info_dict)

            result_df = pd.DataFrame(df_array)
            output_dfs.append(result_df)

    final_df = pd.concat(output_dfs, axis=0)
    return final_df


def all_results_subgroups_models(
    df,
    democols,  ### Shape: {'category1': ['attribute1', 'attribute2', ...], ...}
    policies,
    models=MODEL_TO_COL,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
    plot=False,
    csvpath=None,
):
    category_perfs = []
    for category in democols:
        attribute_perfs = []
        if plot:
            display(Markdown(f"## {category}"))

        for attribute in democols[category]:
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
                models=models,
                include_all=True,
                true_col=true_col,
                csvpath=None,
                bootstrap_ci=True,
                ci_to_use=ci_to_use,
                num_bootstraps=num_bootstraps,
                bootstrap_sample_size=None,
            )

            stats["col"] = [models[m] for m in stats["model"]]
            stats["attribute"] = [attribute] * len(stats)
            attribute_perfs.append(stats)

        if len(attribute_perfs) > 0:
            df_attribute_perfs = pd.concat(attribute_perfs, axis=0)
            df_attribute_perfs["category"] = [category] * len(df_attribute_perfs)
            category_perfs.append(df_attribute_perfs)

    if len(category_perfs) > 0:
        all_perfs = pd.concat(category_perfs, axis=0)

        if csvpath is not None:
            all_perfs.to_csv(f"{csvpath}.csv")
    else:
        all_perfs = None

    return all_perfs


def all_attribute_pairwise_comparisons(allstats, metric="fpr", models=MODEL_TO_COL):
    category_stats = allstats.groupby("category")
    comparison_dfs = []
    for c, category_df in category_stats:
        attribute_stats = category_df.groupby("attribute")
        for a, attribute_df in attribute_stats:
            comparison = pairwise_comparisons_subgroups(attribute_df, metric)
            comparison["col"] = [models[m] for m in comparison["model"]]
            comparison["attribute"] = [a] * len(comparison)
            comparison["category"] = [c] * len(comparison)
            comparison["comp_id"] = f"{c}_{a}_" + comparison["comp_id"]
            comparison_dfs.append(comparison)

    all_comparisons = pd.concat(comparison_dfs, axis=0)
    return all_comparisons


def save_results_isolate_confounders(
    df,
    demographic,  ## This is the one to get performance on.
    democols,  ## What to isolate for.
    policies,
    models=MODEL_TO_COL,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
    plot=False,
    csvpath=None,
):
    category_perfs = []
    for category in democols:
        attribute_perfs = []
        if plot:
            display(Markdown(f"## {category}"))

        for attribute in democols[category]:
            splits = df.groupby(attribute)

            for sval, sdf in splits:
                if plot:
                    display(Markdown(f"#### {demographic}: {attribute} = {sval}"))

                analysis_func = (
                    plot_threshold_stats_subgroups
                    if plot
                    else calc_threshold_stats_subgroups
                )

                stats = analysis_func(
                    sdf,
                    demographic,
                    policies,
                    models,
                    include_all=True,
                    true_col=true_col,
                    csvpath=None,
                    bootstrap_ci=True,
                    ci_to_use=ci_to_use,
                    num_bootstraps=num_bootstraps,
                    bootstrap_sample_size=None,
                )

                if stats is None:
                    continue

                stats["col"] = [models[m] for m in stats["model"]]
                stats["filter_by"] = [attribute] * len(stats)
                stats["filter_val"] = [sval] * len(stats)
                attribute_perfs.append(stats)

        if len(attribute_perfs) > 0:
            df_attribute_perfs = pd.concat(attribute_perfs, axis=0)
            df_attribute_perfs["category"] = [category] * len(df_attribute_perfs)
            category_perfs.append(df_attribute_perfs)

    if len(category_perfs) > 0:
        all_perfs = pd.concat(category_perfs, axis=0)

        if csvpath is not None:
            all_perfs.to_csv(csvpath)

        return all_perfs
    else:
        return None


def all_isolation_pairwise_comparisons(allstats, metric="fpr", models=MODEL_TO_COL):
    category_stats = allstats.groupby("category")
    comparison_dfs = []
    for cat, category_df in category_stats:
        confounder_stats = category_df.groupby("filter_by")
        for con, confounder_df in confounder_stats:
            subset_stats = confounder_df.group_by("filter_val")
            for sub, subset_df in subset_stats:
                comparison = pairwise_comparisons_subgroups(subset_df, metric)
                comparison["col"] = [models[m] for m in comparison["model"]]
                comparison["filter_val"] = [sub] * len(comparison)
                comparison["filter_by"] = [con] * len(comparison)
                comparison["category"] = [cat] * len(comparison)
                comparison["comp_id"] = f"{cat}_{con}_{sub}_" + comparison["comp_id"]
                comparison_dfs.append(comparison)

    all_comparisons = pd.concat(comparison_dfs, axis=0)
    return all_comparisons
