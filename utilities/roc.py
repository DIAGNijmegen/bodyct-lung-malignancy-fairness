import pandas as pd
import numpy as np

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt
import scipy.stats
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
color_palette = sns.color_palette("colorblind", 30)


## Return BootstrappedROCCICurves (evalutils object) for one model on one group.
def calc_roc(df, pred_col, true_col="label", ci_to_use=0.95, num_bootstraps=100):
    y_true = df[true_col].values
    y_pred = df[pred_col].values

    roc = get_bootstrapped_roc_ci_curves(
        y_pred, y_true, num_bootstraps=num_bootstraps, ci_to_use=ci_to_use
    )

    auc = skl_metrics.auc(roc.fpr_vals, roc.mean_tpr_vals)
    auc_info = {"score": auc, "ci-hi": roc.high_az_val, "ci-lo": roc.low_az_val}
    return roc, auc_info


## Check if the label has all true or all false.
## If this is the case, then the ROC will not compute (and useless to do so).
def check_if_roc_ok(df, true_col="label", min_mal=2):
    total = len(df[true_col])
    mal_count = len(df.query(f"{true_col} == 1")[true_col])
    if mal_count == total:
        return False
    if mal_count < min_mal:
        return False
    return True


## Hanley-McNeil (1982) significance test for comparing two independent AUCs
## From http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
def roc_hm_error(auc, n_mal, n_ben):
    # print("auc:", auc)
    q1 = auc / (2 - auc)
    q2 = (2 * auc * auc) / (1 + auc)
    # print("q1:", q1, "q2:", q2)

    se = np.sqrt(
        (auc * (1 - auc) - (n_mal - 1) * (q1 - auc**2) + (n_ben - 1) * (q2 - auc**2))
        / (n_mal * n_ben)
    )
    return se


## Hanley-McNeil (1982) significance test for comparing AUCs of the SAME treatment between INDEPENDENT subgroups.
## See roc_hm_error for calculating the standard error.
## From http://www.med.mcgill.ca/epidemiology/hanley/software/Hanley_McNeil_Radiology_82.pdf
def roc_hm_pairwise_sigtest(aucs):
    z = {g: {g: 0 for g in aucs} for g in aucs}
    p = {g: {g: 1 for g in aucs} for g in aucs}
    for group1 in aucs:
        for group2 in aucs:
            if group1 != group2:
                auc1 = aucs[group1]["score"]
                auc2 = aucs[group2]["score"]
                se1 = aucs[group1]["error"]
                se2 = aucs[group2]["error"]

                auc_diff = auc1 - auc2
                # print("aucdiff:", auc_diff)
                se_diff = np.sqrt(se1**2 + se2**2)
                # print("sediff:", se_diff)

                z[group1][group2] = auc_diff / se_diff
                # print("z:", z)

                p[group1][group2] = (
                    scipy.stats.norm.sf(abs(z[group1][group2])) * 2
                )  ## two-tailed p-value (Normal distribution)
                # print("p:", p)

    return pd.DataFrame(z), pd.DataFrame(p)


## ROC test for different subgroups for a single model.
def calc_rocs_subgroups(
    df,
    cat,
    pred_col,
    include_all=False,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
):
    rocs, aucs = {}, {}
    ## If we want to include the overall result for comparison.
    if include_all:
        rocs["ALL"], aucs["ALL"] = calc_roc(
            df,
            pred_col,
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
        )

    ## Get ROC and AUC for subgroups.
    subgroups = df.groupby(cat, observed=False)
    for subg, dfg in subgroups:
        is_roc_ok = check_if_roc_ok(dfg)
        if not is_roc_ok:
            continue

        rocs[subg], aucs[subg] = calc_roc(
            dfg,
            pred_col,
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
        )

        ## Calculate Hanley-McNeil standard errors.
        n_mal = len(dfg.query(f"{true_col} == 1"))
        n_ben = len(dfg.query(f"{true_col} == 0"))
        aucs[subg]["error"] = roc_hm_error(aucs[subg]["score"], n_mal, n_ben)

    ## Perform Hanley-McNeil significance test (pairwise between subgroups).
    z, p = roc_hm_pairwise_sigtest(aucs)

    aucs = pd.DataFrame(aucs).T
    return rocs, aucs, z, p


## Calculate ROCs for different models (no subgroups).
## Future: could add DeLong test over here.
## Note: models should be dictionary of format {'label': 'model_column"}
def calc_rocs_models(
    df, models=MODEL_TO_COL, true_col="label", ci_to_use=0.95, num_bootstraps=100
):
    rocs, aucs = {}, {}
    for m in models:
        rocs[m], aucs[m] = calc_roc(
            df,
            models[m],
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
        )

    return rocs, aucs


## Calculate ROCs for models for subgroups.
def calc_rocs_subgroups_models(
    df,
    cat,
    models=MODEL_TO_COL,
    include_all=False,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
):
    rocs, aucs, zs, ps = {}, {}, {}, {}
    for m in models:
        rocs[m], aucs[m], zs[m], ps[m] = calc_rocs_subgroups(
            df,
            cat,
            models[m],
            include_all=include_all,
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
        )

    return rocs, aucs, zs, ps


def binary_group_roc_table(aucs, p, subgroups, z=None):
    assert len(subgroups) == 2
    tablerow = {}
    for m in aucs:
        tablerow[m] = {"p": p[m].loc[subgroups[0], subgroups[1]]}
        if z is not None:
            tablerow[m]["z"] = z[m].loc[subgroups[0], subgroups[1]]

        for i in range(2):
            g = subgroups[i]
            tablerow[m][f"Group_{i+1}"] = g
            tablerow[m][f"AUC_{i+1}"] = aucs[m].loc[g, "score"]
            tablerow[m][f"AUC-CI-lo_{i+1}"] = aucs[m].loc[g, "ci-lo"]
            tablerow[m][f"AUC-CI-hi_{i+1}"] = aucs[m].loc[g, "ci-hi"]

        tablerow[m]["AUC_diff"] = tablerow[m][f"AUC_2"] - tablerow[m][f"AUC_1"]

    return pd.DataFrame(tablerow).T


## General plotting function for multiple ROC curves. Need to make figure separately.
def ax_rocs(ax, rocs, title=None, plot_ci=True, catinfo=None):
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

    for i, label in enumerate(rocs):
        roc = rocs[label]
        auc = skl_metrics.auc(roc.fpr_vals, roc.mean_tpr_vals)
        legend_label = f"{label}: AUC = {auc:.2f} (95% CI: {roc.low_az_val:.2f}, {roc.high_az_val:.2f})"
        if catinfo is not None:
            legend_label = f"{label} ({catinfo.loc[label, 'mal']} mal, {catinfo.loc[label, 'ben']} ben): \nAUC = {auc:.2f} ({roc.low_az_val:.2f}, {roc.high_az_val:.2f})"

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
    return


## Plot ROCs between models.
def plot_rocs_models(
    df,
    models=MODEL_TO_COL,
    rocs=None,
    aucs=None,
    dataset_name="NLST",
    title=None,
    imgpath=None,
    plot_ci=False,
    figsize=(6, 6),
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
):
    if rocs is None:
        rocs, aucs = calc_rocs_models(df, models, true_col, ci_to_use, num_bootstraps)
    if title is None:
        title = f"{dataset_name} (n={len(df)}) ROC Curves Across Models"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax_rocs(ax=ax, rocs=rocs, title=title, plot_ci=plot_ci)

    if imgpath is not None:
        plt.savefig(imgpath, dpi=600)

    plt.show()
    return rocs, aucs


def plot_rocs_subgroups(
    df,
    cat,
    models=MODEL_TO_COL,
    roc=None,
    auc=None,
    z=None,
    p=None,
    two_subgroups=False,
    dataset_name="NLST Scans",
    figheight=5,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
    imgpath=None,
    subplots_per_row=3,
):
    df_catinfo = catinfo(df, cat)
    display(df_catinfo)

    if (roc is None) or (auc is None) or (z is None) or (p is None):
        roc, auc, z, p = calc_rocs_subgroups_models(
            df,
            cat,
            models,
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
            include_all=False,
        )

    first_roc = roc[list(roc.keys())[0]]
    if len(first_roc) < 2:
        print("Less than 2 valid groups. SKIP")
        return roc, auc, z, p

    top2_groups = None
    if two_subgroups == True:
        top2_groups = df_catinfo.sort_values(by="mal", ascending=False).index.values[
            0:2
        ]
        new_roc = {m: {g: roc[m][g] for g in top2_groups} for m in models}
        roc = new_roc

    if len(first_roc) == 2:
        # top2_groups = list(first_roc.keys())
        top2_groups = list(
            df_catinfo[df_catinfo.index.isin(list(first_roc.keys()))]
            .sort_values(by="mal", ascending=False)
            .index.values
        )
        two_subgroups = True

    if len(models) <= subplots_per_row:
        fig, ax = plt.subplots(
            1, len(models), figsize=(figheight * len(models), figheight), squeeze=False
        )
    else:
        lm = len(models)
        overflow = lm % subplots_per_row
        overall_height = (lm // subplots_per_row) + (overflow > 0)
        fig, ax = plt.subplots(
            overall_height,
            subplots_per_row,
            figsize=(figheight * subplots_per_row, figheight * overall_height),
            squeeze=False,
        )

    ax = ax.flatten()

    # fig.suptitle(f"{dataset_name} (n={len(df)}) Model ROC Curves Split By {cat}")

    for i, m in enumerate(models):
        title_str = f"{m} on {dataset_name} (n={len(df)}) \nSplit by {cat}"

        if two_subgroups:
            z_show = z[m].loc[top2_groups[0], top2_groups[1]]
            p_show = p[m].loc[top2_groups[0], top2_groups[1]]

            if p_show < 0.001:
                title_str += f" (z={z_show:.2f}, p<0.001)"
            else:
                title_str += f" (z={z_show:.2f}, p={truncate_p(p_show)})"

        ax_rocs(ax[i], roc[m], title=title_str, catinfo=None, plot_ci=True)

    plt.tight_layout()
    if imgpath is not None:
        plt.savefig(imgpath, dpi=300)
    plt.show()

    if two_subgroups:
        display(binary_group_roc_table(auc, p, top2_groups))

    return roc, auc, z, p


def all_results_subgroups_models(
    df,
    democols,  ### Shape: {'category1': ['attribute1', 'attribute2', ...], ...}
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
            df_catinfo = catinfo(df, attribute)
            top2 = df_catinfo.sort_values(by="mal", ascending=False).index.values[0:2]

            analysis_func = plot_rocs_subgroups if plot else calc_rocs_subgroups_models
            if plot:
                display(Markdown(f"### {attribute}"))

            _, auc, z, p = analysis_func(
                df,
                attribute,
                models=models,
                true_col=true_col,
                ci_to_use=ci_to_use,
                num_bootstraps=num_bootstraps,
            )

            if auc is None:
                continue

            first_auc = auc[list(auc.keys())[0]]
            if len(first_auc) < 2:
                continue

            bintable = binary_group_roc_table(auc, p, top2, z)

            for i, g in enumerate(top2):
                bintable[f"Group_{i+1}_mal"] = df_catinfo.loc[g, "mal"]
                bintable[f"Group_{i+1}_ben"] = df_catinfo.loc[g, "ben"]
                bintable[f"Group_{i+1}_pct"] = df_catinfo.loc[g, "pct"]
                bintable[f"Group_{i+1}_pct_mal"] = df_catinfo.loc[g, "pct_mal"]

            bintable["col"] = [models[m] for m in list(bintable.index.values)]
            bintable["attribute"] = [attribute] * len(bintable)
            attribute_perfs.append(bintable)

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


def plot_rocs_isolate_confounder(
    df,
    cat,
    confounder,
    models=MODEL_TO_COL,
    dataset_name="NLST Scans",
    figheight=5,
    true_col="label",
    ci_to_use=0.95,
    num_bootstraps=100,
    imgpath=None,
):
    subsets = df.groupby(confounder)
    fig, ax = plt.subplots(
        2,
        len(models),
        figsize=(figheight * len(models), 2.2 * figheight),
        squeeze=False,
    )
    bintables = []

    for i, (subset_name, subset_df) in enumerate(subsets):
        roc, auc, z, p = calc_rocs_subgroups_models(
            subset_df,
            cat,
            models=models,
            include_all=False,
            true_col=true_col,
            ci_to_use=ci_to_use,
            num_bootstraps=num_bootstraps,
        )

        dfc = catinfo(subset_df, cat)
        two_subgroups = len(dfc) >= 2
        top2_groups = None

        if two_subgroups:
            top2_groups = dfc.sort_values(by="mal", ascending=False).index.values[0:2]
            new_roc = {m: {g: roc[m][g] for g in top2_groups} for m in models}
            roc = new_roc

        for j, m in enumerate(models):
            title_str = f"{m} Split by {cat}\n{dataset_name}, {confounder}: {subset_name} (n={len(subset_df)})"

            if two_subgroups:
                z_show = z[m].loc[top2_groups[0], top2_groups[1]]
                p_show = p[m].loc[top2_groups[0], top2_groups[1]]

                if p_show < 0.001:
                    title_str += f"\n(z={z_show:.2f}, p<0.001)"
                else:
                    title_str += f"\n(z={z_show:.2f}, p={truncate_p(p_show)})"

            ax_rocs(ax[i][j], roc[m], title=title_str, catinfo=None, plot_ci=True)

        if two_subgroups:
            bintable = binary_group_roc_table(auc, p, top2_groups, z)
            bintable["filter_val"] = [subset_name] * len(bintable)
            bintables.append(bintable)

    if len(bintables) > 0:
        allbintable = pd.concat(bintables, axis=0)

    plt.tight_layout()
    if imgpath is not None:
        plt.savefig(imgpath, dpi=300)
    plt.show()

    return allbintable


def save_results_isolate_confounders(
    df,
    demographic,  ## This is the one to get performance on.
    democols,  ## What to isolate for.
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
                df_catinfo = catinfo(sdf, demographic)
                top2 = df_catinfo.sort_values(by="mal", ascending=False).index.values[
                    0:2
                ]

                analysis_func = (
                    plot_rocs_subgroups if plot else calc_rocs_subgroups_models
                )
                if plot:
                    display(Markdown(f"### {demographic}: {attribute} == {sval}"))

                _, auc, z, p = analysis_func(
                    sdf,
                    demographic,
                    models=models,
                    true_col=true_col,
                    ci_to_use=ci_to_use,
                    num_bootstraps=num_bootstraps,
                )

                if auc is None:
                    continue

                first_auc = auc[list(auc.keys())[0]]
                if len(first_auc) < 2:
                    continue

                bintable = binary_group_roc_table(auc, p, top2, z)

                for i, g in enumerate(top2):
                    bintable[f"Group_{i+1}_mal"] = df_catinfo.loc[g, "mal"]
                    bintable[f"Group_{i+1}_ben"] = df_catinfo.loc[g, "ben"]
                    bintable[f"Group_{i+1}_pct"] = df_catinfo.loc[g, "pct"]
                    bintable[f"Group_{i+1}_pct_mal"] = df_catinfo.loc[g, "pct_mal"]

                bintable["col"] = [models[m] for m in list(bintable.index.values)]
                bintable["filter_by"] = [attribute] * len(bintable)
                bintable["filter_val"] = [sval] * len(bintable)
                attribute_perfs.append(bintable)

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
