import pandas as pd
import os
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import json
from IPython.display import display, Markdown
import sys

sys.path.append("../")
sys.path.append("./")

from evalutils.roc import get_bootstrapped_roc_ci_curves
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from . import data
from . import roc
from . import threshold

TABLE_SCORE_PRECISION = 2

COL_TO_MODEL = {
    "DL_cal": "Venkadesh21",
    "Ensemble_Kiran_cal": "Venkadesh21",  ## DLCST
    "thijmen_mean_cal": "De Haas Combined",  ## DLCST
    "Thijmen_mean_cal": "De Haas Combined",
    "Thijmen_local_cal": "De Haas Local",
    "Thijmen_global_hidden_cal": "De Haas Global",
    "Thijmen_global_show_cal": "De Haas Global (Shown Nodule)",
    "sybil_year1": "Sybil (Year 1)",
    "PanCan2b": "PanCan2b",
}


def latex_replace_arrowbrackets(s):
    return (
        str(s)
        .replace("≤", "$\leqslant$")
        .replace("≥", "$\geqslant$")
        .replace("<", "$<$")
        .replace(">", "$>$")
    )


def prettify_result_val(attribute, group):
    if ">" in attribute:
        attribute_value_split_list = attribute.split(">")
        new_attribute = data.rename_cols[attribute_value_split_list[0].strip()]

        if group == "ALL":
            return new_attribute, "ALL"

        new_group = "True"
        if str(group).lower() in ["true", "1", "1.0"]:
            # new_group = f"Above {int(attribute_value_split_list[1].strip())}"
            new_group = f"> {int(attribute_value_split_list[1].strip())}"
        else:
            # new_group = f"{int(attribute_value_split_list[1].strip())} or Below"
            new_group = f"≤ {int(attribute_value_split_list[1].strip())}"

    else:
        new_attribute = data.rename_cols[attribute]
        if group == "ALL":
            return new_attribute, "ALL"

        new_group = group
        if attribute in data.keys:
            if group == "True":
                new_group = True
            elif group == "False":
                new_group = False
            elif pd.isna(group):
                new_group = None
            else:
                new_group = int(float(group))
            new_group = data.keys[attribute][new_group]

        elif attribute in data.boolean_cols:
            if str(group).lower() in ["true", "1", "1.0"]:
                new_group = "True"
            elif str(group).lower() in ["false", "0", "0.0"]:
                new_group = "False"

    return new_attribute, new_group


def pretty_interval(row, precision, group_num=1, metric="AUC"):
    return f"{row[f'{metric}_{group_num}']:.{precision}f} ({row[f'{metric}-CI-lo_{group_num}']:.{precision}f}, {row[f'{metric}-CI-hi_{group_num}']:.{precision}f})"


DLCST_MODELCOLS = {
    "Venkadesh21": "Ensemble_Kiran_cal",
    # "De Haas Combined": "thijmen_mean_cal",
    "PanCan2b": "PanCan2b",
    "Sybil (Year 1)": "sybil_year1",
}

NLST_1172_MODELCOLS = {
    # "Venkadesh21": "DL_cal",
    "De Haas Combined": "Thijmen_mean_cal",
    "De Haas Local": "Thijmen_local_cal",
    "De Haas Global": "Thijmen_global_hidden_cal",
    # "Sybil (Year 1)": "sybil_year1",
    # "PanCan2b": "PanCan2b",
}

NLST_5911_MODELCOLS = {
    "Venkadesh21": "DL_cal",
    # "De Haas Local": "Thijmen_local_cal",
    # "De Haas Global": "Thijmen_global_hidden_cal",
    "Sybil (Year 1)": "sybil_year1",
    "PanCan2b": "PanCan2b",
}

RENAME_POLICIES = {
    "Sensitivity=0.9": "90% Sensitivity",
    "Specificity=0.9": "90% Specificity",
    "Brock": "Brock ILST (6%)",
}

RENAME_MODELS = {
    "Venkadesh": "Venkadesh21",
    "de Haas": "De Haas Combined",
    "de Haas Combined": "De Haas Combined",
    "Sybil year 1": "Sybil (Year 1)",
    "de Haas Local": "De Haas Local",
    "de Haas Global (hidden nodule)": "De Haas Global",
    "de Haas Global (shown nodule)": "De Haas Global (Shown Nodule)",
    "PanCan2b": "PanCan2b",
}


def roc_results_pretty(df, model_order_0, precision=TABLE_SCORE_PRECISION):
    model_order = [RENAME_MODELS[m] for m in model_order_0]
    df.rename(index=RENAME_MODELS, inplace=True)
    df.reset_index(inplace=True, names="model")

    ## Get AUC info in nice interval notation.
    df["Group_1_AUC"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=1), axis=1
    )
    df["Group_2_AUC"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=2), axis=1
    )

    df1 = df[
        ["model", "category", "attribute", "Group_1", "Group_1_AUC", "Group_1_mal"]
    ].rename(
        columns={
            "Group_1": "Group",
            "Group_1_AUC": "ROC AUC",
            "Group_1_mal": "Malignant Scans",
        }
    )
    df2 = df[
        ["model", "category", "attribute", "Group_2", "Group_2_AUC", "Group_2_mal", "p"]
    ].rename(
        columns={
            "Group_2": "Group",
            "Group_2_AUC": "ROC AUC",
            "Group_2_mal": "Malignant Scans",
        }
    )
    df_res = pd.concat([df1, df2])

    df_res["Category"] = df_res.apply(
        lambda row: data.rename_types[row["category"]], axis=1
    )
    df_res["Attribute"] = df_res.apply(
        lambda row: prettify_result_val(row["attribute"], row["Group"])[0],
        axis=1,
    )
    df_res["Group"] = df_res.apply(
        lambda row: prettify_result_val(row["attribute"], row["Group"])[1],
        axis=1,
    )
    df_res["p"] = df_res.apply(lambda row: data.truncate_p(row["p"]), axis=1)
    df_res = df_res.set_index(
        pd.MultiIndex.from_frame(df_res[["Category", "Attribute", "Group"]])
    )

    model_results = {m: dfm[["ROC AUC", "p"]] for m, dfm in df_res.groupby("model")}
    df_out = pd.concat(model_results, axis=1)
    df_out["Malignant Scans"] = df_res["Malignant Scans"].drop_duplicates()
    df_out = df_out.sort_values(by="Malignant Scans", ascending=False)
    df_out = df_out.sort_index(
        ascending=True, level=["Category", "Attribute"], sort_remaining=False
    )[model_order + ["Malignant Scans"]]
    return df_out


RENAME_METRICS = {"fpr": "FPR", "fnr": "FNR", "tpr": "TPR", "tnr": "TNR"}

RENAME_METRICS_hi = {f"{k}_hi": f"{v}_hi" for k, v in RENAME_METRICS.items()}
RENAME_METRICS_lo = {f"{k}_lo": f"{v}_lo" for k, v in RENAME_METRICS.items()}


def threshold_stats_pretty(df, policies, demographic_for_isolations=None):
    policies.rename(columns=RENAME_POLICIES, inplace=True)
    df["policy"] = df["policy"].replace(RENAME_POLICIES)
    df["model"] = df.apply(lambda row: COL_TO_MODEL[row["col"]], axis=1)
    df["category"] = df.apply(lambda row: data.rename_types[row["category"]], axis=1)

    if demographic_for_isolations is None:
        df["attribute2"] = df.apply(
            lambda row: prettify_result_val(row["attribute"], row["group"])[0], axis=1
        )
        df["group"] = df.apply(
            lambda row: prettify_result_val(row["attribute"], row["group"])[1], axis=1
        )
        df["attribute"] = df["attribute2"]

    else:
        df["filter_by2"] = df.apply(
            lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[0],
            axis=1,
        )
        df["filter_val"] = df.apply(
            lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[1],
            axis=1,
        )
        df["filter_by"] = df["filter_by2"]
        df["group"] = df.apply(
            lambda row: prettify_result_val(demographic_for_isolations, row["group"])[
                1
            ],
            axis=1,
        )

    df.rename(columns=RENAME_METRICS, inplace=True)
    df.rename(columns=RENAME_METRICS_lo, inplace=True)
    df.rename(columns=RENAME_METRICS_hi, inplace=True)

    return df, policies


def metric_outside_ci(val_is_outside_ci):
    return "*" if val_is_outside_ci else ""


def ci_dont_intersect(ci_notouch):
    return "**" if ci_notouch else ""


def check_ci(val_is_outside_ci, ci_notouch):
    if ci_notouch and val_is_outside_ci:
        return "**"
    if ci_notouch and not val_is_outside_ci:
        return "???"
    if val_is_outside_ci and not ci_notouch:
        return "*"
    if not val_is_outside_ci and not ci_notouch:
        return ""


def threshold_results_pretty(
    df, model_order, metric="FPR", precision=TABLE_SCORE_PRECISION
):
    df["model"] = df.apply(lambda row: COL_TO_MODEL[row["col"]], axis=1)
    df = df.query('Group_1 != "ALL" & Group_2 != "ALL"').drop_duplicates(
        subset="comp_id"
    )

    ## Get metric info in nice interval notation.
    df[f"Group_1_{metric}"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=1, metric=metric),
        axis=1,
    )
    df[f"Group_2_{metric}"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=2, metric=metric),
        axis=1,
    )

    df1 = df[
        [
            "comp_id",
            "model",
            "policy",
            "category",
            "attribute",
            "Group_1",
            f"Group_1_{metric}",
            "Group_1_mal",
            f"{metric}_diff",
        ]
    ].rename(
        columns={
            "Group_1": "Group",
            f"Group_1_{metric}": metric,
            "Group_1_mal": "Malignant Scans",
        }
    )
    # df1['CI'] = df1.apply(lambda row: metric_outside_ci(row['CI']), axis=1)
    # df1['CI'] = df1.apply(lambda row: f"{row[f'{metric}_diff']:.{precision}f}", axis=1)
    df1["CI"] = "" * len(df1)
    df1["comp_id"] = df1.apply(
        lambda row: row["comp_id"].split(row["policy"])[1][1:], axis=1
    )

    df2 = df[
        [
            "comp_id",
            "model",
            "policy",
            "category",
            "attribute",
            "Group_2",
            f"Group_2_{metric}",
            "Group_2_mal",
            f"{metric}_outside_CI",
            f"{metric}_CI_notouch",
        ]
    ].rename(
        columns={
            "Group_2": "Group",
            f"Group_2_{metric}": metric,
            "Group_2_mal": "Malignant Scans",
        }
    )
    df2["CI"] = df2.apply(
        lambda row: check_ci(row[f"{metric}_outside_CI"], row[f"{metric}_CI_notouch"]),
        axis=1,
    )
    df2["comp_id"] = df2.apply(
        lambda row: row["comp_id"].split(row["policy"])[1][1:], axis=1
    )
    df_res = pd.concat([df1, df2])

    if df_res["category"].iloc[0] in data.rename_types.keys():
        df_res["Category"] = df_res.apply(
            lambda row: data.rename_types[row["category"]], axis=1
        )
    else:
        df_res["Category"] = df_res["category"]

    if df_res["attribute"].iloc[0] in data.rename_cols.keys() and df_res[
        "attribute"
    ].iloc[0] not in ["Age", "BMI"]:
        df_res["Attribute"] = df_res.apply(
            lambda row: prettify_result_val(row["attribute"], row["Group"])[0],
            axis=1,
        )
        df_res["Group"] = df_res.apply(
            lambda row: prettify_result_val(row["attribute"], row["Group"])[1],
            axis=1,
        )
    else:
        df_res["Attribute"] = df_res["attribute"]

    df_res = df_res.rename(columns={"policy": "Policy"})
    df_res = df_res.set_index(
        pd.MultiIndex.from_frame(df_res[["Policy", "Category", "Attribute", "Group"]])
    )
    model_results = {m: dfm[[metric, "CI"]] for m, dfm in df_res.groupby("model")}

    df_out = pd.concat(model_results, axis=1)
    df_out["Malignant Scans"] = df_res["Malignant Scans"].drop_duplicates()
    df_out = df_out.sort_values(by="Malignant Scans", ascending=False)

    df_out = df_out.sort_index(
        ascending=True, level=["Policy", "Category", "Attribute"], sort_remaining=False
    )[model_order]
    return df_out


def num_with_percent(x, s, precision):
    return f'{0 if np.isnan(x[f"{s}_freq"]) else int(x[f"{s}_freq"])} ({0 if np.isnan(x[f"{s}_norm"]) else np.around(x[f"{s}_norm"], precision)})'


def get_malignancy_rate(row, dfs):
    att_df = dfs[(dfs[row["attribute"]] == row["value"])]
    if len(att_df) == 0:
        return 0
    att_df_mal = len(att_df.query("label == 1"))
    att_df_ben = len(att_df.query("label == 0"))
    # print(subgroup, row['attribute'], row['value'], att_df_mal, att_df_ben)
    return (100 * att_df_mal) / (att_df_mal + att_df_ben)


def confounders_by_attribute(df, attribute, cols, precision=1):
    dfsets = {}
    subgroups = []
    for val, dfv in df.groupby(attribute):
        subgroups.append(val)
        dfsets.update(
            {
                f"{val}-Scans": dfv,
                f"{val}-Mal": dfv.query("label == 1"),
                f"{val}-Ben": dfv.query("label == 0"),
            }
        )

    subgroups_pretty_key = {
        sg: prettify_result_val(attribute, sg)[1] for sg in subgroups
    }

    cat_df = data.combine_diff_dfs(cols["cat"], data.diffs_category_prevalence, dfsets)
    cols_cat = ["Mal", "Ben", "Total %"]
    for s in subgroups:
        cat_df[f"{subgroups_pretty_key[s]} Mal"] = cat_df[f"{s}-Mal_freq"].apply(int)
        cat_df[f"{subgroups_pretty_key[s]} Ben"] = cat_df[f"{s}-Ben_freq"].apply(int)
        cat_df[f"{subgroups_pretty_key[s]} Total %"] = cat_df[f"{s}-Scans_norm"].apply(
            lambda x: np.around(x, precision)
        )

        # for c in cols_to_use:
        #     cat_df[f"{subgroups_pretty_key[s]} {c} (%)"] = cat_df.apply(
        #         lambda x: num_with_percent(x, f"{s}-{c}", precision), axis=1
        #     )

    cat_df = cat_df[pd.notna(cat_df["value"])]
    cat_df = cat_df[cat_df["attribute"] != attribute]

    cat_df["Category"] = cat_df.apply(
        lambda row: data.rename_types[row["category"]], axis=1
    )
    cat_df["Confounder"] = cat_df.apply(
        lambda row: prettify_result_val(row["attribute"], row["value"])[0],
        axis=1,
    )
    cat_df["Subset"] = cat_df.apply(
        lambda row: prettify_result_val(row["attribute"], row["value"])[1],
        axis=1,
    )
    cat_df = cat_df.set_index(
        pd.MultiIndex.from_frame(cat_df[["Category", "Confounder", "Subset"]])
    )

    cat_df = cat_df[
        # [
        #     f"{subgroups_pretty_key[s]} {col} (%)"
        #     for s, col in itertools.product(subgroups, cols_to_use)
        # ]
        [
            f"{subgroups_pretty_key[s]} {col}"
            for s, col in itertools.product(subgroups, cols_cat)
        ]
        + [
            f"diff_{s1}-Scans_{s2}-Scans"
            for s1, s2 in itertools.combinations(subgroups, 2)
        ]
    ]
    cat_df = cat_df.sort_index(ascending=True)

    cols_num = ["Mal", "Ben"]
    num_df = data.combine_diff_dfs(cols["num"], data.diffs_numerical_means, dfsets)
    num_df["Category"] = num_df.apply(
        lambda row: data.rename_types[row["category"]], axis=1
    )
    num_df["Confounder"] = num_df.apply(
        lambda row: prettify_result_val(row["attribute"], row["value"])[0],
        axis=1,
    )
    num_df["Subset"] = num_df.apply(
        lambda row: prettify_result_val(row["attribute"], row["value"])[1],
        axis=1,
    )
    num_df = num_df.set_index(
        pd.MultiIndex.from_frame(num_df[["Category", "Confounder", "Subset"]])
    )

    num_df = num_df[num_df["attribute"] != attribute]
    num_df = num_df[num_df["value"].isin(["Median (IQR)", "Mean (SD)"])]
    num_df = num_df[
        [f"{s}-{col}" for s, col in itertools.product(subgroups, cols_num)]
        + [
            f"diff_{s1}-Scans_{s2}-Scans"
            for s1, s2 in itertools.combinations(subgroups, 2)
        ]
    ]
    num_df = num_df.rename(
        columns={
            f"{s}-{col}": f"{subgroups_pretty_key[s]} {col}"
            for s, col in itertools.product(subgroups, cols_num)
        }
    )
    num_df = num_df.sort_index(ascending=True)

    if len(subgroups) == 2:
        rename_diff = {f"diff_{subgroups[0]}-Scans_{subgroups[1]}-Scans": "Difference"}
        cat_df.rename(columns=rename_diff, inplace=True)
        num_df.rename(columns=rename_diff, inplace=True)
        cat_df["Abs Diff"] = cat_df["Difference"].abs()

        cat_df_cols = [
            f"{subgroups_pretty_key[s]} {col}"
            for s, col in itertools.product(subgroups, cols_cat)
        ]
        cat_df2 = cat_df[cat_df_cols]
        cat_df2.columns = pd.MultiIndex.from_tuples(
            [
                (subgroups_pretty_key[s], col)
                for s, col in itertools.product(subgroups, cols_cat)
            ]
        )
        cat_df2["Difference"] = cat_df["Difference"]
        cat_df2["Abs Diff"] = cat_df["Abs Diff"]
        cat_df = cat_df2

    return cat_df, num_df


def sort_multiindex_by_confounders(df, col="Abs Diff", topn=10):
    attribute_max = df.groupby("Confounder")[col].max()
    sorted_attributes = attribute_max.sort_values(ascending=False).index
    sorted_topn = list(sorted_attributes)[0:topn]

    df2 = df.reset_index()
    df2["Confounder"] = pd.Categorical(
        df2["Confounder"], categories=sorted_attributes, ordered=True
    )
    df2 = df2.sort_values("Confounder")
    df2 = df2.set_index(["Category", "Confounder", "Subset"])
    df2 = df2.query("Confounder in @sorted_topn")
    df2 = df2.drop(columns=["Difference", "Abs Diff"]).droplevel(0)
    return df2, sorted_topn


def roc_isolations_pretty(df0, attribute, model, precision=TABLE_SCORE_PRECISION):
    df = df0.rename(index=RENAME_MODELS)
    df.reset_index(inplace=True, names="model")
    df = df.query(f'model == "{model}"')

    ## Get AUC info in nice interval notation.
    df["Group_1_AUC"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=1), axis=1
    )
    df["Group_2_AUC"] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=2), axis=1
    )

    df["Category"] = df.apply(lambda row: data.rename_types[row["category"]], axis=1)
    df["Confounder"] = df.apply(
        lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[0], axis=1
    )
    df["Subset"] = df.apply(
        lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[1], axis=1
    )
    df["p"] = df.apply(lambda row: data.truncate_p(row["p"]), axis=1)
    df = df.set_index(
        pd.MultiIndex.from_frame(df[["Category", "Confounder", "Subset"]])
    )

    ## Get list of subgroups.
    subgroups = list(
        set(list(pd.unique(df["Group_1"])) + list(pd.unique(df["Group_2"])))
    )
    subgroups_pretty_key = {
        sg: prettify_result_val(attribute, sg)[1] for sg in subgroups
    }

    ## Group 1 and Group 2 don't necessarily always align. We need to realign them.
    # cols_to_realign = ['mal', 'ben', 'pct', 'pct_mal', 'AUC']
    cols_to_realign = {"AUC": "ROC AUC"}

    def realign_group_num(row, subgroup, col="AUC"):
        if row["Group_1"] == subgroup:
            return row[f"Group_1_{col}"]
        if row["Group_2"] == subgroup:
            return row[f"Group_2_{col}"]
        else:
            return None

    for sg in subgroups:
        for c in cols_to_realign:
            df[f"{subgroups_pretty_key[sg]} {cols_to_realign[c]}"] = df.apply(
                lambda row: realign_group_num(row, sg, c), axis=1
            )

    df2 = df[
        [
            f"{subgroups_pretty_key[sg]} {col}"
            for sg, col in itertools.product(subgroups, cols_to_realign.values())
        ]
    ]
    multicol = pd.MultiIndex.from_tuples(
        list(
            itertools.product(
                list(subgroups_pretty_key.values()), cols_to_realign.values()
            )
        )
    )
    df2.columns = multicol
    df2["p"] = df["p"]
    # df = df[['p'] + [f"{subgroups_pretty_key[sg]} AUC" for sg in subgroups]]
    return df2


def prevalence_plus_isolated_roc(
    dataset,
    attribute,
    cols,
    results,
    model,
    topn=10,
    result_prec=TABLE_SCORE_PRECISION,
):
    prevalence, _ = confounders_by_attribute(dataset, attribute, cols, precision=1)
    prevalence, topn_confounders = sort_multiindex_by_confounders(prevalence, topn=topn)

    roc_results = roc_isolations_pretty(
        results, attribute, model, precision=result_prec
    ).droplevel(0)

    combined_df = pd.concat([prevalence, roc_results], axis=1).dropna(axis=0)

    subgroups = list(
        set(list(pd.unique(results["Group_1"])) + list(pd.unique(results["Group_2"])))
    )
    subgroups_pretty = [prettify_result_val(attribute, sg)[1] for sg in subgroups]
    subcols_to_align = ["Mal", "Ben", "Total %", "ROC AUC"]

    combined_df = combined_df[
        list(itertools.product(subgroups_pretty, subcols_to_align)) + [("p", "")]
    ]

    for s, c in itertools.product(subgroups_pretty, ["Mal", "Ben"]):
        combined_df[(s, c)] = combined_df[(s, c)].astype(int)

    return combined_df, topn_confounders


def threshold_isolations_pretty(
    df, model, metric="TPR", precision=TABLE_SCORE_PRECISION
):
    df["model"] = df.apply(lambda row: COL_TO_MODEL[row["col"]], axis=1)
    df = df.query(f'model == "{model}" & Group_1 != "ALL" & Group_2 != "ALL"')
    df = df.sort_values(by=["comp_id", "Group_1"], ascending=False).drop_duplicates(
        subset="comp_id"
    )

    if df["Group_1"].nunique() > 1 or df["Group_2"].nunique() > 1:
        print("ERROR: probably did not do top2=True in the pairwise comparisons.")
        return None

    g1, g2 = pd.unique(df["Group_1"])[0], pd.unique(df["Group_2"])[0]

    ## Get metric info in nice interval notation.
    df[g1] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=1, metric=metric),
        axis=1,
    )
    df[g2] = df.apply(
        lambda row: pretty_interval(row, precision, group_num=2, metric=metric),
        axis=1,
    )
    df["CI"] = df.apply(
        lambda row: check_ci(row[f"{metric}_outside_CI"], row[f"{metric}_CI_notouch"]),
        axis=1,
    )

    if df["category"].iloc[0] in data.rename_types.keys():
        df["Category"] = df.apply(
            lambda row: data.rename_types[row["category"]], axis=1
        )
    else:
        df["Category"] = df["category"]

    if df["filter_by"].iloc[0] in data.rename_cols.keys():
        df["Confounder"] = df.apply(
            lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[0],
            axis=1,
        )
        df["Subset"] = df.apply(
            lambda row: prettify_result_val(row["filter_by"], row["filter_val"])[1],
            axis=1,
        )
    else:
        df["Confounder"] = df["filter_by"]
        df["Subset"] = df["filter_val"]

    df = df.set_index(
        pd.MultiIndex.from_frame(df[["policy", "Category", "Confounder", "Subset"]])
    )
    df = df.sort_index(
        ascending=True,
        level=["policy", "Category", "Confounder", "Subset"],
        sort_remaining=False,
    )

    df = df[[g1, g2, "CI"]]
    return df


def threshold_isolation_pairwise(
    df,
    demo,
    model,
    policies,
    metric_tuples=[("TPR", "90% Specificity"), ("TNR", "90% Sensitivity")],
    topn_confs=None,
    precision=TABLE_SCORE_PRECISION,
    pairwise_comps=None,
):
    df_pretty, _ = threshold_stats_pretty(df, policies, demographic_for_isolations=demo)
    metrics = [m for m, _ in metric_tuples]

    if pairwise_comps is None:
        pairwise_comps = {
            m: threshold.all_isolation_pairwise_comparisons(df_pretty, metric=m)
            for m in metrics
        }

    perf_tables = {}
    for m, p in metric_tuples:
        if m not in pairwise_comps:
            pairwise_comps[m] = threshold.all_isolation_pairwise_comparisons(
                df_pretty, metric=m
            )

        res = threshold_isolations_pretty(
            pairwise_comps[m], model=model, metric=m, precision=precision
        )
        res = res.xs(p, level="policy").droplevel(0)
        if topn_confs:
            res = res.loc[topn_confs, :]

        perf_tables[f"{m} ({p})"] = res

    df_out = pd.concat(perf_tables, axis=1)
    return df_out, pairwise_comps
