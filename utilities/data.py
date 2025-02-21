import pandas as pd
import os
import numpy as np
import copy

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt
import scipy.stats
from IPython.display import display, Markdown

import sklearn.metrics as skl_metrics
import warnings

MODEL_TO_COL = {
    "Venkadesh": "DL_cal",
    "de Haas Combined": "Thijmen_mean_cal",
    "de Haas Local": "Thijmen_local_cal",
    "de Haas Global (hidden nodule)": "Thijmen_global_hidden_cal",
    "de Haas Global (shown nodule)": "Thijmen_global_show_cal",
    "Sybil year 1": "sybil_year1",
    # "Sybil year 2": "sybil_year2",
    # "Sybil year 3": "sybil_year3",
    # "Sybil year 4": "sybil_year4",
    # "Sybil year 5": "sybil_year5",
    # "Sybil year 6": "sybil_year6",
    "PanCan2b": "PanCan2b",
}


## Get prevalence info for a category in the dataset.
def catinfo(df, cat, include_all=False):
    groups = df.groupby(cat)
    info_dict = {"num": [], "pct": [], "mal": [], "ben": [], "pct_mal": []}
    subgroup_names = []

    for subg, dfg in groups:
        subgroup_names.append(subg)
        info_dict["num"].append(len(dfg))
        info_dict["pct"].append(100 * len(dfg) / len(df))
        info_dict["mal"].append(len(dfg.query("label == 1")))
        info_dict["ben"].append(len(dfg.query("label == 0")))
        info_dict["pct_mal"].append(100 * len(dfg.query("label == 1")) / len(dfg))

    df_catinfo = pd.DataFrame(info_dict, index=subgroup_names)
    return df_catinfo


def prep_nlst_preds(df, democols=None, scanlevel=True, sybil=True, tijmen=False):
    if scanlevel:
        nodule_drop_cols = [
            "CoordX",
            "CoordY",
            "CoordZ",
            "NoduleType",
            "NoduleID",
            "AnnotationID",
            "Mean_Entropy_Kiran",
        ]
        nodule_agg_cols = [
            "Spiculation",
            "Diameter_mm",
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

    ## TODO: add logic for filtering model columns and demo columns
    if democols:
        democols = copy.deepcopy(democols)
        if scanlevel:
            democols["num"].pop("nodule")

    models = copy.deepcopy(MODEL_TO_COL)

    if not sybil:
        for i in range(6):
            if f"Sybil year {i+1}" in models.keys():
                models.pop(f"Sybil year {i+1}")

    if not tijmen:
        models.pop("de Haas Combined")

    if tijmen:
        df = df[(~df["Thijmen_mean"].isna())]
    if sybil:
        df = df[(~df["sybil_year1"].isna())]

    df, democols = bin_numerical_columns(df, democols)
    return df, democols, models


### CREATE BINARY bins of numerical columns (for our analysis).
def bin_numerical_columns(df, democols):
    if democols is None:
        return df, democols

    ### Cutoff values - incldue in the left interval.
    cutoff_values = {
        "height": 68,
        "weight": 180,
        "smokeage": 16,
        "smokeday": 25,
        "smokeyr": 40,
        "pkyr": 55,
        "NoduleCounts": 1,  ### NLST
        "NoduleCountPerScan": 1,  ### DLCST
        "Diameter_mm": 7,
        "Age": 61,
    }

    numerical_cols = democols["num"]
    for category in numerical_cols:
        for attribute in numerical_cols[category]:
            if attribute not in cutoff_values.keys():
                continue

            query_string = f"{attribute} > {cutoff_values[attribute]}"
            df[query_string] = df.eval(query_string)
            democols["cat"][category].append(query_string)

        democols["cat"][category] = sorted(list(set(democols["cat"][category])))

    return df, democols


def bmi_calc(height, weight):
    return (weight * 703) / (height * height)


def corrmat(df, rows, cols, method="kendall", vmin=-1, vmax=1, cmap="RdYlGn"):
    cols_list = list(set(rows).union(set(cols)))
    corrmat = df[cols_list].corr(method=method)

    plt.figure(figsize=(len(cols) * 0.6, len(rows) * 0.5))
    sns.heatmap(corrmat.loc[rows, cols], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.show()

    return corrmat


def cat_dist_df(c="Gender", dfsets={}):
    dfdict = {}
    for m in dfsets:
        dfdict[f"{m}_freq"] = (
            dfsets[m][c].value_counts(normalize=False, dropna=False).astype(int)
        )
        dfdict[f"{m}_norm"] = 100 * dfsets[m][c].value_counts(
            normalize=True, dropna=False
        ).round(6)

    for i, m1 in enumerate(dfsets):
        for j, m2 in enumerate(dfsets):
            if j > i:
                dfdict[f"diff_{m1}_{m2}"] = (
                    dfdict[f"{m1}_norm"] - dfdict[f"{m2}_norm"]
                ).round(4)

    df = pd.DataFrame(dfdict).drop_duplicates()
    return df


def num_dist_df(c="Gender", dfsets={}):
    dfdict = {}
    for m in dfsets:
        dfdict[f"{m}"] = dfsets[m][c].describe(percentiles=[0.5]).round(4)

    for i, m1 in enumerate(dfsets):
        for j, m2 in enumerate(dfsets):
            if j > i:
                dfdict[f"diff_{m1}_{m2}"] = dfdict[f"{m1}"] - dfdict[f"{m2}"]

    df = pd.DataFrame(dfdict).drop_duplicates()
    df.drop(index=["count", "max", "min", "std"], inplace=True)
    return df


def combine_col_dfs(cols={}, df_func=cat_dist_df, dfsets={}, dispdf=False):
    splitdfs = []
    for cat in cols:
        if dispdf:
            display(Markdown(f"### {cat}"))

        for c in cols[cat]:
            df = df_func(c, dfsets)
            if dispdf:
                display(df)

            df["category"] = [cat] * len(df)
            df["attribute"] = [c] * len(df)
            df["value"] = df.index.values

            dfcols = df.columns.tolist()
            dfcols = dfcols[-3:] + dfcols[:-3]
            df = df[dfcols]
            df.reset_index(inplace=True, drop=True)
            df.sort_values(by="value", ascending=True, inplace=True)

            splitdfs.append(df)

    return pd.concat(splitdfs, axis=0, ignore_index=True)
