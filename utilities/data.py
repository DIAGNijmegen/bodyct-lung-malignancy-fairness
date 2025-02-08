import pandas as pd
import os
import numpy as np

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt
import scipy.stats

import sklearn.metrics as skl_metrics
import warnings

MODEL_TO_COL = {
    "Venkadesh": "DL_cal",
    "de Haas Combined": "Thijmen_mean_cal",
    "de Haas Local": "Thijmen_local_cal",
    "de Haas Global (hidden nodule)": "Thijmen_global_hidden_cal",
    "de Haas Global (shown nodule)": "Thijmen_global_show_cal",
    "Sybil year 1": "sybil_year1",
    "Sybil year 2": "sybil_year2",
    "Sybil year 3": "sybil_year3",
    "Sybil year 4": "sybil_year4",
    "Sybil year 5": "sybil_year5",
    "Sybil year 6": "sybil_year6",
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


def prep_nlst_preds(df, scanlevel=True, sybil=True, tijmen=True):
    if scanlevel:
        nodule_drop_cols = [
            "CoordX",
            "CoordY",
            "CoordZ",
            "NoduleID",
            "AnnotationID",
            "Mean_Entropy_Kiran",
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
