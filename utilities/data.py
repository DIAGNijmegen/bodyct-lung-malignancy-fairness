import pandas as pd
import os
import numpy as np
import copy

import seaborn as sns
from evalutils.roc import get_bootstrapped_roc_ci_curves
import matplotlib.pyplot as plt
import scipy.stats
from IPython.display import display, Markdown
import statsmodels.api as sm
import statsmodels.stats.proportion as smp
import statsmodels.stats.weightstats as smw


import sklearn.metrics as skl_metrics
import warnings

MODEL_TO_COL = {
    "Venkadesh": "DL_cal",
    "Sybil year 1": "sybil_year1",
    "PanCan2b": "PanCan2b",
}

keys = {
    "Gender": {1: "Male", 2: "Female"},
    "Sex": {1: "Male", 2: "Female"},
    "Overweight": {True: "≥ 25", False: "< 25"},
    "race": {
        1: "White",
        2: "Black",
        3: "Asian",
        4: "Native American",
        5: "Native Hawaiian",
        6: "More than one race",
        # 7: 'Unknown',
        # 95: 'Unknown',
        # 96: 'Unknown',
        # 98: 'Unknown',
        # 99: 'Unknown'
    },
    "WhiteOrBlack": {
        1: "White",
        2: "Black",
        # 7: 'Unknown',
        # 95: 'Unknown',
        # 96: 'Unknown',
        # 98: 'Unknown',
        # 99: 'Unknown'
    },
    "ethnic": {1: "Hispanic/Latino", 2: "Not Hispanic/Latino"},
    "marital": {
        1: "Never Married",
        2: "Married",
        3: "Widowed",
        4: "Separated",
        5: "Divorced",
    },
    "educat": {
        1: "8th grade or less",
        2: "9th-11th grade",
        3: "HS Graduate / GED",
        4: "Post-HS training",
        5: "Associate Degree",
        6: "Bachelors Degree",
        7: "Graduate School",
    },
    "LC_stage": {
        110: "IA",
        120: "IB",
        210: "IIA",
        220: "IIB",
        310: "IIIA",
        320: "IIIB",
        400: "IV",
        888: "TNM not available",
        900: "Occult Carcinoma",
        994: "Carcinoid, cannot be assessed",
        999: "Unknown, cannot be assessed",
    },
}

binary_key = {
    1: True,
    0: False,
    True: True,
    False: False,
    1.0: True,
    0.0: False,
}
### True or False columns.
boolean_cols = [
    "Overweight",
    "Married",
    "HighSchoolPlus",
    "HS-or-more",
    "more-than-HS",
    "NonHispanicWhite",
    "Unfinished_ed",
    "smokelive",
    "cigar",
    "cigsmok",
    "smokework",
    "pipe",
    "wrkbaki",
    "wrkfoun",
    "wrkchem",
    "wrkasbe",
    "wrkfire",
    "wrksand",
    "wrkfarm",
    "wrkcoal",
    "wrkpain",
    "wrkweld",
    "wrkflou",
    "wrkbutc",
    "wrkhard",
    "wrkcott",
    "diagasbe",
    "diagchas",
    "diagpneu",
    "diagstro",
    "diagemph",
    "diagbron",
    "diagsili",
    "diagsarc",
    "diaghear",
    "diagdiab",
    "diagadas",
    "diagcopd",
    "diagfibr",
    "diagtube",
    "diaghype",
    "diagchro",
    "canckidn",
    "cancphar",
    "canccolo",
    "cancoral",
    "cancpanc",
    "canccerv",
    "cancstom",
    "cancthyr",
    "canctran",
    "cancnasa",
    "canclary",
    "cancbrea",
    "cancesop",
    "cancblad",
    "canclung",
    "GroundGlassOpacity",
    "NoduleInUpperLung",
    "Perifissural",
    "NonSolid",
    "Calcified",
    "Spiculation",
    "PartSolid",
    "Solid",
    "SemiSolid",
    "FamilyHistoryLungCa",
    "PersonalCancerHist",
    "wrknomask",
    "Emphysema",
    "Adenosquamous_carcinoma",
    "Small_cell_carcinoma",
    "Bronchiolo-alveolar_carcinoma",
    "Carcinoid_tumor",
    "Adenocarcinoma",
    "Squamous_cell_carcinoma",
    "Unclassified_carcinoma",
    "Large_cell_carcinoma",
]

rename_types = {
    "demo": "Demographics",
    "smoke": "Smoking",
    "nodule": "Nodule",
    "other": "Other",
    "work": "Work History",
    "disease": "Disease Diagnosis",
    "canchist": "Previous Cancer Diagnosis",
    "lungcanc": "Lung Cancer",
    "scanner": "Scanner",
}

rename_cols = {
    ### NLST
    "BMI": "Body Mass Index",
    "Age": "Age",
    "height": "Height",
    "weight": "Weight",
    "smokeage": "Age at Smoking Onset",
    "smokeday": "Cigarettes per Day",
    "smokeyr": "Total Years of Smoking",
    "pkyr": "Pack-Years",
    "CoordX": "Nodule X",
    "CoordY": "Nodule Y",
    "CoordZ": "Nodule Z",
    "Mean_Entropy_Kiran": "Mean Entropy Score (Venkadesh)",
    "NoduleCounts": "Nodules Per Scan",
    "Diameter_mm": "Diameter (mm)",
    "SliceCount": "Slices In Scan",
    "Overweight": "BMI",
    "educat": "Education Status",
    "HS-or-more": "Graduated HS",
    "more-than-HS": "Post-HS Education",
    "Gender": "Sex",
    "Married": "Married",
    "HighSchoolPlus": "HS Education",
    "NonHispanicWhite": "Non-Hispanic White",
    "Unfinished_ed": "Unfinished Education",
    "WhiteOrBlack": "White or Black",
    "marital": "Marital Status",
    "ethnic": "Ethnicity",
    "race": "Race",
    "smokelive": "Lived w/ Smoker",
    "cigar": "Smoked Cigars",
    "cigsmok": "Current Smoker",
    "smokework": "Worked w/ Smoker",
    "pipe": "Smoked Pipe",
    "wrkbaki": "Work - Baking",
    "wrkfoun": "Work - Foundry or Steel Milling",
    "wrkchem": "Work - Chemicals or Plastics Mfg.",
    "wrkasbe": "Work - Asbestos",
    "wrkfire": "Work - Firefighting",
    "wrksand": "Work - Sandblasting",
    "wrkfarm": "Work - Farming",
    "wrkcoal": "Work - Coal Mining",
    "wrkpain": "Work - Painting",
    "wrkweld": "Work - Welding",
    "wrkflou": "Work - Flour/Feed or Grain Milling",
    "wrkbutc": "Work - Butchering or Meat Packing",
    "wrkhard": "Work - Hard Rock Mining",
    "wrkcott": "Work - Cotton or Jute Processing",
    "diagasbe": "Asbestosis Diag.",
    "diagchas": "Childhood Asthma Diag.",
    "diagpneu": "Pneumonia Diag.",
    "diagstro": "Stroke",
    "diagemph": "Emphysema Diag.",
    "diagbron": "Bronchiectasis Diag.",
    "diagsili": "Silicosis Diag.",
    "diagsarc": "Sarcoidosis Diag.",
    "diaghear": "Heart Disease or Attack",
    "diagdiab": "Diabetes Diag.",
    "diagadas": "Adult Asthma Diag.",
    "diagcopd": "COPD Diag.",
    "diagfibr": "Lung Fibrosis Diag.",
    "diagtube": "Tuberculosis Diag.",
    "diaghype": "Hypertension Diag.",
    "diagchro": "Chronic Bronchitis Diag.",
    "canckidn": "Prev. Cancer - Kidney",
    "cancphar": "Prev. Cancer - Pharynx",
    "canccolo": "Prev. Cancer - Colorectal",
    "cancoral": "Prev. Cancer - Oral",
    "cancpanc": "Prev. Cancer - Prancreatic",
    "canccerv": "Prev. Cancer - Cervical",
    "cancstom": "Prev. Cancer - Stomach",
    "cancthyr": "Prev. Cancer - Thyroid",
    "canctran": "Prev. Cancer - Transitional Cell",
    "cancnasa": "Prev. Cancer - Nasal",
    "canclary": "Prev. Cancer - Larynx",
    "cancbrea": "Prev. Cancer - Breast",
    "cancesop": "Prev. Cancer - Esophageal",
    "cancblad": "Prev. Cancer - Bladder",
    "canclung": "Prev. Cancer - Lung",
    "GroundGlassOpacity": "Ground-Glass Nodule",
    "NoduleInUpperLung": "Nodule in Upper Lung",
    "Perifissural": "Perfissural Nodule",
    "NonSolid": "Non-Solid Nodule",
    "Calcified": "Calcified Nodule",
    "Spiculation": "Spiculated Nodule",
    "PartSolid": "Part-Solid Nodule",
    "Solid": "Solid Nodule",
    "SemiSolid": "Semi-Solid Nodule",
    "FamilyHistoryLungCa": "Family History of LC",
    "PersonalCancerHist": "Previous Cancer Diagnosis",
    "wrknomask": "Work w/o Mask",
    "Emphysema": "Emphysema in Scan",
    "LC_stage": "LC Stage",
    "Adenosquamous_carcinoma": "Adenosquamous Carcinoma",
    "Small_cell_carcinoma": "Small Cell Carcinoma",
    "Bronchiolo-alveolar_carcinoma": "Bronchiolo-Alveolar Carcinoma",
    "Carcinoid_tumor": "Carcinoid Tumor",
    "Adenocarcinoma": "Adenocarcinoma",
    "Squamous_cell_carcinoma": "Squamous Cell Carcinoma",
    "Unclassified_carcinoma": "Unclassified Carcinoma",
    "Large_cell_carcinoma": "Large Cell Carcinoma",
    "Manufacturer": "Manufacturer",
    "ManufacturersModelName": "Model Name",
}

pretty_boolean_cols = [rename_cols[c] for c in boolean_cols]


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


def democols_to_list(democols):
    cols_num = sum(democols["num"].values(), start=[])
    cols_cat = sum(democols["cat"].values(), start=[])
    cols_list = cols_num + cols_cat
    return cols_list


def prep_nlst_preds(
    df,
    democols=None,
    scanlevel=True,
    sybil=True,
    bin_num=True,
    pretty=False,
):
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
            "DL_cal",
        ]
        # Not including Sybil here because it's already scan-level of course.

        df = df.drop(nodule_drop_cols, axis=1, errors="ignore")
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

    if sybil:
        df = df[(~df["sybil_year1"].isna())]

    if pretty:
        df, democols = nlst_pretty_labels(df, democols)
    if bin_num:
        df, democols = bin_numerical_columns(df, democols, pretty=pretty)

    return df, democols, models


### CREATE BINARY bins of numerical columns (for our analysis).
def bin_numerical_columns(df, democols, pretty=False):
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
        "NoduleCounts": 1,
        "Diameter_mm": 6,  ### previously 7. based on some screening guidelines (BTS, ILST, Lung-RADS)
        "Age": 61,  ### 62 or over. based on data median.
        "SliceCount": 200,  ### Truncated by Sybil preprocessing
    }

    if pretty:
        cutoff_values = {rename_cols[k]: v for k, v in cutoff_values.items()}

    numerical_cols = democols["num"]
    for category in numerical_cols:
        for attribute in numerical_cols[category]:
            if attribute not in cutoff_values.keys():
                continue

            query_string = f"{attribute} > {cutoff_values[attribute]}"
            query_string_backticks = f"`{attribute}` > {cutoff_values[attribute]}"
            df.loc[:, query_string] = df.eval(query_string_backticks)
            democols["cat"][category].append(query_string)

        democols["cat"][category] = sorted(list(set(democols["cat"][category])))

    return df, democols


def corrmat(df, rows, cols, method="kendall", vmin=-1, vmax=1, cmap="RdYlGn"):
    cols_list = list(set(rows).union(set(cols)))
    corrmat = df[cols_list].corr(method=method)

    plt.figure(figsize=(len(cols) * 0.6, len(rows) * 0.5))
    sns.heatmap(corrmat.loc[rows, cols], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.show()

    return corrmat


def nlst_pretty_labels(df, nlst_democols):
    pretty_cols = {}

    for typ in nlst_democols:
        if typ == "info":
            continue
        pretty_cols[typ] = {}
        for cat in nlst_democols[typ]:
            pretty_cols[typ][rename_types[cat]] = []
            for att in nlst_democols[typ][cat]:
                if att not in df.columns:
                    continue
                pretty_cols[typ][rename_types[cat]].append(rename_cols[att])

                if att in keys:
                    df[att] = df[att].replace(keys[att])
                elif att in boolean_cols or df[att].dtype == bool:
                    df[att] = df[att].replace(binary_key)

    df2 = df.rename(columns=rename_cols)
    return df2, pretty_cols


## Truncate P values, according to rules (JMIR): https://support.jmir.org/hc/en-us/articles/360000002012-How-should-P-values-be-reported
def truncate_p(p):
    if np.isnan(p):
        return None

    ## For P < 0.001
    if p < 0.001:
        return f"< .001"

    elif p < 0.01:
        p_out = np.floor(p * 10**3) / 10**3
        return f"{p_out:.3f}".lstrip("0")

    p_out = np.floor(p * 10**2) / 10**2
    return f"{p_out:.2f}".lstrip("0")


## Includes score test for proportions of disease prevalence.
## Score test recommended by Tang et al. (2012): https://doi.org/10.1002/bimj.201100216
def diffs_category_prevalence(c="Gender", dfsets={}, include_stat=False):
    dfdict = {}
    for m in dfsets:
        dfdict[f"{m}_freq"] = dfsets[m][c].value_counts(normalize=False, dropna=False)
        dfdict[f"{m}_norm"] = 100 * dfsets[m][c].value_counts(
            normalize=True, dropna=False
        ).round(6)

    df = pd.DataFrame(dfdict).drop_duplicates().fillna(0)

    for i, m1 in enumerate(dfsets):
        for j, m2 in enumerate(dfsets):
            if j > i:
                df[f"diff_{m1}_{m2}"] = df[f"{m1}_norm"] - df[f"{m2}_norm"]

    if include_stat:
        for i, m1 in enumerate(dfsets):
            for j, m2 in enumerate(dfsets):
                if j <= i:
                    continue
                n1, n2 = len(dfsets[m1]), len(dfsets[m2])
                stats, pvals = [], []
                for val, row in df.iterrows():
                    f1, f2 = row[f"{m1}_freq"], row[f"{m2}_freq"]
                    s, p = smp.test_proportions_2indep(
                        f1, n1, f2, n2, return_results=False, method="wald"
                    )
                    stats.append(s)
                    pvals.append(p)

                df[f"stat_{m1}_{m2}"] = stats
                df[f"p_{m1}_{m2}"] = pvals

    return df


def diffs_numerical_means(c="Gender", dfsets={}, include_stat=False):
    dfdict = {}
    for m in dfsets:
        dfdict[f"{m}"] = dfsets[m][c].describe(percentiles=[0.25, 0.5, 0.75]).round(4)

    for i, m1 in enumerate(dfsets):
        for j, m2 in enumerate(dfsets):
            if j > i:
                dfdict[f"diff_{m1}_{m2}"] = dfdict[f"{m1}"] - dfdict[f"{m2}"]

    for m in dfsets:
        dfdict[f"{m}"][
            "Mean (SD)"
        ] = f'{dfdict[f"{m}"]["mean"]:.1f} ({dfdict[f"{m}"]["std"]:.1f})'
        dfdict[f"{m}"][
            "Median (IQR)"
        ] = f'{int(dfdict[f"{m}"]["50%"])} ({int(dfdict[f"{m}"]["75%"] - dfdict[f"{m}"]["25%"])})'

    for i, m1 in enumerate(dfsets):
        for j, m2 in enumerate(dfsets):
            if j > i:
                dfdict[f"diff_{m1}_{m2}"]["Mean (SD)"] = (
                    dfdict[f"{m1}"]["mean"] - dfdict[f"{m2}"]["mean"]
                )
                dfdict[f"diff_{m1}_{m2}"]["Median (IQR)"] = (
                    dfdict[f"{m1}"]["50%"] - dfdict[f"{m2}"]["50%"]
                )

    df = pd.DataFrame(dfdict).drop_duplicates()
    df.drop(index=["count", "max", "min"], inplace=True)

    if include_stat:
        for i, m1 in enumerate(dfsets):
            for j, m2 in enumerate(dfsets):
                if j <= i:
                    continue
                stats, pvals = [], []
                for val, row in df.iterrows():
                    x1, x2 = dfsets[m1][c].dropna(), dfsets[m2][c].dropna()
                    t, p, dof = smw.ttest_ind(x1, x2)
                    stats.append(t)
                    pvals.append(p)

                df[f"stat_{m1}_{m2}"] = stats
                df[f"p_{m1}_{m2}"] = pvals

    return df


def combine_diff_dfs(
    cols={},
    df_func=diffs_category_prevalence,
    dfsets={},
    dispdf=False,
    include_stat=False,  ## Not applicable for comparing datasets since they have overlap.
):
    splitdfs = []
    for cat in cols:
        if dispdf:
            display(Markdown(f"### {cat}"))

        for c in cols[cat]:
            df = df_func(c, dfsets, include_stat)
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


def demographics_across_datasets(dfsets, cols={}, dispdf=False, include_stat=False):
    cat_df = combine_diff_dfs(
        cols, dfsets, diffs_category_prevalence, dispdf, include_stat
    )
    num_df = combine_diff_dfs(cols, dfsets, diffs_numerical_means, dispdf, include_stat)
    return cat_df, num_df
