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

## directory where results are
CHANSEY_ROOT = "/data/bodyct"  ## or "W:"
EXPERIMENT_DIR = f"{CHANSEY_ROOT}/experiments/lung-malignancy-fairness-shaurya"
TEAMS_DIR = "C:/Users/shaur/OneDrive - Radboudumc/Documents - Master - Shaurya Gaur/General/Data"
BASE_DIR = TEAMS_DIR

FILE_DIR = f"{BASE_DIR}/melba-files-2"
INPUT_DIR = f"{BASE_DIR}/inputs"
RESULTS_DIR = f"{BASE_DIR}/melba-results-2"
FIG_DIR = f"{BASE_DIR}/melba-figs-2"
TAB_DIR = f"{BASE_DIR}/melba-tables-2"

## Plot settings
sns.set_style("white")
sns.set_theme(
    "talk",
    "whitegrid",
    "dark",
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)
color_palette = sns.color_palette("colorblind", 30)

MODEL_TO_COL = {
    "Venkadesh": "DL_cal",
    "Sybil year 1": "sybil_year1",
    "Sybil year 2": "sybil_year2",
    "Sybil year 3": "sybil_year3",
    "Sybil year 4": "sybil_year4",
    "Sybil year 5": "sybil_year5",
    "Sybil year 6": "sybil_year6",
    "PanCan2b": "PanCan2b",
}

ILST_THRESHOLD = 0.06

THRESHOLD_POLICIES = (("Sensitivity", 0.9), ("Specificity", 0.9))

NLST_POSSIBLE_CONFOUNDERS = {
    "num": {
        "demo": ["BMI", "Age", "height", "weight"],
        "smoke": ["smokeage", "smokeday", "smokeyr", "pkyr"],
        "other": [
            "NoduleCounts",
            "Diameter_mm",
            # "SliceCount"
        ],
    },
    "cat": {
        "demo": [
            "Age > 61",
            "Gender",
            # "HighSchoolPlus",
            # "Married",
            "Overweight",
            "ethnic",
            "height > 68",
            "weight > 180",
        ],
        "smoke": [
            "cigar",
            "cigsmok",
            "pipe",
            "pkyr > 55",
            "smokeage > 16",
            "smokeday > 25",
            "smokelive",
            "smokework",
            "smokeyr > 40",
        ],
        "work": [
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
        ],
        "disease": [
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
        ],
        "canchist": [
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
        ],
        "nodule": [
            "GroundGlassOpacity",
            "NoduleInUpperLung",
            "Perifissural",
            "NonSolid",
            "Calcified",
            "Spiculation",
            "PartSolid",
            "Solid",
            "SemiSolid",
        ],
        "other": [
            "Diameter_mm > 6",
            "Emphysema",
            "FamilyHistoryLungCa",
            "NoduleCounts > 1",
            "PersonalCancerHist",
            "wrknomask",
        ],
        "lungcanc": [
            "LC_stage",
            "Adenosquamous_carcinoma",
            "Small_cell_carcinoma",
            "Bronchiolo-alveolar_carcinoma",
            "Carcinoid_tumor",
            "Adenocarcinoma",
            "Squamous_cell_carcinoma",
            "Unclassified_carcinoma",
            "Large_cell_carcinoma",
        ],
    },
}
