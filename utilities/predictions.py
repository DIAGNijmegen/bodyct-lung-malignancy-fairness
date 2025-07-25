import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


def make_calibration_plots(labels, raw_outputs, calibrated_outputs, title=""):
    """
    Make the calibration curves or the reliability diagrams to show how well calibrated a classifier is

    Parameters
    ----------
    labels: array of binary labels
    raw_outputs: array of confidences between 0-1 (uncalibrated)
    calibrated_outputs: array of confidences between 0-1 (calibrated)
    title: additional title describing the dataset for which the calibration curve is being plotted

    Returns
    -------
    figure with plots

    """

    figure = plt.figure(figsize=(5, 5), constrained_layout=True)

    colors = sns.color_palette("colorblind")

    # Platt's scaling

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, calibrated_outputs, n_bins=10
    )
    plt.scatter(mean_predicted_value, fraction_of_positives)
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="DL algorithm after Platt's scaling",
        color=colors[4],
    )

    # uncalibrated

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, raw_outputs, n_bins=10
    )
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="DL algorithm",
        color=colors[0],
    )

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    plt.title(
        f"Calibration plots (reliability curves)" + title,
        # fontsize=8,
    )
    plt.grid()
    plt.ylabel("Fraction of positives")
    plt.xlabel("Predicted probability")
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.legend(loc="lower right")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0.1, 1.1, 0.1))

    return figure


def calibrate_preds(
    preds,
    labels,
    estimator_model=LogisticRegression,
    method="sigmoid",
):
    logits = logit(np.array(preds))
    labels = np.array(labels)
    clf = CalibratedClassifierCV(estimator=estimator_model(), cv=10, method=method)
    clf.fit(logits.reshape(-1, 1), labels)
    calibrated_outputs = clf.predict_proba(logits.reshape(-1, 1))[:, 1]
    return calibrated_outputs


def check_scoredists(df, columns, label="label", fh=4):
    fig, ax = plt.subplots(1, len(columns), figsize=(fh * len(columns), fh))
    ax = ax.flatten()

    for i, m in enumerate(columns):
        sns.histplot(
            data=df,
            ax=ax[i],
            x=m,
            hue=label,
            common_norm=False,
            stat="probability",
            bins=10,
        )
        ax[i].set_title(f"{m}")

    plt.tight_layout()
    return fig


def sigmoid(x):
    """Compute sigmoid values for each set of scores in x."""
    s = 1 / (1 + np.exp(-x))
    return s


def load_tijmen_results(
    base_path,
    model_name,
    label_name="label",
    local=True,
    valid=True,
    train=False,
    num_folds=10,
):

    if valid:
        base_path = rf"{base_path}/valid_fold"
    elif train:
        base_path = rf"{base_path}/train_fold"
    else:
        base_path = rf"{base_path}/fold"

    if local:
        column_name = "y_local"
    else:
        column_name = "y_pred"

    df = None
    # Loop over the folds
    for fold in range(num_folds):
        # Generate the path for each fold's CSV file
        file_path = rf"{base_path}{fold}/results.csv"

        # Read the CSV file for the current fold
        fold_df = pd.read_csv(file_path)

        fold_df[column_name] = sigmoid(fold_df[column_name])

        if df is None:
            df = fold_df.copy()
        else:
            # Merge on 'img_names' and add 'y_Combined' values
            df = pd.merge(
                df,
                fold_df[["img_names", column_name, "y"]],
                on="img_names",
                how="outer",
                suffixes=("", f"_fold{fold}"),
            )

    # Ensure the column from the first fold is kept
    df[f"{column_name}_fold0"] = df[column_name]
    df[f"y_fold0"] = df["y"]
    df.drop(columns=[column_name], inplace=True)

    # Calculate entropy across folds
    df["entropy"] = df[[f"{column_name}_fold{i}" for i in range(10)]].apply(
        lambda row: np.mean(
            [-(p * np.log2(p)) - (1 - p) * np.log2(1 - p) for p in row]
        ),
        axis=1,
    )

    # Calculate mean prediction across all model results for each image
    df[model_name] = df[[f"{column_name}_fold{i}" for i in range(10)]].mean(axis=1)
    df[label_name] = df[[f"y_fold{i}" for i in range(10)]].mean(axis=1)
    df["AnnotationID"] = df["img_names"]
    return df


## Function adapted from Sybil GitHub repository:
#           https://github.com/reginabarzilaygroup/Sybil/blob/1e358f8069cb4d0986071ed50d9836200a6ed625/sybil/datasets/nlst.py#L322
##      as mentioned in its GitHub issue #72 by Peter Mikhael (Sybil author).
def sybil_label(row, max_followup=6):
    screen_timepoint = row["timepoint"]
    days_since_rand = row["scr_days{}".format(screen_timepoint)]
    days_to_cancer_since_rand = row["candx_days"]
    days_to_cancer = days_to_cancer_since_rand - days_since_rand
    years_to_cancer = (
        int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
    )
    days_to_last_followup = int(row["fup_days"] - days_since_rand)
    years_to_last_followup = days_to_last_followup // 365
    y = years_to_cancer < max_followup
    y_seq = np.zeros(max_followup)
    cancer_timepoint = row["cancyr"]
    if y:
        if years_to_cancer > -1:
            assert screen_timepoint <= cancer_timepoint
        time_at_event = years_to_cancer
        y_seq[years_to_cancer:] = 1
    else:
        time_at_event = min(years_to_last_followup, max_followup - 1)
    y_mask = np.array(
        [1] * (time_at_event + 1) + [0] * (max_followup - (time_at_event + 1))
    )
    assert len(y_mask) == max_followup
    return y
