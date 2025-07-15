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
