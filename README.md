# Fairness of Lung Malignancy Risk Estimation Models
*Shaurya Gaur | MSc Thesis | Sep 2024 - Apr 2025*

This repository investigates the biases of the below risk estimation models. The main examinations are the ROC curve and threshold-based metrics (true/false positive/negative rates, etc.). However, code exploring score distributions, linear regressions, calibration curve disparities, and precision-recall curves (PRC) is also inlcluded. We evaluate on the NLST and DLCST datasets.

We evaluate the following risk estimation models:

* Venkadesh21 (aka "Kiran's Model"): [DIAGNijmegen/bodyct-nodule-malignancy](https://github.com/DIAGNijmegen/bodyct-nodule-malignancy)

* De Haas Models (aka "Tijmen's Model"): [DIAGNijmegen/bodyct-lung-malignancy](https://github.com/DIAGNijmegen/bodyct-lung-malignancy)

* Sybil (from [MIT Jameel Clinic](https://github.com/reginabarzilaygroup/Sybil)): [DIAGNijmegen/bodyct-sybil-lung-cancer-risk-2](https://github.com/DIAGNijmegen/bodyct-sybil-lung-cancer-risk-2)

* PanCan2b (Brock Malignancy Calculator): included in [DIAGNijmegen/bodyct-common](https://github.com/DIAGNijmegen/bodyct-common/blob/master/clinical_models/BrockMalignancyCalculator.py)

## Overview of Files

In the main directory, Jupyter (`.ipynb`) notebooks evaluations start with the following naming scheme: `METRIC_DATASET_CTTYPE`. 

* Metrics: `roc`, `thresholds`, `scoredists`, `linreg`, `prc`, `calibrationdiffs`
* Datasets: `nlst` or `dlcst`
* CT Type: `nodules` or `scans`

After these fields, there are sometimes some other indicatiosn:

* `tijmen` indicates that NLST the evaluation includes the De Haas combined model. This model's final linear layer was trained on a different validation split (and not like the cross-validation for the other De Haas and Venkadesh models), and so it can be only validated on a subset 20% of the size of the other validation set.

* `confound_DEMOGRAPHIC`: we see if the performance disparity of that demographic persists when isolating for confounders.

There are also other files.

* `false_MODEL_DATASET`: analyzing false positives and false negatives from a particular model on a particular dataset.

* `save_all_results`: saving results which are plotted in other files to CSV and inspecting them.

* `thesis_tables`: make tables using nice labels for the whole thesis evaluation.

* `calibrate_preds`: Apply calibrations to the Venkadesh and De Haas models. Sybil and PanCan2b are already calibrated.

## Subdirectories

Here's what the subfolders contain.

* `utilities`: Common functions and constants for the other files.

* `slicecount`: Extract slice counts from series in NLST for analysis.

* `mhatodicom`: Convert MHA to DICOM for Sybil predictions.

* `transform_attentions`: Perform file conversions for Sybil attention visualizations.

* `nlst`: Various notebooks grabbing information from NLST data. This includes predictions, demographics, and analyzing demographics and clinical confounders.




