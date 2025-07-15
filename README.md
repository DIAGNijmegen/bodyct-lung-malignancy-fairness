# Fairness Evaluation of Malignancy Risk Estimation Models for Lung Cancer Screening

Shaurya Gaur, Fennie van der Graaf, Lena Philipp, Michel Vitale, Alessa Hering, Colin Jacobs

*FAIMI Special Issue 2025, MELBA Journal*

This repository investigates the biases of the below risk estimation models. The main examinations are the ROC curve and threshold-based metrics (true/false positive/negative rates, etc.). However, code exploring score distributions, linear regressions, calibration curve disparities, and precision-recall curves (PRC) is also inlcluded. We evaluate on the NLST dataset.

We evaluate the following risk estimation models:

* Venkadesh21: [DIAGNijmegen/bodyct-nodule-malignancy](https://github.com/DIAGNijmegen/bodyct-nodule-malignancy)

* Sybil (from [MIT Jameel Clinic](https://github.com/reginabarzilaygroup/Sybil)): [DIAGNijmegen/bodyct-sybil-lung-cancer-risk](https://github.com/DIAGNijmegen/bodyct-sybil-lung-cancer-risk)

* PanCan2b (Brock Malignancy Calculator): included in [DIAGNijmegen/bodyct-common](https://github.com/DIAGNijmegen/bodyct-common/blob/master/clinical_models/BrockMalignancyCalculator.py)

## Process

The pipeline follows these steps, which are outlined in the files here.

1. **Load predictions** (`collect_preds.ipynb`): This performs the following steps.

    - Load Venkadesh21 and PanCan2b NLST predictions (requires one spreadsheet for both).
    - Calibrate the Venkadesh21 model's NLST predictions using Platt's scaling.
    - Merge predictions together into one sheet. 
    - Load Sybil split data and determine which series are appropriate to collect validation predictions.
    - Load Sybil's inference on those validation series (in Sybil repo) and merge.

2. **Load demographics** (`collect_demos.ipynb`): Take in a NLST participant dictionary, and collect demographic and confounder information to add to the dataset of predictions.

3. **Subgroup Performance Analysis** (`save_subgroup_analysis.ipynb`): Run and save results from subgroup performance analysis. This collects AUC scores and threshold-based metrics (sensitivity, specificity, etc.) for the demographics and confounders collected above. It does this on the NLST scan--level sets for all of the models.

4. **Tables for Results and Appendix** (`thesis_tables.ipynb`): This makes the relevant tables for the results collected so far. Also includes tables for the confounder analysis (below). Here, we can easily also create ROC and threshold plots for the figures of results we want to see more closely.

5. **Save Confounder Analysis** (`save_confounder_analysis.ipynb`): Based on the results, collect performance results for select demographic groups, isolating for other characteristics (potential confounders).

## Overview of Files

The following files are required (in the `FILE_DIR` directory as labeled in `utilities/info.py`).

- Venkadesh21 and PanCan predictions: `NLST_DL_vs_PanCan_Venk21.csv`.
- Sybil PatientID to Split information: `sybil-nlst-pid_tp_series2split.p`
- Sybil Inferences: `sybil-inference-1172.csv` and `sybil-inference-4739.csv` (Could run as one job as well.)
- NLST Participant Dictionary: `participant_d040722.csv`

Files for the NLST predictions merged, with and without demographic columns, are generated into `FILE_DIR`. 

Performance analysis results are found in a separate results directory (`RESULTS_DIR`, also in `utilities/info.py`).

The `thesis_tables.ipynb` file will generate tables into `TAB_DIR` and figures into `FIG_DIR` (for drag-drop into Overleaf).


