import os
import sys
import time
import pandas as pd
import SimpleITK as sitk
import numpy as np
import time
from pathlib import Path
import multiprocessing
import mhatodicom

DATASET_NAME = "nlst"
LOCAL_PC = True
root_dir = "/mnt/w" if LOCAL_PC else "/data/bodyct"
EXPERIMENT_DIR = f"{root_dir}/experiments/lung-malignancy-fairness-shaurya"
DATA_DIR = f"{EXPERIMENT_DIR}/{DATASET_NAME}"

csv_path = f"{DATA_DIR}/nlst_demo_v1_w4preds.csv"

print("Reading DataFrame")
df = pd.read_csv(csv_path)
df = df[(~df['Thijmen_mean'].isna()) & (df['InSybilTrain'] == False)]
series_instance_uids = pd.unique(df['SeriesInstanceUID']).tolist()
print(len(series_instance_uids), " series instance uids")

def check_mha_dicom(mha_file, output_dir, pixel_dtype=np.int16):
    try:
        assert pixel_dtype in [np.int16, np.float64]

        if pixel_dtype == np.int16:
            mha_image = sitk.ReadImage(str(mha_file), sitk.sitkInt16)
        elif pixel_dtype == np.float64:
            mha_image = sitk.ReadImage(str(mha_file), sitk.sitkFloat64)

        fail_slices = mhatodicom.check_all_slices_created(output_dir, mha_image.GetDepth())
        
        if len(fail_slices) > 0:
            print(f"ERROR: {output_dir} - slices missing: {fail_slices}", file=sys.stderr)
            return False
        return True
    except:
        print(f"ERROR: {output_dir} - something went wrong :(", file=sys.stderr)
        return False
    
def mha_dicom_process(i, total, series_instance_uid):
    src_dir = f"{root_dir}/experiments/0-{DATASET_NAME}-mha"
    output_folder_path = f"{DATA_DIR}/DICOM_files"

    mha_filename = series_instance_uid + '.mha'
    mha_filepath = os.path.join(src_dir, mha_filename)

    print(f"{i+1} / {total}: checking {series_instance_uid} ... ")
    if os.path.exists(mha_filepath):
        output_subfolder = mhatodicom.process_mha_file(mha_filepath, output_folder_path)
        success = check_mha_dicom(mha_filepath, output_subfolder)
        if success:
            print(f"{i+1} / {total}: SUCCESSfully converted")
            return True
        if not success: 
            print(f"{i+1} / {total}: FAILED to convert {series_instance_uid}", file=sys.stderr)
            return False
    else:
        print(f"File {mha_filename} not found in the source directory.")
        return False

print("Starting!")
converterpool = multiprocessing.Pool()
result_bools = converterpool.starmap(mha_dicom_process, 
                    zip(range(len(series_instance_uids)), [len(series_instance_uids)] * len(series_instance_uids), series_instance_uids))
print("Finished!")