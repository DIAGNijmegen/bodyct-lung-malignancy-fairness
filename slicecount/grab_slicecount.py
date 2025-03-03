import os
import pandas as pd
import SimpleITK as sitk
import sys

CHANSEY_ROOT = "/data/bodyct"
EXPERIMENT_DIR = f"{CHANSEY_ROOT}/experiments/lung-malignancy-fairness-shaurya"
NLST_PREDS = f"{EXPERIMENT_DIR}/nlst"  ## Comment out if not using Teams backup (aka Chansey is up :)
MHADIR_PATH = f"{CHANSEY_ROOT}/experiments/0-nlst-mha"


def get_image_size(file_path):
    # Read the MHA file using SimpleITK
    image = sitk.ReadImage(file_path)
    # Get the size of the image (in the form [x, y, z])
    return image.GetSize()


def get_image_sizes_from_dataframe(df, directory):
    # Create a list to store results
    sizes = []

    # Loop through the filenames in the dataframe
    for i, filename in enumerate(df["SeriesInstanceUID"]):
        file_path = os.path.join(directory, f"{filename}.mha")
        try:
            size = get_image_size(file_path)
            print(f"{i+1} / {len(df)}: {filename}.mha ... size = {size}", end="\r")
            sizes.append(
                (filename, size[0], size[1], size[2])
            )  # Include filename in the result
        except Exception as e:
            print(f"{i+1} / {len(df)}: {filename}.mha ... ERROR = {e}", end="\r")
            sizes.append(
                (filename, None, None, None)
            )  # If an error occurs, append None for all dimensions

    # Create a new dataframe with filename, x, y, z columns
    sizes_df = pd.DataFrame(
        sizes, columns=["SeriesInstanceUID", "series_x", "series_y", "series_z"]
    )
    return sizes_df


if __name__ == "__main__":

    df = pd.read_csv(f"{NLST_PREDS}/nlst_demov4_allmodels_cal.csv").drop_duplicates(
        subset="SeriesInstanceUID"
    )
    # Get image sizes and append to the original DataFrame
    sizes_df = get_image_sizes_from_dataframe(df, MHADIR_PATH)
    sizes_df.to_csv(f"{EXPERIMENT_DIR}/nlst/nlst_demov4_scan_sizes.csv", index=False)

    merged_df = pd.merge(df, sizes_df, on="filename", how="left")
    sizes_df.to_csv(f"{EXPERIMENT_DIR}/nlst/nlst_demov4_allmodels_cal.csv", index=False)
