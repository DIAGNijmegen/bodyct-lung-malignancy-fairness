import os
import time
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path

def read_csv_series_instance_uids(csv_path):
    """Reads the CSV file and returns a list of SeriesInstanceUID values."""
    df = pd.read_csv(csv_path)
    return df['SeriesInstanceUID'].tolist()

def write_slices(series_tag_values, new_img, out_dir, i, writer, spacing):
    image_slice = new_img[:, :, i]

    # Tags shared by the series.
    list(
        map(
            lambda tag_value: image_slice.SetMetaData(
                tag_value[0], tag_value[1]
            ),
            series_tag_values,
        )
    )

    # Slice specific tags.
    #   Instance Creation Date
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
    #   Instance Creation Time
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

    # Setting the type to CT so that the slice location is preserved and
    # the thickness is carried over.
    image_slice.SetMetaData("0008|0060", "CT")
    # Slice thickness
    image_slice.SetMetaData("0018|0050", str(spacing[2]))
    image_slice.SetMetaData("0008|0070", "AnonymousManufacturer")

    # (0020, 0032) image position patient determines the 3D spacing between
    # slices.
    #   Image Position (Patient)
    image_slice.SetMetaData(
        "0020|0032",
        "\\".join(map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))),
    )
    #   Instance Number
    image_slice.SetMetaData("0020|0013", str(i))

    # Write to the output directory and add the extension dcm, to force
    # writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir, str(i) + ".dcm"))
    writer.Execute(image_slice)

def mha_to_dicom(mha_file, output_dir, pixel_dtype=np.int16):
    assert pixel_dtype in [np.int16, np.float64]

    if pixel_dtype == np.int16:
        mha_image = sitk.ReadImage(str(mha_file), sitk.sitkInt16)
    elif pixel_dtype == np.float64:
        mha_image = sitk.ReadImage(str(mha_file), sitk.sitkFloat64)

    spacing = mha_image.GetSpacing()

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    direction = mha_image.GetDirection()
    series_tag_values = [
        ("0008|0031", modification_time),  # Series Time
        ("0008|0021", modification_date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        (
            "0020|000e",
            "1.2.826.0.1.3680043.2.1125."
            + modification_date
            + ".1"
            + modification_time,
        ),  # Series Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),  # Image Orientation
        ("0008|103e", "Created-SimpleITK"),  # Series Description
    ]

    if pixel_dtype == np.float64:
        rescale_slope = 0.001  # keep three digits after the decimal point
        series_tag_values = series_tag_values + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|1052", "0"),  # rescale intercept
            ("0028|0100", "16"),  # bits allocated
            ("0028|0101", "16"),  # bits stored
            ("0028|0102", "15"),  # high bit
            ("0028|0103", "1"),
        ]  # pixel representation

    list(
        map(
            lambda i: write_slices(series_tag_values, mha_image, output_dir, i, writer, spacing),
            range(mha_image.GetDepth()),
        )
    )

def process_mha_file(mha_path, base_output_folder):
    subfolder_name = os.path.splitext(os.path.basename(mha_path))[0]
    output_subfolder = os.path.join(base_output_folder, subfolder_name)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    return output_subfolder

if __name__ == "__main__":
    csv_path = r"W:\experiments\Sybil_Fennie\DLCST\Ensemble_Sybil\all_cases\missing_uids.csv"
    src_dir = r"W:\experiments\0-dlcst-mha"
    output_folder_path = r"W:\experiments\Sybil_Fennie\DLCST\DICOM_files\missinguids"

    series_instance_uids = read_csv_series_instance_uids(csv_path)

    for series_instance_uid in series_instance_uids:
        mha_filename = series_instance_uid + '.mha'
        mha_filepath = os.path.join(src_dir, mha_filename)
        if os.path.exists(mha_filepath):
            output_subfolder = process_mha_file(mha_filepath, output_folder_path)
            mha_to_dicom(mha_filepath, output_subfolder)
        else:
            print(f"File {mha_filename} not found in the source directory.")
