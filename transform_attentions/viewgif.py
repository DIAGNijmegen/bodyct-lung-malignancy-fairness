from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

ATTENTION_GIF_DIR_DIR = (
    f"V:/experiments/lung-malignancy-fairness-shaurya/nlst/sybil_attentions/attention"
)
ATTENTION_IMG_DIR = f"V:/experiments/lung-malignancy-fairness-shaurya/nlst/sybil_attentions/attention_imgs"


def get_frames_from_gif(gif_path):
    gif = Image.open(gif_path)
    frames = []
    while True:
        frames.append(np.array(gif))
        try:
            gif.seek(gif.tell() + 1)
        except EOFError:
            break

    return frames


def show_gif_slider(gif_path):
    # Load the GIF using PIL (Pillow)
    frames = get_frames_from_gif(gif_path)

    # Number of frames in the GIF
    num_frames = len(frames)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)
    im = ax.imshow(frames[0], cmap="gray")
    ax.axis("off")

    # Add the slider
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(ax_slider, "Frame", 0, num_frames - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        frame_idx = int(slider.val)
        im.set_data(frames[frame_idx])
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        f"V:/experiments/lung-malignancy-fairness-shaurya/nlst/sybil_fn_brock_top25.csv"
    )

    pancan_cols = [
        "Age",
        "Gender",
        "race",
        "FamilyHistoryLungCa",
        "Emphysema",
        "Diameter [mm]",
        "NoduleInUpperLung",
        "PartSolid",
        "NoduleCounts",
        "Spiculation",
    ]
    nodule_loc = ["CoordX", "CoordY", "CoordZ", "loclup", "locrup"]
    cols_to_show = pancan_cols + [
        "weight",
        "BMI",
        "Adenocarcinoma",
        "Squamous_cell_carcinoma",
        "diaghype",
        "wrkasbe",
        "wrkfoun",
        "cigar",
        "pipe",
    ]

for i in range(1):
    seriesuid = df["SeriesInstanceUID"][i]
    print("Series ID:", seriesuid)
    gif_path = f"{ATTENTION_GIF_DIR_DIR}/serie_{seriesuid}.gif"
    print(df.iloc[i][pancan_cols + nodule_loc])
    show_gif_slider(gif_path)
