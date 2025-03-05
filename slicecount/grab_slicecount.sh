#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=0
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --qos=high
#SBATCH --no-container-remap-root
#SBATCH --container-mounts=/data/bodyct:/data/bodyct
#SBATCH --container-image="doduo.umcn.nl#uokbaseimage/diag:tf2.12-pt2.0-v1" 

python3 /data/bodyct/experiments/lung-malignancy-fairness-shaurya/gitrepo/slicecount/grab_slicecount.py
