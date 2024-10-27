#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=0
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --qos=high
#SBATCH --nodelist=dlc-groudon
#SBATCH --no-container-remap-root
#SBATCH --container-mounts=/data/bodyct:/data/bodyct
#SBATCH --container-image="doduo1.umcn.nl#uokbaseimage/diag:tf2.12-pt2.0-v1" 
#SBATCH --output=mhatodicom.out
#SBATCH --error=mhatodicom.err

python3 /data/bodyct/experiments/lung-malignancy-fairness-shaurya/gitrepo/mhatodicom/mhatodicom.py
