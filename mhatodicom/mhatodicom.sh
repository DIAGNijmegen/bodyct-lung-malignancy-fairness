#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --qos=high
#SBATCH --nodelist=dlc-nidoking
#SBATCH --no-container-remap-root
#SBATCH --container-mounts=/data/bodyct:/data/bodyct
#SBATCH --container-image=doduo1.umcn.nl#fennievandergraaf/inference_sybil:0
#SBATCH --output=mhatodicom.out
#SBATCH --error=mhatodicom.err

python3 /data/bodyct/experiments/lung-malignancy-fairness-shaurya/gitrepo/mhatodicom/mhatodicom.py