#!/bin/bash

# Job name
#SBATCH -J mask_vae

# Request GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 4

# Memory per node (adjust as needed)
#SBATCH --mem=16G

# Wall time (HH:MM:SS)
#SBATCH -t 24:00:00

# Output file
#SBATCH -o mask_vae-%j.out

# Error file
#SBATCH -e mask_vae-%j.err

# Email notifications (optional - uncomment and add your email)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anish_mitagar@brown.edu

# Load required modules
module load miniconda3/23.11.0s
module load cuda

# Activate virtual environment if you have one
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate med_img
which python

# Print some information
echo "Starting VAE training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script
python3 main_vae.py

echo "Job completed"