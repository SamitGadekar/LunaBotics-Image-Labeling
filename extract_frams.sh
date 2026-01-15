#!/usr/bin/bash

#SBATCH --job-name=frame_extraction
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=csso
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=frames_%j.out
#SBATCH --error=frames_%j.err

module load cuda anaconda
conda activate /scratch/gilbreth/raghav21/conda/lunabotics_env

cd /path/to/image_labeling

OMP_NUM_THREADS=80

python frame_extraction.py 2025
python frame_extraction.py 2024
