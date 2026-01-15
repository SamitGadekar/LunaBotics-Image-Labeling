#!/usr/bin/bash

#SBATCH --job-name=grounded_sam
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=csso
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=sam_%j.out
#SBATCH --error=sam_%j.err

module load cuda anaconda

# make sure to create the environment and use python 3.13
conda activate /scratch/gilbreth/raghav21/conda/lunabotics_env


cd /path/to/image_labeling

OMP_NUM_THREADS=80

# the paths for the bags is very vague, will need to be changed for better structure 
# (ex. outputs, scripts, drive, etc.)

# can change labels, input/output dir, and threshold
python automated_grounded_sam.py \
  --input_dir /scratch/gilbreth/raghav21/outputs/frames \
  --output_dir /scratch/gilbreth/raghav21/outputs/labels \
  --labels rocks \ 
  --threshold 0.3 \
  --ignore annotated \
  --no_progress
