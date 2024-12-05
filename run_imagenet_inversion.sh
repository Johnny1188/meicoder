#!/bin/bash
#SBATCH --output="%J.out"
#SBATCH --error="%J.err"
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:volta:1
#SBATCH --time=20:00:00


echo "Start: $(date)"
nvidia-smi

### Activate conda environment
source $HOME/miniconda3/bin/activate csng

### Run the notebook
cd $HOME/cs-433-project

export $(cat .env | xargs)

# Using the SLURM job ID to create a unique filename for the output notebook
jupyter nbconvert --to notebook --execute --output resnet_inversion_${SLURM_JOB_ID}.ipynb csng/brainreader_mouse/resnet_inversion.ipynb

echo "End: $(date)"
