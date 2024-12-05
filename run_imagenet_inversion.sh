#!/bin/bash
#SBATCH --output="%J.out"
#SBATCH --error="%J.err"
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:volta:1
#SBATCH --time=2:00:00

# Record the start time
START_TIME=$(date +%s)

echo "Start: $(date)"

### Activate conda environment
source $HOME/miniconda3/bin/activate csng

### Run the notebook
cd $HOME/cs-433-project

export $(cat .env | xargs)

# Using the SLURM job ID to create a unique filename for the output notebook
jupyter nbconvert --to notebook --execute --output resnet_inversion_${SLURM_JOB_ID}.ipynb csng/brainreader_mouse/resnet_inversion.ipynb

nvidia-smi

echo "End: $(date)"

# Record the end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# Format the elapsed time as HH:MM:SS
printf "Job Duration: %02d:%02d:%02d\n" $((ELAPSED_TIME/3600)) $(((ELAPSED_TIME%3600)/60)) $((ELAPSED_TIME%60))
