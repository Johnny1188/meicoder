#!/bin/bash
#SBATCH --output="%J.out"
#SBATCH --error="%J.err"
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:volta:1
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mikulas.vanousek@epfl.ch

# Record the start time
START_TIME=$(date +%s)

echo "Start: $(date)"

### Activate conda environment
source $HOME/miniconda3/bin/activate csng

### Run the notebook
cd $HOME/cs-433-project

export $(cat .env | xargs)

# Using the SLURM job ID to create a unique filename for the output notebook
OUTPUT_FILE=resnet_inversion_${SLURM_JOB_ID}.ipynb
jupyter nbconvert --to notebook --execute --allow-errors --output $OUTPUT_FILE csng/imagenet/resnet_inversion.ipynb

# Remove write permission for everyone, including yourself
chmod -w $OUTPUT_FILE

echo "End: $(date)"

# Record the end time and calculate duration
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

# Format the elapsed time as HH:MM:SS
printf "Job Duration: %02d:%02d:%02d\n" $((ELAPSED_TIME/3600)) $(((ELAPSED_TIME%3600)/60)) $((ELAPSED_TIME%60))
