#!/usr/bin/env bash
SBATCH --output="%J.out"
SBATCH --error="%J.err"
SBATCH --partition=gpu
SBATCH --qos=gpu_free
SBATCH --gres=gpu:volta:2

echo "Start: $(date)"
nvidia-smi

### activate conda environment
source $HOME/miniconda3/bin/activate csng

### run the python script
cd $HOME/cs-433-project
export $(cat .env | xargs)
python csng/brainreader_mouse/train_encoder.py

echo "End: $(date)"

