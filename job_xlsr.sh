#!/bin/bash
#SBATCH --job-name=opensmile_g
#SBATCH --partition=H100-SLT,H100,H100-PCI,RTXA6000,L40S,A100-40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=10:00:00          
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

srun \
  --container-image=/netscratch/eerdogan/resnet_formants.sqsh \
  --container-mounts=/opt:/opt,/ds:/ds:ro,/ds-slt:/ds-slt,/netscratch/eerdogan:/netscratch/eerdogan,"$(pwd)":"$(pwd)" \
  --container-workdir="$(pwd)" \
  bash -c "
    echo 'Installing requirements...'
    pip install -r requirements.txt --quiet
    echo 'Requirements installed successfully!'
    echo 'Starting training...'
    python ./main_xlsr.py
  " 