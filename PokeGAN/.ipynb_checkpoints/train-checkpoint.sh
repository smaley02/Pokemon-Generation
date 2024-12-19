#!/bin/bash
#SBATCH --job-name=pokegan_train
#SBATCH --mail-user=smaley@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --output pokegan-train-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --gpus=a100:1
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents

date;hostname;pwd
module load conda
conda activate pokegan

python train.py

date;hostname;pwd

