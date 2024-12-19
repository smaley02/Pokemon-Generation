#!/bin/bash
#SBATCH --job-name=pokediffusion_sample_256
#SBATCH --mail-user=smaley@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --output sample-1710000-256-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --partition=gpu
#SBATCH --time=168:00:00
#SBATCH --gpus=a100:1
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents

date;hostname;pwd
module load conda
conda activate improved-diffusion

export OPENAI_LOGDIR="log_256" 

python scripts/image_sample.py --model_path /blue/rcstudents/smaley/pokegan/improved-diffusion/log_256/ema_0.9999_1710000.pt --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --num_samples 900

date;hostname;pwd