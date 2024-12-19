#!/bin/bash
#SBATCH --job-name=pokediffusion_train_256
#SBATCH --mail-user=smaley@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --output pokegan-diffusion-256-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --time=168:00:00
#SBATCH --gpus=a100:1
#SBATCH --account=cap4773
#SBATCH --qos=cap4773

date;hostname;pwd
module load conda
conda activate improved-diffusion

export OPENAI_LOGDIR="log_256" 

python scripts/image_train.py --data_dir /blue/rcstudents/smaley/pokegan/customsprites/blk_bg --image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 16 --resume_checkpoint /blue/rcstudents/smaley/pokegan/improved-diffusion/log_256/model_ckpts/model990000.pt

date;hostname;pwd

