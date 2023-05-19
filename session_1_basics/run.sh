#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00 
#SBATCH -A isc2023-aac

DATA_DIR="/scratch/zt1/project/isc2023/shared/"

. /scratch/zt1/project/isc2023/shared/tutorial-venv/bin/activate

python train_mp.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations

