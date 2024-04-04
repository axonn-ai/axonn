#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:05:00 
#SBATCH -A bhatele-lab-cmsc 

#CHANGE IF YOUR DATA IS SOMEWHERE ELSE
DATA_DIR="../data/"

G_INTRA_ROW=2
G_INTRA_COL=1

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 

torchrun --nproc_per_node 2 train.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-data 1  --micro-batch-size 4 --checkpoint-activations

