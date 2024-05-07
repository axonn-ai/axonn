#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128
#SBATCH --time=00:05:00
#SBATCH -A isc2023-aac


DATA_DIR="../data"

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 


cmd="torchrun --nproc_per_node 2 train_deepspeed.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --checkpoint-activations --deepspeed_config ./ds_config.json" 

echo "${cmd}"

eval "${cmd}"
