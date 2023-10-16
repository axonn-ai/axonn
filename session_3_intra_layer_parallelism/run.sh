#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:05:00 
#SBATCH -A isc2023-aac 

DATA_DIR="/scratch/zt1/project/isc2023/shared/"

module load gcc/9.4.0 openmpi/gcc

. /scratch/zt1/project/isc2023/shared/tutorial-venv/bin/activate

G_INTRA_ROW=2
G_INTRA_COL=2

cmd="mpirun -np 4 python train_axonn_intra_layer.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-data 1  --micro-batch-size 4 --checkpoint-activations"

echo ${cmd}
eval ${cmd}

