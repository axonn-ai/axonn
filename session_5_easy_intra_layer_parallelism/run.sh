#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:05:00 
#SBATCH -A bhatele-lab-cmsc 


# Set the data directory to the directory containing the MNIST dataset
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate up one level in the directory hierarchy
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Set the path to the dataset directory relative to the parent directory
DATA_DIR="${PARENT_DIR}/session_1_basics"


#module load gcc/9.4.0 openmpi/gcc

#. /home/jwendlan/tutorial-venv/bin/activate

G_INTRA_ROW=1
G_INTRA_COL=2

cmd="NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" torchrun --nproc-per-node 2 train.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-data 1  --micro-batch-size 4 --checkpoint-activations"

echo ${cmd}
eval ${cmd}
