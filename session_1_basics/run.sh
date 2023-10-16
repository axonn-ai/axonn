#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:05:00
#SBATCH -A isc2023-aac

DATA_DIR="/scratch/zt1/project/isc2023/shared/"

. /scratch/zt1/project/isc2023/shared/tutorial-venv/bin/activate

SCRIPT=train.py
ARGS="--num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64"

MIXED_PRECISION="${MIXED_PRECISION:=false}"
CHECKPOINT_ACTIVATIONS="${CHECKPOINT_ACTIVATIONS:=false}"

if [ ${MIXED_PRECISION} == "true" ]; then
	SCRIPT="train_mp.py"
fi

if [ ${CHECKPOINT_ACTIVATIONS} == "true" ]; then
	SCRIPT="train_mp.py"
	ARGS="${ARGS} --checkpoint-activations"
fi

cmd="python ${SCRIPT} ${ARGS}"
echo $cmd

python ${SCRIPT} ${ARGS}

