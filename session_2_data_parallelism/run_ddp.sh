DATA_DIR="/scratch/zt1/project/bhatele-lab/shared/"

## Command for DDP
mpirun -np 4 python train_ddp.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations

