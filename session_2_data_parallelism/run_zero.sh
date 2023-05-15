DATA_DIR="/scratch/zt1/project/bhatele-lab/shared/"

mpirun -np 4 python train_zero.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations --deepspeed_config ./ds_config.json 
