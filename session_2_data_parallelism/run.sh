## Command for DDP
mpirun -np 2 python train_zero.py --num-layers 4 --hidden-size 2048 --data-dir /scratch0/ssingh37/ --batch-size 32 --lr 0.001 --image-size 64

## Command for DeepSpeed
deepspeed train_zero.py --num-layers 4 --hidden-size 2048 --data-dir /scratch0/ssingh37/ --batch-size 32 --lr 0.001 --image-size 64 --deepspeed_config ./ds_config.json
