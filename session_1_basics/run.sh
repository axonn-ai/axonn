## normal training
python train.py --num-layers 22 --hidden-size 2048 --data-dir /scratch0/ssingh37/ --batch-size 32 --lr 0.001 --image-size 28

## with mixed precision
python train_mp.py --num-layers 22 --hidden-size 2048 --data-dir /scratch0/ssingh37/ --batch-size 32 --lr 0.001 --image-size 28 

## with mixed precision and activation checkpointing
python train_mp.py --num-layers 22 --hidden-size 2048 --data-dir /scratch0/ssingh37/ --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations

