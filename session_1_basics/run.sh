## normal training
DATA_DIR="/scratch/zt1/project/bhatele-lab/shared/"
python train_mp.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations

