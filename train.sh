SUMMIT_FS_HOME=/gpfs/alpine/csc452/scratch/ssingh37/
export LC_CTYPE=en_US.UTF-8
export PAMI_ENABLE_STRIPING=1
export PAMI_IBV_ADAPTER_AFFINITY=1
export PAMI_IBV_DEVICE_NAME="mlx5_0:1,mlx5_3:1"
export PAMI_IBV_DEVICE_NAME_1="mlx5_3:1,mlx5_0:1"
export PYTHONPATH="/gpfs/alpine/csc452/scratch/ssingh37/axonn:$PYTHONPATH"


nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=$head
export MASTER_PORT=29500

G_inter=12
G_data=8
mbs=4
bs=16384
transformer_args='-N 48 -D 6336 -H 36'

jsrun --smpiargs='-gpu' -n 16 -a 6 -g 6 -c 42 -r 1 python -u examples/test_lm.py --G-inter $G_inter --G-data $G_data --micro-batch-size $mbs --batch-size $bs $transformer_args --dataset wikitext --cpu-offload

