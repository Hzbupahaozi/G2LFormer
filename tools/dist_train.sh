#!/usr/bin/env bash
# CONFIG=$1
# GPUS=$2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:3} \
#     --no-validate

# if [ $# -lt 3 ]
# then
#     echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
#     exit
# fi

CONFIG=$1
WORK_DIR=$2
GPUS=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


if [ ${GPUS} == 1 ]; then
    python3 $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} ${@:4} --no-validate
else
    python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}  --no-validate
fi

# if [ ${GPUS} == 1 ]; then
#     python3 $(dirname "$0")/train.py  $CONFIG --work-dir=${WORK_DIR} ${@:4}
# else
#     python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#         $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --launcher pytorch ${@:4}
# fi
