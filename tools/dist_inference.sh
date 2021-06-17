#!/usr/bin/env bash

set -Eeo pipefail

trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
}

NUMGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    NUMGPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))
fi

CONFIG=$1
CHECKPOINT=$2
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$NUMGPUS --master_port=$PORT \
    $(dirname "$0")/inference.py  $CONFIG $CHECKPOINT --launcher pytorch ${@:3}
