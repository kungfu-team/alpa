#!/bin/bash
set -e

ray start --head

python main.py \
    --workdir=/tmp/mnist \
    --config=configs/default.py \
    --config.batch_size 8192 \
    --use_ray \
    > /data/training.log 2>&1
