#!/bin/bash

nvidia-smi

ray start --head --disable-usage-stats

sleep 5

python benchmark/alpa/benchmark.py \
    --suite gpt.perf_test_auto \
    --num-hosts 2 \
    --num-devices-per-host 2
# python benchmark/alpa/benchmark.py \
#     --suite gpt.perf_test_auto
