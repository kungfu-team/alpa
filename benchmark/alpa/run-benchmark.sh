#!/bin/bash

ray start --head

# python benchmark_one_case.py \
#     --model gpt \
#     --niter 100 \
#     --case gpt.perf_test_fast_2d \
#     --num-hosts 1 \
#     --num-devices-per-host 4

python benchmark.py \
    --suite gpt.perf_test_auto \
    --num-hosts 1 \
    --num-devices-per-host 4
