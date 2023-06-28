#!/bin/bash

nvidia-smi

sleep 10

ray start --address='controller:6379'

tail -f /dev/null
