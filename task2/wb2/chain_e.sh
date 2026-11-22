#!/bin/bash
cd ~/t2wb/code
./run_gpu.sh 0 score.py --repo unsloth/Llama-3.2-1B-Instruct --kind ep --families LLAMA --sys date --tag fp8t --quant fp8t --lite
./run_gpu.sh 0 score.py --repo Qwen/Qwen2.5-3B-Instruct --kind ep --families QWEN_A --sys tmpl --tag fp8t --quant fp8t --lite
./run_gpu.sh 0 score.py --repo Qwen/Qwen2.5-3B-Instruct --kind ep --families QWEN_A --sys tmpl --tag watch --watch 3000 --lite
