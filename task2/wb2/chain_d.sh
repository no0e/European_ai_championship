#!/bin/bash
cd ~/t2wb/code
./run_gpu.sh 3 score.py --repo HuggingFaceTB/SmolLM2-360M-Instruct --kind ref --sys tmpl --tag watch --watch 3000 --lite
./run_gpu.sh 3 score.py --repo Qwen/Qwen2.5-1.5B-Instruct --kind ep --families QWEN_B --sys tmpl --tag watch --watch 3000 --lite
./run_gpu.sh 3 score.py --repo unsloth/Llama-3.2-1B-Instruct --kind ep --families LLAMA --sys date --tag watch --watch 3000 --lite
