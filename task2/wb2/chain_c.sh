#!/bin/bash
cd ~/t2wb/code; Q=Qwen/Qwen2.5-1.5B-Instruct
./run_gpu.sh 0 score.py --repo $Q --kind ep --families QWEN_B --sys none --tag none --lite
./run_gpu.sh 0 score.py --repo $Q --kind ep --families QWEN_B --sys "You are a helpful assistant." --tag helpful --lite
./run_gpu.sh 0 score.py --repo Qwen/Qwen2.5-3B-Instruct --kind ep --families QWEN_A --sys tmpl --tag tmpl --lite
./run_gpu.sh 0 score.py --repo Qwen/Qwen3-1.7B --kind ep --families QWEN_A --sys tmpl --tag tmpl --lite --per-endpoint 30
