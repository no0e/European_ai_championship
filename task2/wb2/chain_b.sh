#!/bin/bash
cd ~/t2wb/code; S=HuggingFaceTB/SmolLM2-360M-Instruct
./run_gpu.sh 3 score.py --repo $S --kind ref --sys tmpl --tag dec --dec --lite
./run_gpu.sh 3 score.py --repo Qwen/Qwen2.5-3B-Instruct --kind ep --families QWEN_A --sys tmpl --tag dec --dec --lite
for q in fp8t fp8c fp8ch fp8raw int8c; do ./run_gpu.sh 3 score.py --repo $S --kind ref --sys tmpl --tag $q --quant $q --lite; done
