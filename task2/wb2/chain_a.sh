#!/bin/bash
cd ~/t2wb/code; Q=Qwen/Qwen2.5-1.5B-Instruct; L=unsloth/Llama-3.2-1B-Instruct
./run_gpu.sh 0 score.py --repo $Q --kind ep --families QWEN_B --sys tmpl --tag dec --dec --lite
./run_gpu.sh 0 score.py --repo $L --kind ep --families LLAMA --sys date --tag dec --dec --lite
for q in fp8t fp8c fp8ch fp8raw int8c; do ./run_gpu.sh 0 score.py --repo $Q --kind ep --families QWEN_B --sys tmpl --tag $q --quant $q --lite; done
