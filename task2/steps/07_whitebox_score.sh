#!/bin/bash
# White box: every collected answer is scored in teacher forcing under the identified local checkpoint
# (no request to the task server). One pickle per (checkpoint, endpoints, variant) in $T2_WORK/whitebox/out/.
#   dec   : base model, with decoding statistics      fp8t : weights cast to fp8 e4m3, one scale per tensor
#   watch : expected mass of the 3,000 most frequent tokens (token-frequency residuals)
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
export T2W="$T2_WORK/whitebox" CUDA_VISIBLE_DEVICES=$T2_GPU TOKENIZERS_PARALLELISM=false
cd "$D/src/whitebox"
QB=Qwen/Qwen2.5-1.5B-Instruct; QA=Qwen/Qwen2.5-3B-Instruct; LL=unsloth/Llama-3.2-1B-Instruct; SM=HuggingFaceTB/SmolLM2-360M-Instruct
s() { $PY score.py "$@" --lite >> "$T2_WORK/logs/07_score.log" 2>&1; }
# identification of the checkpoints (NLL per token; compare with ident.py)
s --repo $QB --kind ep --families QWEN_B --sys tmpl --tag dec --dec
s --repo $QA --kind ep --families QWEN_A --sys tmpl --tag dec --dec
s --repo $LL --kind ep --families LLAMA --sys date --tag dec --dec
s --repo $SM --kind ref --sys tmpl --tag dec --dec
s --repo $QA --kind ep --families QWEN_A --sys tmpl --tag tmpl
s --repo Qwen/Qwen3-1.7B --kind ep --families QWEN_A --sys tmpl --tag tmpl --per-endpoint 30
s --repo $QB --kind ep --families QWEN_B --sys none --tag none
s --repo $QB --kind ep --families QWEN_B --sys "You are a helpful assistant." --tag helpful
# simulated quantisation (fp8t is the variant that separates ref-02; the others are kept for comparison)
for q in fp8t fp8c fp8ch fp8raw int8c; do s --repo $QB --kind ep --families QWEN_B --sys tmpl --tag $q --quant $q; done
for q in fp8t fp8c fp8ch fp8raw int8c; do s --repo $SM --kind ref --sys tmpl --tag $q --quant $q; done
s --repo $LL --kind ep --families LLAMA --sys date --tag fp8t --quant fp8t
s --repo $QA --kind ep --families QWEN_A --sys tmpl --tag fp8t --quant fp8t
# token-frequency residuals
s --repo $SM --kind ref --sys tmpl --tag watch --watch 3000
s --repo $QB --kind ep --families QWEN_B --sys tmpl --tag watch --watch 3000
s --repo $LL --kind ep --families LLAMA --sys date --tag watch --watch 3000
s --repo $QA --kind ep --families QWEN_A --sys tmpl --tag watch --watch 3000
ls "$T2W/out"
