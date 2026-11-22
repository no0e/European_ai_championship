#!/bin/bash
# Signals read from the pickles of step 07, validated on the ref endpoints (known truth) before the ep endpoints.
# Every table goes to $T2_WORK/logs/08_*.log; endpoint_evidence.csv summarises what was read from them.
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
W=$D/src/whitebox; O=$T2_WORK/whitebox/out; L=$T2_WORK/logs; cd "$T2_WORK"
$PY "$W/ident.py" $O/*__dec.pkl $O/*__tmpl.pkl $O/*__none.pkl $O/*__helpful.pkl > $L/08_ident.log      # which checkpoint
$PY "$W/dec.py" $O/Qwen2.5-1.5B-Instruct__ep__dec.pkl QWEN_B > $L/08_decoding.log                      # pure temperature sampling
for spec in "SmolLM2-360M-Instruct ref -" "Qwen2.5-1.5B-Instruct ep QWEN_B" "Qwen2.5-3B-Instruct ep QWEN_A" "Llama-3.2-1B-Instruct ep LLAMA"; do
  set -- $spec
  $PY "$W/prof2.py" $O/$1__$2__dec.pkl $3 0.7 >> $L/08_temperature.log                                 # fitted temperature (T=1)
  $PY "$W/llr.py" $O/$1__$2__dec.pkl $O/$1__$2__fp8t.pkl 0.7 $([ "$3" = - ] || echo $3) >> $L/08_fp8_llr.log   # fp8
  $PY "$W/resid2.py" $O/$1__$2__watch.pkl $3 >> $L/08_residuals.log                                     # LoRA
done
$PY "$W/repetition.py" --out repetition.csv > $L/08_repetition.log                                       # watermark
tail -n 5 $L/08_repetition.log
