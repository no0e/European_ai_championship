#!/bin/bash
# Black-box tests on the long answers (no request): permutation MMD, item agreement, LoRA dose-effect axis,
# then the structures compatible with the public diagnostic scores and the posterior of each endpoint (bayes.py).
set -euo pipefail; D=$(cd "$(dirname "$0")/.." && pwd); source "$D/config.env"
cd "$T2_WORK"; S=$D/src/structure; L=logs
$PY "$S/analyze.py" QWEN_A QWEN_B LLAMA > $L/06_analyze.log 2>&1
$PY "$S/fulltest.py" QWEN_A QWEN_B LLAMA > $L/06_fulltest.log 2>&1
$PY "$S/agree.py" > $L/06_agree.log 2>&1
$PY "$S/dose.py" > $L/06_dose.log 2>&1
$PY "$S/fit.py" > $L/06_fit.log 2>&1
$PY "$S/bayes.py" > $L/06_bayes.log 2>&1
grep "P(dans le groupe principal)" $L/06_bayes.log
