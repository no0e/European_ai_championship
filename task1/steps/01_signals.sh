#!/bin/bash
# Global log-likelihood of the known message under the official decoders -> regime (full / not full), 97 % right.
source "$(dirname "$0")/lib.sh"; set -euo pipefail
for s in train test; do
  CUDA_VISIBLE_DEVICES=$(gpu 0) $PY "$S/compute_signals.py" --split $s --data "$T1_DATA/release" --models "$T1_MODELS" \
    > logs/01_signals_$s.log 2>&1
done
ls -la "$T1_DATA"/release/signals_*.npz
