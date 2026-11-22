# Shared helpers, sourced by every step.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$D/config.env"
S=$D/src; DN=$S/dense
mkdir -p "$T1_WORK/logs"; cd "$T1_WORK"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
G=($GPUS)
gpu() { echo "${G[$(($1 % ${#G[@]}))]}"; }          # i-th GPU of the list, round robin
train() { CUDA_VISIBLE_DEVICES=$1 $PY "$DN/train_dense.py" "${@:2}"; }
# folds <gpu> <method> <run> <init with {f}> <pseudo with {f} or none> <steps> <options...> : one fine-tuning per CV fold
folds() {
  local g=$1 m=$2 run=$3 init=$4 pseudo=$5 st=$6; shift 6
  for f in 0 1 2 3 4; do
    [ -f runs/$run/f$f/${m}_test.npz ] && continue
    local ps=""; [ "$pseudo" != none ] && ps="--pseudo ${pseudo//\{f\}/$f} --pseudo_frac 0.2"
    train $g --method $m --run $run --fold $f --init ${init//\{f\}/$f} $ps "$@" --steps $st --eval_every $st \
      >> logs/cv_${m}_${run}.log 2>&1
  done
  echo "$m $run: $(ls runs/$run/f*/${m}_test.npz 2>/dev/null | wc -l) folds"
}
