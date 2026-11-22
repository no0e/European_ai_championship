#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=02:00:00
#SBATCH --job-name=t3_k4
#SBATCH --output=/p/home/jusers/lamirand1/jureca/k3q/logs/job_%j.out
# File de commandes pilotée depuis le HOME (monté partout) : ~/k3q/queue.txt, une commande par ligne, préfixe = n° de GPU ou *.
# Les commandes s exécutent dans unlearning/k3 (scratch). Journaux dans ~/k3q/logs. STOP dans la file = fin.
Q=$HOME/k3q
cd /p/scratch/training2666/diette1/Sevel_Labs_ENSAI/unlearning/k3 || { echo "scratch inaccessible"; exit 1; }
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
touch $Q/queue.txt $Q/submit.txt
# code déposé dans le home -> scratch ; état du scratch -> home (lisible depuis n importe quel noeud de connexion)
cp $Q/stage/k3/* . ; cp $Q/stage/k3sim/* sim/ ; mkdir -p ../k4/logs ; cp $Q/stage/k4/* ../k4/
sync_out() { cp ../output/submit_log.txt logs/queue.log $Q/out/ 2>/dev/null; ls sub > $Q/out/sub_ls.txt; grep -h "alpha=" logs/real_*.log > $Q/out/real_alpha.txt 2>/dev/null; python res.py sim/C > $Q/out/res_C.txt 2>/dev/null; grep -H "alpha=" logs/real_*.log | cut -c1-260 > $Q/out/real_alpha.txt; }
sync_out
echo "internet : $(curl -s -m 10 -o /dev/null -w "%{http_code}" http://34.147.229.55/ 2>&1)" > $Q/out/internet.txt
# soumissions : une ligne = un fichier (relatif à unlearning/) dans ~/k3q/submit.txt, envoyées l une après l autre
( while true; do f=$(head -1 $Q/submit.txt 2>/dev/null); if [ -n "$f" ]; then sed -i 1d $Q/submit.txt; (cd .. && python k3/submit.py $f) >> $Q/out/submit.out 2>&1; cp ../output/submit_log.txt $Q/out/; else sleep 5; fi; done ) &
( while true; do sleep 60; sync_out; done ) &
declare -A PID
n=0
while true; do
  for g in 0 1 2 3; do
    if [ -n "${PID[$g]:-}" ] && kill -0 "${PID[$g]}" 2>/dev/null; then continue; fi
    line=$(grep -m1 -n "^$g " $Q/queue.txt)
    [ -z "$line" ] && line=$(grep -m1 -n "^\* " $Q/queue.txt)
    [ -z "$line" ] && continue
    ln=${line%%:*}; cmd=${line#*:}; cmd=${cmd#* }
    sed -i "${ln}d" $Q/queue.txt
    n=$((n+1)); echo "$(date +%H:%M:%S) GPU $g [$n] $cmd" >> $Q/logs/queue.log
    ( CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=6 bash -c "$cmd" > "$Q/logs/cmd_${SLURM_JOB_ID}_$n.log" 2>&1; echo "$(date +%H:%M:%S) fin [$n] code $?" >> $Q/logs/queue.log ) &
    PID[$g]=$!
  done
  if grep -q "^STOP" $Q/queue.txt; then wait; break; fi
  sleep 3
done
echo fin
