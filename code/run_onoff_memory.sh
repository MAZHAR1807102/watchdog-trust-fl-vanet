#!/bin/bash
# Slow-recovery trust against on-off grayholes, for Paper A (grayhole task, V2V only, no uploads).
# Existing on-off data (drop 0.9 for 10 s, honest 20 s): data/m*_d0.9_on10off20_s*. Trust memory M = 30, 60, 90 s
# (trust = worst windowed trust of the last M s). Same FL settings as results_onoff/ (sign_flip, 3 folds, batch 128).
# Waits for run_ablation.sh, runs as background QoS on the efficiency cores, resumable.
#   JOBS=6 nohup taskpolicy -b bash code/run_onoff_memory.sh >/dev/null 2>&1 &
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
JOBS=${JOBS:-6}
FL="--folds 3 --batch-size 128 --attack sign_flip --methods Trust-FLAD,Trust-FLAD+Krum,Trust-FLAD+Median"
LOG=$OUT/onoff_memory.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
caffeinate -ims -w $$ & disown
throttle() { while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done; }
{
  echo "waiting for run_ablation.sh $(date)"
  while pgrep -f run_ablation.sh >/dev/null; do sleep 60; done
  echo "start $(date)"
  for M in 30 60 90; do
    res=$OUT/results_onoff_mem$M
    mkdir -p "$res"
    for m in 0.2 0.3 0.4 0.5 0.6 0.7; do for s in 1 2 3 4 5; do
      grep -q "^wrote" "$res/m${m}_s${s}.log" 2>/dev/null && continue
      throttle
      ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/m${m}_d0.9_on10off20_s${s}_features.csv" --seed $s $FL \
          --trust-memory $M --out "$res" > "$res/m${m}_s${s}.log" 2>&1 || echo "FAILED mem$M m$m s$s" ) &
    done; done
  done
  wait
  echo "end $(date)"
} > "$LOG" 2>&1
echo "done" >> "$LOG"
