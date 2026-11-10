#!/bin/bash
# Overnight v3 ring sweep: sign_flip, ALIE (z=1), and the no-attack control.
# 6 ratios x 5 seeds, 3-fold over vehicles, local batch size 128, 9 parallel jobs.
# Expected time on this Mac: about 4-5 h. Keep the laptop plugged in with the lid open.
#
# Usage (survives closing the terminal):
#   nohup bash code/run_overnight.sh >/dev/null 2>&1 &
# Progress: tail -f experiments/overnight.log
# Results:  experiments/report.txt (+ v3_*_f1.png)

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
R="0.2 0.3 0.4 0.5 0.6 0.7"
SD="1 2 3 4 5"
export JOBS=${JOBS:-9} EXTRA="--folds 3 --batch-size 128"
mkdir -p "$OUT"
LOG=$OUT/overnight.log
caffeinate -ims -w $$ &          # keep the Mac awake while this script runs

{
  echo "start $(date)"
  for atk in sign_flip alie; do
    echo "== $atk  $(date +%H:%M)"
    bash "$HERE/run_sweep_v3.sh" "$R" "$SD" 0.5 $atk | grep -E "FAILED|finished|datasets"
  done
  echo "== none (control)  $(date +%H:%M)"
  METHODS="FedAvg,Median,Trust-FLAD+Median" bash "$HERE/run_sweep_v3.sh" "$R" "$SD" 0.5 none \
    | grep -E "FAILED|finished"
  echo "end $(date)"
} > "$LOG" 2>&1

{
  for atk in sign_flip alie none; do
    echo "################ $atk"
    "$PY" "$HERE/plot_v3.py" "$OUT/results" $atk 2>/dev/null
    echo
  done
  echo "App PDR range over datasets:"
  grep -h "App PDR" "$OUT"/data/*_summary.txt | awk '{print $3}' | sort -n | sed -n '1p;$p'
} > "$OUT/report.txt"
echo "done: $OUT/report.txt" >> "$LOG"
