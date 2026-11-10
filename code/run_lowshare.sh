#!/bin/bash
# Realistic (low) attacker shares for Paper A: 0, 2 and 5 of 50 vehicles (m 0.0 / 0.04 / 0.1), drop 0.5.
# m0.0: no-attack only (does the filter cost anything when nobody attacks?).
# m0.04, m0.1: no-attack ceiling, sign-flip and label-flip, all 11 rules, 3 folds, batch 128.
# Data must exist (ns-3, see README.md). Resumable. Results -> results_lowshare/.
#   JOBS=6 nohup taskpolicy -b bash code/run_lowshare.sh >/dev/null 2>&1 &
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
JOBS=${JOBS:-6}
RES=$OUT/results_lowshare
LOG=$OUT/lowshare.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
throttle() { while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done; }
mkdir -p "$RES"
{
  echo "start $(date)"
  for spec in "0.0:none" "0.04:none" "0.04:sign_flip" "0.04:label_flip" "0.1:none" "0.1:sign_flip" "0.1:label_flip"; do
    m=${spec%%:*}; atk=${spec#*:}
    for s in 1 2 3 4 5; do
      log="$RES/m${m}_s${s}_${atk}.log"
      grep -q "^wrote" "$log" 2>/dev/null && continue   # resume
      throttle
      ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/m${m}_d0.5_s${s}_features.csv" --seed $s --attack $atk \
          --folds 3 --batch-size 128 --out "$RES" > "$log" 2>&1 && echo "done m$m s$s $atk" \
        || echo "FAILED m$m s$s $atk" ) &
    done
  done
  wait
  echo "end $(date)"
} > "$LOG" 2>&1
echo "done" >> "$LOG"
