#!/bin/bash
# Defences against on-off grayholes, for Paper A. Data: data/m*_d0.9_on10off20_s* (drop 0.9 for 10 s, honest 20 s).
#   memW30 / memW60: trust memory 60 s + 30 / 60 s trust warm-up before the first FL round
#   cum / cumW30:    cumulative trust (window 100 s = whole past) without / with a 30 s warm-up
# Resumable, background QoS on the efficiency cores:
#   JOBS=6 nohup taskpolicy -b bash code/run_onoff_fix.sh >/dev/null 2>&1 &
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
JOBS=${JOBS:-6}
FL="--folds 3 --batch-size 128 --attack sign_flip --methods Trust-FLAD,Trust-FLAD+Krum,Trust-FLAD+Median"
LOG=$OUT/onoff_fix.log
throttle() { while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done; }
{
  echo "start $(date)"
  for spec in "memW30:--trust-memory 60 --warmup 30" "memW60:--trust-memory 60 --warmup 60" \
              "cum:--trust-window 100" "cumW30:--trust-window 100 --warmup 30"; do
    name=${spec%%:*}; args=${spec#*:}
    res=$OUT/results_onoff_$name
    mkdir -p "$res"
    for m in 0.5 0.6 0.7; do for s in 1 2 3 4 5; do
      grep -q "^wrote" "$res/m${m}_s${s}.log" 2>/dev/null && continue   # resume
      throttle
      ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/m${m}_d0.9_on10off20_s${s}_features.csv" --seed $s $FL $args \
          --out "$res" > "$res/m${m}_s${s}.log" 2>&1 || echo "FAILED $name m$m s$s" ) &
    done; done
  done
  wait
  echo "end $(date)"
} > "$LOG" 2>&1
echo "done" >> "$LOG"
