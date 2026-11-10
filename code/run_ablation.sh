#!/bin/bash
# Ablation of the trust filter design for Paper A (conference, papers_extension/A_conference):
# threshold (gmm vs otsu vs none), trust mode (window vs static), trust window W (5, 10, 20 s).
# Same data and FL settings as the overnight ring run (dropRate 0.5, sign_flip, 3 folds, batch 128).
# Each variant gets its own results folder (W is not part of main_fl_v3's output tag).
#   JOBS=6 nohup taskpolicy -b bash code/run_ablation.sh >/dev/null 2>&1 &
#   (taskpolicy -b = background QoS: macOS keeps it on the 6 cooler efficiency cores; resumable)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
JOBS=${JOBS:-4}   # 9 parallel jobs heated the Mac to 97 C (2026-11-09)
FL="--folds 3 --batch-size 128 --attack sign_flip --methods Trust-FLAD,Trust-FLAD+Krum,Trust-FLAD+Median"
LOG=$OUT/ablation.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
caffeinate -ims -w $$ & disown
throttle() { while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done; }
{
  echo "start $(date)"
  for spec in "otsu:--threshold otsu" "none:--threshold none" "static:--trust-mode static" \
              "W5:--trust-window 5" "W20:--trust-window 20"; do
    name=${spec%%:*}; args=${spec#*:}
    mkdir -p "$OUT/results_abl_$name"
    for m in 0.2 0.3 0.4 0.5 0.6 0.7; do for s in 1 2 3 4 5; do
      grep -q "^wrote" "$OUT/results_abl_$name/m${m}_s${s}.log" 2>/dev/null && continue   # resume
      throttle
      ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/m${m}_d0.5_s${s}_features.csv" --seed $s $FL $args \
          --out "$OUT/results_abl_$name" > "$OUT/results_abl_$name/m${m}_s${s}.log" 2>&1 \
        || echo "FAILED $name m$m s$s" ) &
    done; done
  done
  wait
  echo "end $(date)"
} > "$LOG" 2>&1
echo "done" >> "$LOG"
