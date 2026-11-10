#!/bin/bash
# Label-flip sweep for Paper A (main table, right half): p_d 0.5, ratios 0.2-0.7, seeds 1-5, 11 rules,
# 3 folds, batch 128. Same settings as run_overnight.sh; data must exist. Results -> experiments/results_labelflip/.
#   JOBS=6 nohup taskpolicy -b bash code/run_labelflip.sh >/dev/null 2>&1 &
HERE=$(cd "$(dirname "$0")" && pwd)
export JOBS=${JOBS:-6} RESULTS=results_labelflip EXTRA="--folds 3 --batch-size 128"
bash "$HERE/run_sweep_v3.sh" "0.2 0.3 0.4 0.5 0.6 0.7" "1 2 3 4 5" 0.5 label_flip \
  > "$HERE/../experiments/labelflip.log" 2>&1
