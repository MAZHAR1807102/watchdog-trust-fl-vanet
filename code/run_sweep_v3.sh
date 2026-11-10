#!/bin/bash
# Generate v3 datasets and run the FL pipeline on each.
# Usage: bash run_sweep_v3.sh "RATIOS" "SEEDS" DROP ATTACK
#   e.g. bash run_sweep_v3.sh "0.2 0.3 0.4 0.5 0.6 0.7" "1 2 3 4 5" 0.5 sign_flip
# Env: JOBS (parallel FL runs, default ncpu/2), OUT, NS3, PY,
#      RESULTS (results subdir, default results), METHODS (comma list, default all),
#      EXTRA (extra args for main_fl_v3.py), SIMEXTRA (extra args for the ns-3 sim),
#      TAGX (dataset-name suffix, e.g. _on10off20, so variants do not collide)
RATIOS=${1:-"0.2 0.3 0.4 0.5 0.6 0.7"}
SEEDS=${2:-"1 2 3 4 5"}
DROP=${3:-0.5}
ATTACK=${4:-sign_flip}
NS3=${NS3:-$HOME/NS3/ns-allinone-3.42/ns-3.42}
PY=${PY:-$HOME/miniforge3/bin/python3}
JOBS=${JOBS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/../experiments}
RESULTS=${RESULTS:-results}
METHODS=${METHODS:-all}
mkdir -p "$OUT/data" "$OUT/$RESULTS"
export DYLD_LIBRARY_PATH=$NS3/build/lib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BIN=$NS3/build/scratch/ns3.42-vanet-trust-fl-dataset_v3-default

for m in $RATIOS; do for s in $SEEDS; do
  tag="m${m}_d${DROP}${TAGX}_s${s}"
  [ -f "$OUT/data/${tag}_features.csv" ] || "$BIN" --malicious=$m --dropRate=$DROP --seed=$s $SIMEXTRA --output="$OUT/data/$tag" > /dev/null
done; done
echo "datasets ready"

for m in $RATIOS; do for s in $SEEDS; do
  tag="m${m}_d${DROP}${TAGX}_s${s}"
  while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/${tag}_features.csv" --attack $ATTACK --seed $s \
      --methods "$METHODS" $EXTRA --out "$OUT/$RESULTS" > "$OUT/$RESULTS/${tag}_${ATTACK}.log" 2>&1 \
    && echo "done $tag" || echo "FAILED $tag" ) &
done; done
wait
echo "sweep finished"
