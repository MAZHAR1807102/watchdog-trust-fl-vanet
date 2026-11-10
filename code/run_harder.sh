#!/bin/bash
# Harder-attacker experiments on the ring setup (3-fold, batch 128, 5 seeds, 11 methods).
#   decoupled  poisoners are honest-forwarding vehicles (data m0.2_d0.5), fraction 0.2-0.7
#   onoff      attackers drop 90% for 10 s, forward honestly for 20 s (mean drop 0.3)
#   d0.3/d0.2  constant grayhole drop rate 0.3 / 0.2
#   alie_z2/z3 ALIE with a larger shift on the existing d0.5 data
# Each gets its own results_<name>/ folder; on-off and constant-drop runs also get a
# no-attack control (FedAvg, Median, Trust-FLAD+Median) in the same folder.
# Expected time: about 4.5 h. Keep the Mac plugged in with the lid open.
#
# Usage:    nohup bash code/run_harder.sh >/dev/null 2>&1 &
# Progress: tail -f experiments/harder.log
# Results:  experiments/report_harder.txt

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../experiments
PY=${PY:-$HOME/miniforge3/bin/python3}
R="0.2 0.3 0.4 0.5 0.6 0.7"
SD="1 2 3 4 5"
JOBS=${JOBS:-9}
FL="--folds 3 --batch-size 128"
LOG=$OUT/harder.log
CONTROL="FedAvg,Median,Trust-FLAD+Median"
export JOBS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
caffeinate -ims -w $$ & disown   # keep the Mac awake; disown so `wait` and job counts ignore it

sweep() {  # name drop attack [extra FL args] -- uses SIMEXTRA/TAGX/METHODS from env
  RESULTS=results_$1 EXTRA="$FL $4" bash "$HERE/run_sweep_v3.sh" "$R" "$SD" $2 $3 \
    | grep -E "FAILED|finished"
}

{
  echo "start $(date)"

  echo "== decoupled  $(date +%H:%M)"
  mkdir -p "$OUT/results_decoupled"
  for p in $R; do for s in $SD; do
    while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    ( "$PY" "$HERE/main_fl_v3.py" --csv "$OUT/data/m0.2_d0.5_s${s}_features.csv" --seed $s \
        --attack sign_flip --poison-set decoupled --poison-frac $p $FL \
        --out "$OUT/results_decoupled" > "$OUT/results_decoupled/p${p}_s${s}.log" 2>&1 \
      || echo "FAILED decoupled p$p s$s" ) &
  done; done
  wait
  echo "finished"

  echo "== onoff  $(date +%H:%M)"
  SIMEXTRA="--onTime=10 --offTime=20" TAGX=_on10off20 sweep onoff 0.9 sign_flip
  SIMEXTRA="--onTime=10 --offTime=20" TAGX=_on10off20 METHODS=$CONTROL sweep onoff 0.9 none

  for d in 0.3 0.2; do
    echo "== d$d  $(date +%H:%M)"
    sweep d$d $d sign_flip
    METHODS=$CONTROL sweep d$d $d none
  done

  for z in 2 3; do
    echo "== alie z=$z  $(date +%H:%M)"
    sweep alie_z$z 0.5 alie "--alie-z $z"
  done

  echo "end $(date)"
} > "$LOG" 2>&1

{
  for spec in decoupled:sign_flip onoff:sign_flip onoff:none d0.3:sign_flip d0.3:none \
              d0.2:sign_flip d0.2:none alie_z2:alie alie_z3:alie; do
    name=${spec%%:*}; atk=${spec##*:}
    echo "################ $name / $atk"
    "$PY" "$HERE/plot_v3.py" "$OUT/results_$name" $atk 2>/dev/null
    echo
  done
  echo "App PDR by dataset family (min, max):"
  for fam in d0.3_s d0.2_s d0.9_on10off20_s; do
    echo "  $fam: $(grep -h 'App PDR' "$OUT"/data/m*_${fam}*_summary.txt | awk '{print $3}' | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
  done
} > "$OUT/report_harder.txt"
echo "done: $OUT/report_harder.txt" >> "$LOG"
