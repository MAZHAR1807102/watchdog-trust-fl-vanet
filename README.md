# Watchdog Trust for Byzantine-Robust Federated Learning in VANETs

Code, data and results for the paper *"Watchdog Trust for Byzantine-Robust Federated Learning in VANETs:
A Measurable Breakdown Condition, and Where It Fails"*.

A trust filter scores each vehicle by watchdog observations of its packet forwarding and discards low-trust
model updates before robust aggregation. With attacker pass rate β and honest rejection rate γ, robust
aggregation holds while β/(1−γ) < (1−ρ)/ρ. Grayhole attackers in ns-3 really drop transit packets; clients are
non-IID; evaluation is vehicle-disjoint 3-fold cross-validation.

## Layout

| Path | Contents |
|---|---|
| `code/vanet-trust-fl-dataset_v3.cc` | ns-3.42 simulator: 50 vehicles, 2 km ring road, 802.11a, AODV, grayhole attackers, watchdog features |
| `code/main_fl_v3.py` | FL pipeline: trust filter (GMM threshold, windowed trust, memory, warm-up) and 11 aggregation rules |
| `code/run_*.sh` | Experiment scripts (see below); `plot_v3.py` prints F1 / β / γ tables |
| `experiments/data/` | 135 simulated datasets (27 configurations x 5 seeds) (`m<share>_d<drop>[_on10off20]_s<seed>`): per-second features, neighbour log, summary |
| `experiments/results*/` | Per-round CSVs and `*_summary.json` for every FL run; `report*.txt` are the printed tables |
| `paper/` | LaTeX source (`main.tex`, `refs.bib`), figures and the scripts that generate the table and figures |

## Reproduce the paper's numbers from the stored results

Python 3.13 with the packages in `requirements.txt`.

```bash
cd paper
python make_tables.py          # main table (add --write to update main.tex)
python make_figs.py            # figs/fig_phase.pdf, figs/fig_trust.pdf; prints the 56-setting phase check
python perround_check.py       # per-round analysis of 1,260 TF+Median runs -> perround.csv
latexmk -pdf main.tex
```

## Re-run the experiments

1. Build the simulator in ns-3.42:
   ```bash
   cp code/vanet-trust-fl-dataset_v3.cc $NS3/scratch/
   cd $NS3 && ./ns3 build scratch/vanet-trust-fl-dataset_v3
   ```
2. Run the scripts (set `NS3=` and `PY=` if they differ from the defaults at the top of `run_sweep_v3.sh`).
   All are resumable.

| Script | Paper part | Output |
|---|---|---|
| `run_overnight.sh` | sign-flip, ALIE z=1, no-attack ceiling (ρ 0.2–0.7) | `results/` |
| `run_labelflip.sh` | label-flip | `results_labelflip/` |
| `run_lowshare.sh` | 0, 2 and 5 attackers of 50 | `results_lowshare/` |
| `run_harder.sh` | weak (p_d 0.2/0.3), on-off, decoupled attackers; ALIE z=2/3 | `results_d0.2`, `results_d0.3`, `results_onoff`, `results_decoupled`, `results_alie_z2/3` |
| `run_ablation.sh` | Otsu / no filter / static trust / window 5 and 20 s | `results_abl_*` |
| `run_onoff_memory.sh` | trust memory 30/60/90 s | `results_onoff_mem*` |
| `run_onoff_fix.sh` | memory + warm-up, cumulative trust | `results_onoff_memW*`, `results_onoff_cum*` |

One simulation takes about 8 s; one FL run (11 rules, 3 folds, 20 rounds) a few minutes on a laptop CPU.

## Citation

Paper under review; citation will be added on acceptance.
