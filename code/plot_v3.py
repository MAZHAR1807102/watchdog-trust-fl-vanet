#!/usr/bin/env python3
"""Aggregate experiments/results*/*summary.json into tables + a 2-panel plot.

Usage: python plot_v3.py [results_dir] [attack]
Writes <results_dir>/../v3_<attack>_f1.png and prints F1 (mean +- std) and
trust-filter beta/gamma per attacker ratio.
"""
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "../experiments/results")
ATTACK = sys.argv[2] if len(sys.argv) > 2 else "sign_flip"

BASELINES = ["FedAvg", "Krum", "TrimmedMean", "Median", "Krum+Bucket", "Median+Bucket"]
TRUST = ["Trust-FLAD", "Trust-FLAD+Krum", "Trust-FLAD+Median",
         "Trust-FLAD+Krum+Bucket", "Trust-FLAD+Median+Bucket"]
# reference categorical palette (light), fixed order per panel
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

def x_of(path):
    """x value of a run: poisoner fraction for decoupled runs, else attacker ratio."""
    d = re.search(r"_dec([\d.]+)_summary", path)
    if d:
        return float(d.group(1))
    m = re.search(r"m([\d.]+)_d[\d.]+", path)
    return float(m.group(1)) if m else None


f1 = defaultdict(lambda: defaultdict(list))
beta = defaultdict(list); gamma = defaultdict(list); kept = defaultdict(list)
for path in glob.glob(str(RES / f"*_{ATTACK}_*summary.json")):
    ratio = x_of(path)
    if ratio is None:
        continue
    d = json.load(open(path))["results"]
    for meth, r in d.items():
        f1[meth][ratio].append(r["f1"])
    if "Trust-FLAD" in d:
        beta[ratio].append(d["Trust-FLAD"]["beta"])
        gamma[ratio].append(d["Trust-FLAD"]["gamma"])
        kept[ratio].append(d["Trust-FLAD"]["accepted_mean"])

ratios = sorted({r for v in f1.values() for r in v})
DEC = bool(glob.glob(str(RES / f"*_{ATTACK}_*_dec*summary.json")))
if not ratios:
    sys.exit(f"no summaries for attack={ATTACK} in {RES}")
n_seeds = {r: len(f1["FedAvg"].get(r, [])) for r in ratios}
print(f"attack={ATTACK}  seeds per ratio: {n_seeds}\n")

print(f"{'method':<26}" + "".join(f"{int(r*100):>13}%" for r in ratios))
for meth in BASELINES + TRUST:
    if meth not in f1:
        continue
    row = f"{meth:<26}"
    for r in ratios:
        v = f1[meth].get(r, [])
        row += f"  {np.mean(v):.3f}±{np.std(v):.2f}  " if v else f"{'--':>14}"
    print(row)
print(f"\n{'trust filter':<26}" + "".join(f"{int(r*100):>13}%" for r in ratios))
for name, src in [("beta (attackers passed)", beta), ("gamma (honest rejected)", gamma),
                  ("clients kept / round", kept)]:
    print(f"{name:<26}" + "".join(f"  {np.mean(src[r]):>6.2f}±{np.std(src[r]):.2f} " for r in ratios))

extra = defaultdict(lambda: defaultdict(list))
for path in glob.glob(str(RES / f"*_{ATTACK}_*summary.json")):
    x = x_of(path)
    if x is not None:
        for meth, r in json.load(open(path))["results"].items():
            for key in ("pick_mal_rate", "bkt_dirty_frac"):
                if key in r:
                    extra[(meth, key)][x].append(r[key])
if extra:
    print(f"\n{'Krum / bucket diagnostics':<40}" + "".join(f"{int(r*100):>8}%" for r in ratios))
    for (meth, key), byr in sorted(extra.items()):
        label = f"{meth} {'poison picked' if key == 'pick_mal_rate' else 'dirty buckets'}"
        print(f"{label:<40}" + "".join(f"{np.mean(byr[r]):>9.2f}" if byr.get(r) else f"{'--':>9}"
                                       for r in ratios))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, group, title in [(axes[0], BASELINES, "Robust baselines (no trust)"),
                         (axes[1], TRUST, "Trust-FLAD variants")]:
    for i, meth in enumerate(group):
        if meth not in f1:
            continue
        xs = [r * 100 for r in ratios]
        ys = [np.mean(f1[meth].get(r, [np.nan])) for r in ratios]
        es = [np.std(f1[meth].get(r, [0])) for r in ratios]
        ax.errorbar(xs, ys, yerr=es, label=meth, color=COLORS[i], marker=MARKERS[i],
                    markersize=6, linewidth=2, capsize=2,
                    linestyle="--" if meth.endswith("+Bucket") else "-")
    ax.axvline(50, color="#9a9890", ls=":", lw=1)
    ax.set_title(title, fontsize=11, color="#0b0b0b")
    ax.set_xlabel("Model poisoners (% of vehicles)" if DEC else "Malicious vehicles (%)",
                  color="#52514e")
    ax.grid(alpha=0.25); ax.set_ylim(-0.03, 1.03)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
axes[0].set_ylabel("F1 (mean ± std over seeds)", color="#52514e")
fig.suptitle(f"v3 {RES.name}: {ATTACK}", fontsize=12)
fig.tight_layout()
suffix = "" if RES.name == "results" else f"_{RES.name}"
out = RES.parent / f"v3_{ATTACK}_f1{suffix}.png"
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
