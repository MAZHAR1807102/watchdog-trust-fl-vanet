#!/usr/bin/env python3
"""Figures for the Trust-FLAD (v3) paper, from ../experiments/ (run from paper/).

  python make_figs.py   -> figs/fig_f1.pdf, figs/fig_phase.pdf, and the phase table on stdout
"""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
D = HERE / '../experiments'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'figure.dpi': 200})
(HERE / 'figs').mkdir(exist_ok=True)

FAMILY = {  # results folder -> (setting label, family used for the no-attack ceiling)
    'results': ('grayhole 0.5', 'd0.5'), 'results_alie_z2': ('grayhole 0.5', 'd0.5'),
    'results_alie_z3': ('grayhole 0.5', 'd0.5'), 'results_d0.3': ('grayhole 0.3', 'd0.3'),
    'results_d0.2': ('grayhole 0.2', 'd0.2'), 'results_onoff': ('on-off', 'onoff'),
    'results_decoupled': ('decoupled', 'd0.5'), 'results_labelflip': ('grayhole 0.5', 'd0.5'),
    'results_lowshare': ('grayhole 0.5', 'd0.5')}


def load():
    rows = []
    for folder, (label, fam) in FAMILY.items():
        for f in glob.glob(str(D / folder / '*_summary.json')):
            j = json.load(open(f))
            c = j['config']
            rho = c['poison_frac'] if c.get('poison_set') == 'decoupled' else \
                float(re.search(r'/m([\d.]+)_', j['csv']).group(1))
            atk = c['attack'] + (f"_z{c['alie_z']:g}" if c['attack'] == 'alie' else '')
            for meth, r in j['results'].items():
                rows.append(dict(folder=folder, setting=label, family=fam, attack=atk, rho=rho,
                                 seed=c['seed'], method=meth, f1=r['f1'], beta=r.get('beta'), gamma=r.get('gamma')))
    return pd.DataFrame(rows)


r = load()
ceil = r[(r.attack == 'none') & (r.method == 'FedAvg')].groupby(['family', 'rho']).f1.mean()

# --- F1 vs rho: sign-flip and ALIE z=3 on the main (drop 0.5) setting ---------------------------
M = [('FedAvg', '#999999', ':'), ('Krum', '#e67e22', '-'), ('Median', '#1baf7a', '-'),
     ('Trust-FLAD', '#8e44ad', '-'), ('Trust-FLAD+Krum', '#2a78d6', '-'), ('Trust-FLAD+Median', '#c0392b', '--')]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, (folder, atk, title) in zip(axes, [('results', 'sign_flip', 'sign-flip ($\\lambda = 5$)'),
                                           ('results_alie_z3', 'alie_z3', 'ALIE ($z = 3$)')]):
    g0 = r[(r.folder == folder) & (r.attack == atk)]
    for meth, col, ls in M:
        s = g0[g0.method == meth].groupby('rho').f1
        ax.errorbar(s.mean().index, s.mean().values, yerr=1.96 * s.sem().values, color=col, ls=ls, marker='o',
                    ms=3, lw=1.3, capsize=2, label=meth.replace('Trust-FLAD', 'TF'))
    c = ceil.loc['d0.5']
    ax.plot(c.index, c.values, 'k--', lw=0.9, label='no attack')
    ax.set_title(title); ax.set_xlabel('attacker share $\\rho$'); ax.grid(alpha=.3)
axes[0].set_ylabel('F1 (pooled 3-fold)')
axes[0].legend(loc='lower left', ncol=2)
fig.tight_layout()
fig.savefig(HERE / 'figs/fig_f1.pdf')

# --- phase transition: TF+Median relative F1 vs predicted rho_surv ---------------------------------
tf = r[r.method == 'Trust-FLAD'].groupby(['folder', 'setting', 'family', 'attack', 'rho'])[['beta', 'gamma']].mean()
tm = r[r.method == 'Trust-FLAD+Median'].groupby(['folder', 'setting', 'family', 'attack', 'rho']).f1.mean()
ph = tf.join(tm).reset_index()
ph = ph[ph.attack != 'none']
ph['surv'] = ph.rho * ph.beta / (ph.rho * ph.beta + (1 - ph.rho) * (1 - ph.gamma))
ph['rel'] = [row.f1 / ceil.loc[(row.family, row.rho)] for row in ph.itertuples()]
# exclude settings where the task itself is not learnable (no-attack F1 < 0.5, e.g. 2 attackers of 50)
ph['ceil'] = [ceil.loc[(row.family, row.rho)] for row in ph.itertuples()]
print(f"excluded (no-attack F1 < 0.5): {(ph.ceil < 0.5).sum()} settings")
ph = ph[ph.ceil >= 0.5]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for (setting, atk), g in ph.groupby(['setting', 'attack']):
    ax.scatter(g.surv, g.rel, s=14, label=f'{setting}, {atk}')
ax.axvline(0.5, color='k', lw=0.8, ls='--')
ax.set_xlabel('predicted attacker share after filtering $\\rho_{surv}$')
ax.set_ylabel('TF+Median F1 / no-attack F1')
ax.grid(alpha=.3)
ax.legend(fontsize=5.5, loc='lower left')
fig.tight_layout()
fig.savefig(HERE / 'figs/fig_phase.pdf')

ok = ((ph.surv < 0.5) == (ph.rel > 0.5))
print(ph[['setting', 'attack', 'rho', 'beta', 'gamma', 'surv', 'f1', 'rel']].round(2).to_string(index=False))
print(f"\nrho_surv < 0.5 predicts relative F1 > 0.5 in {ok.sum()}/{len(ph)} settings")
print('wrote figs/fig_f1.pdf, figs/fig_phase.pdf')

# --- trust distributions: honest vs attacker windowed trust (rho = 0.3) --------------------------
import sys
sys.path.insert(0, str(HERE / '../code'))
import main_fl_v3 as v3  # noqa: E402

cfg = dict(v3.DEFAULTS)
fams = [('m0.3_d0.5_s{}', 'grayhole $p_d$ = 0.5'), ('m0.3_d0.2_s{}', 'grayhole $p_d$ = 0.2'),
        ('m0.3_d0.9_on10off20_s{}', 'on-off')]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
for ax, (pat, title) in zip(axes, fams):
    th, tm = [], []
    for s in range(1, 6):
        f = D / 'data' / (pat.format(s) + '_features.csv')
        if not f.exists():
            continue
        df, _ = v3.sanitize(pd.read_csv(f))
        lab = df.groupby('vehicle_id').label.first()
        tr = v3.TrustTracker(df, cfg)
        for t in np.arange(15, 96, 5):
            tt = tr.trust_at(t)
            th += [x for v, x in tt.items() if not lab[v] and not np.isnan(x)]
            tm += [x for v, x in tt.items() if lab[v] and not np.isnan(x)]
    bins = np.linspace(0, 1, 26)
    ax.hist(th, bins, density=True, alpha=.6, color='#2a78d6', label='honest')
    ax.hist(tm, bins, density=True, alpha=.6, color='#c0392b', label='attacker')
    ax.set_title(title); ax.set_xlabel('windowed trust $T_v$'); ax.grid(alpha=.3)
axes[0].set_ylabel('density'); axes[0].legend(loc='upper left')
fig.tight_layout()
fig.savefig(HERE / 'figs/fig_trust.pdf')
print('wrote figs/fig_trust.pdf')
