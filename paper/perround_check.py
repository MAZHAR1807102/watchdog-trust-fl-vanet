"""Per-round check of the breakdown condition for TF+Median.

The pooled rho_surv (Prop. 1, from beta and gamma averaged over the run) ignores how the attacker share among
accepted updates varies from round to round. Here, for every TF+Median run (setting x seed x fold), we read the
per-round log (acc_hon, acc_mal) and compute rho_r = acc_mal / (acc_mal + acc_hon), then ask which statistic
predicts whether the run collapses (final-round F1 < 0.5 x no-attack F1).

Decoupled runs are skipped: the log counts grayholes as 'mal', but the poisoners are other vehicles.
Usage: ~/miniforge3/bin/python3 perround_check.py   (prints tables; writes perround.csv)
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / '../experiments'

FOLDERS = {  # folder -> (setting label, family for the no-attack ceiling)
    'results': ('grayhole 0.5', 'd0.5'), 'results_alie_z2': ('grayhole 0.5', 'd0.5'),
    'results_alie_z3': ('grayhole 0.5', 'd0.5'), 'results_d0.3': ('grayhole 0.3', 'd0.3'),
    'results_d0.2': ('grayhole 0.2', 'd0.2'), 'results_onoff': ('on-off', 'onoff'),
    'results_onoff_mem30': ('on-off mem30', 'onoff'), 'results_onoff_mem60': ('on-off mem60', 'onoff'),
    'results_onoff_mem90': ('on-off mem90', 'onoff'), 'results_abl_otsu': ('abl otsu', 'd0.5'),
    'results_abl_W5': ('abl W5', 'd0.5'), 'results_abl_W20': ('abl W20', 'd0.5'),
    'results_abl_static': ('abl static', 'd0.5')}
METHOD = 'Trust-FLAD+Median'
LAST = 5  # rounds at the end that decide the final model


def ceilings():
    rows = []
    for folder, (_, fam) in FOLDERS.items():
        for f in glob.glob(str(D / folder / '*_summary.json')):
            j = json.load(open(f))
            if j['config']['attack'] == 'none' and 'FedAvg' in j['results']:
                rho = float(re.search(r'/m([\d.]+)_', j['csv']).group(1))
                rows.append((fam, rho, j['results']['FedAvg']['f1']))
    return pd.DataFrame(rows, columns=['family', 'rho', 'f1']).groupby(['family', 'rho']).f1.mean()


def load():
    rows = []
    for folder, (label, fam) in FOLDERS.items():
        for f in glob.glob(str(D / folder / '*_summary.json')):
            j = json.load(open(f))
            c = j['config']
            if c['attack'] == 'none' or c.get('poison_set') == 'decoupled':
                continue
            atk = c['attack'] + (f"_z{c['alie_z']:g}" if c['attack'] == 'alie' else '')
            rho = float(re.search(r'/m([\d.]+)_', j['csv']).group(1))
            stem = f[:-len('_summary.json')]
            for csv in sorted(glob.glob(f'{glob.escape(stem)}_{METHOD}_f*.csv')):
                p = pd.read_csv(csv)
                kept = p.acc_hon + p.acc_mal
                rr = (p.acc_mal / kept.where(kept > 0)).fillna(0).values
                beta = p.acc_mal.sum() / (p.acc_mal + p.rej_mal).sum()
                gamma = p.rej_hon.sum() / (p.acc_hon + p.rej_hon).sum()
                rows.append(dict(
                    folder=folder, setting=label, family=fam, attack=atk, rho=rho, seed=c['seed'],
                    fold=int(csv[-5]), f1_final=p.f1.iloc[-1], beta=beta, gamma=gamma,
                    pooled=rho * beta / (rho * beta + (1 - rho) * (1 - gamma)),
                    mean_r=rr.mean(), max_r=rr.max(), first_r=rr[0], last_r=rr[-LAST:].mean(),
                    bad_frac=(rr >= 0.5).mean(), bad_last=(rr[-LAST:] >= 0.5).mean(),
                    unknown_r1=p.unknown.iloc[0]))
    return pd.DataFrame(rows)


def accuracy(df, pred):
    return (pred == df.collapsed).mean(), int(((pred == 0) & (df.collapsed == 1)).sum()), \
        int(((pred == 1) & (df.collapsed == 0)).sum())


ceil = ceilings()
df = load()
df['rel'] = [r.f1_final / ceil.loc[(r.family, r.rho)] for r in df.itertuples()]
df['collapsed'] = (df.rel < 0.5).astype(int)
df.to_csv(HERE / 'perround.csv', index=False)

print(f'{len(df)} TF+Median runs (setting x seed x fold), {df.collapsed.sum()} collapsed (final F1 < 0.5 x ceiling)\n')
print('Predictor of collapse (run level)            accuracy  missed  false alarms')
for name, pred in [
        ('pooled rho_surv >= 0.5 (Prop. 1)', df.pooled >= 0.5),
        ('mean per-round rho_r >= 0.5', df.mean_r >= 0.5),
        ('any round rho_r >= 0.5', df.max_r >= 0.5),
        ('first round rho_r >= 0.5', df.first_r >= 0.5),
        (f'mean rho_r over last {LAST} rounds >= 0.5', df.last_r >= 0.5),
        (f'any of last {LAST} rounds rho_r >= 0.5', df.bad_last > 0),
        (f'majority of last {LAST} rounds rho_r >= 0.5', df.bad_last > 0.5)]:
    a, miss, fa = accuracy(df, pred.astype(int))
    print(f'  {name:42s}  {a:6.3f}  {miss:6d}  {fa:6d}')

# where pooled and per-round disagree
dis = df[(df.pooled < 0.5) & (df.collapsed == 1)]
print(f'\nCollapsed although pooled rho_surv < 0.5: {len(dis)} runs')
print(dis.groupby(['setting', 'attack', 'rho']).agg(
    runs=('seed', 'size'), pooled=('pooled', 'mean'), mean_r=('mean_r', 'mean'), first_r=('first_r', 'mean'),
    bad_frac=('bad_frac', 'mean'), last_r=('last_r', 'mean'), rel=('rel', 'mean')).round(2).to_string())

# per-round trajectory in the problem case vs a surviving neighbour
print('\nOn-off mem60, rho 0.7: per-run statistics')
print(df[(df.folder == 'results_onoff_mem60') & (df.rho == 0.7)][
    ['seed', 'fold', 'pooled', 'first_r', 'mean_r', 'bad_frac', 'last_r', 'unknown_r1', 'rel']].round(2).to_string(index=False))
