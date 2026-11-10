#!/usr/bin/env python3
"""
Trust-FLAD -- training pipeline (v3)
====================================

Changes relative to main_fl_v2.py:

  [A2] THRESHOLD. Otsu always splits a distribution in two, even when it is
       one honest mode plus a few attackers; on the v2 sweep it cut inside
       the honest mode and rejected 73-97% of honest vehicles. v3 adds
         --threshold gmm   2-component Gaussian mixture on trust (label-free).
                           If BIC prefers ONE component there is no evidence
                           of a separate attacker population -> accept all.
                           Otherwise cut where P(honest component) = 0.5.
         --threshold none  no filtering, trust is only a weight.
         --threshold otsu  v2 behaviour, kept for comparison.
         --threshold 0.6   fixed operator threshold.
       Every round logs accepted/rejected honest and malicious clients, and
       the summary reports beta (attacker pass rate) and gamma (honest
       rejection rate).

  [A3] WINDOWED TRUST. v2 computed one trust value per vehicle from the whole
       run, including the future. v3 maps FL round r to simulation time t_r
       and computes trust from counter deltas in (t_r - W, t_r]:
           forward_ratio = d(forwarded) / d(handled)
           route_part    = d(relayed control) / d(seen control)
       Too few observations in the window -> carry the last known value;
       never observed -> cold start (--cold-start accept|reject).
       --trust-mode static gives the full-run value (v2-like, uses the future).

  [A4] NON-IID CLIENTS. --clients observed gives each vehicle the feature
       rows of the neighbours it was within radio range of, at the times it
       was in range (from <prefix>_neighbors.csv, written by the v3 ns-3
       script). Client datasets now differ in size and class mix. Averaging
       rules weight by sample count. --clients iid keeps the v2 random shards.

  [B]  BUCKETING (Karimireddy et al., ICLR 2022): *+Bucket rules average
       shuffled updates in buckets of --bucket-size before Krum/Median, to
       stop non-IID honest clients from looking like outliers.

  [B]  ALIE attack (--attack alie, --alie-z) and Krum selection logging:
       pick_mal = attackers inside the update/bucket Krum chose,
       bkt_dirty = buckets containing at least one attacker (reporting only).

  [D]  DECOUPLED POISONERS (--poison-set decoupled --poison-frac p): model
       poisoners are honest-forwarding vehicles, so behavioural trust cannot
       see them. Tests the limit of the method.

  [C]  K-FOLD over vehicles (--folds k): every vehicle is tested once per
       seed; metrics are computed on the pooled test predictions.

  Also: FedAvg weights by sample count; clients with no data do not
  participate; v2's no-op assert removed.

Feature CSVs from the v2 simulator (no fwd_requested column) still load:
trust then uses packets_forwarded / (forwarded + dropped).

Usage:
    python main_fl_v3.py --csv run_features.csv --attack sign_flip
    python main_fl_v3.py --csv run_features.csv --threshold otsu --trust-mode static --clients iid
"""

import argparse, copy, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------- config ---

# v3 simulator schema: watchdog-based, no ground-truth bookkeeping.
FEATURES_V3 = [
    'packets_sent', 'packets_received', 'bytes_sent', 'bytes_received',
    'data_pkts_sent', 'data_pkts_received', 'rreq_received', 'rreq_forwarded',
    'rrep_sent', 'rrep_received', 'fwd_requested', 'fwd_overheard',
    'forward_ratio', 'route_participation', 'avg_delay_ms', 'neighbor_count',
]
# v2 simulator schema.
FEATURES_V2 = [
    'packets_sent', 'packets_received', 'packets_forwarded', 'packets_dropped',
    'bytes_sent', 'bytes_received', 'data_pkts_sent', 'data_pkts_received',
    'data_pkts_forwarded', 'data_pkts_dropped', 'rreq_received', 'rreq_forwarded',
    'rrep_received', 'rrep_forwarded', 'forward_ratio', 'drop_ratio',
    'data_forward_ratio', 'route_participation', 'avg_delay_ms', 'neighbor_count',
]

# Trust weights -- MUST match alpha/beta reported in the paper.
ALPHA, BETA = 0.7, 0.3

DEFAULTS = dict(
    rounds_plain=20, local_epochs=3, batch_size=32, lr=0.01,
    trust_threshold='gmm', test_vehicle_ratio=0.2, seed=42, krum_f=None,
    attack='sign_flip', attack_scale=5.0, attack_sigma=1.0, alie_z=1.0, uniform_after_filter=False, bucket_size=2,
    trust_mode='window', trust_window=10.0, min_obs=5, cold_start='accept',
    clients='auto', poison_set='label', poison_frac=0.3, uploads=None,
    trust_memory=0.0, trust_norm='none', norm_radius=300.0, warmup=0.0,
)


# ----------------------------------------------------------------- model ---

class BlackholeDetector(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class VanetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ------------------------------------------------------------- sanitising ---

def sanitize(df):
    """Repair known ns-3 export defects (v2 uint64 underflow). Returns (df, report)."""
    rep = {}
    for c in df.select_dtypes('number').columns:
        bad = df[c] > 1e12
        if bad.any():
            rep[f'underflow_{c}'] = int(bad.sum())
            df.loc[bad, c] = 0
    for c in [c for c in df.columns if c.endswith('_ratio') or c == 'route_participation']:
        oob = (df[c] < 0) | (df[c] > 1)
        if oob.any():
            rep[f'out_of_range_{c}'] = int(oob.sum())
            df[c] = df[c].clip(0, 1)
    return df, rep


def schema(df):
    return 'v3' if 'fwd_requested' in df.columns else 'v2'


# ---------------------------------------------------- trust (label-free) ---
# Nothing in this section may read df['label'].

def _trust_counters(df):
    """Cumulative (numerator, denominator) series per trust component, indexed
    by (timestamp, vehicle_id)."""
    d = df.set_index(['timestamp', 'vehicle_id']).sort_index()
    if schema(df) == 'v3':
        fr = (d['fwd_overheard'], d['fwd_requested'])
        rp = (d['rreq_forwarded'], d['rreq_received'])
    else:
        fr = (d['packets_forwarded'], d['packets_forwarded'] + d['packets_dropped'])
        rp = (d['rreq_forwarded'] + d['rrep_forwarded'], d['rreq_received'] + d['rrep_received'])
    return {'fr': fr, 'rp': rp}


class TrustTracker:
    """Per-round behavioural trust.

    window mode: ratios of counter deltas over (t - W, t]. A component with
    fewer than min_obs events in the window keeps its last known value; a
    vehicle with no known forward ratio yet is 'unknown' (NaN).
    static mode: ratio over the whole run (uses future data; v2-like).
    """

    def __init__(self, df, cfg):
        self.cfg = cfg
        self.times = np.sort(df.timestamp.unique())
        self.vehicles = sorted(df.vehicle_id.unique())
        self.pos = None
        if 'pos_x' in df.columns:   # broadcast in beacons, so observable
            d = df.set_index(['timestamp', 'vehicle_id']).sort_index()
            self.pos = (d.pos_x.unstack('vehicle_id').reindex(columns=self.vehicles).ffill(),
                        d.pos_y.unstack('vehicle_id').reindex(columns=self.vehicles).ffill())
        self.counters = {}
        for name, (num, den) in _trust_counters(df).items():
            self.counters[name] = (num.unstack('vehicle_id').reindex(columns=self.vehicles).ffill().fillna(0),
                                   den.unstack('vehicle_id').reindex(columns=self.vehicles).ffill().fillna(0))
        self.reset()

    def reset(self):
        self.last = {name: pd.Series(np.nan, index=self.vehicles) for name in self.counters}

    def _at(self, frame, t):
        idx = np.searchsorted(self.times, t, side='right') - 1
        if idx < 0:
            return pd.Series(0.0, index=self.vehicles)
        return frame.loc[self.times[idx]]

    def round_time(self, r, rounds):
        # --warmup: trust observes the network for `warmup` s before round 1
        t0, t1 = self.times[0] + self.cfg.get('warmup', 0.0), self.times[-1]
        return t0 + (t1 - t0) * r / rounds

    def _window_trust(self, t):
        """Trust over (t - W, t] only, NaN where too few observations (no state)."""
        W, out = self.cfg['trust_window'], {}
        for name, (num, den) in self.counters.items():
            dn = self._at(num, t) - self._at(num, t - W)
            dd = self._at(den, t) - self._at(den, t - W)
            out[name] = (dn / dd.where(dd >= self.cfg['min_obs'])).clip(0, 1)
        return ALPHA * out['fr'] + BETA * out['rp'].fillna(out['fr'])

    def _local_norm(self, trust, t):
        """Neighbourhood-relative trust: divide by what good forwarders achieve nearby
        (75th percentile of known trust within norm_radius at time t, incl. self).
        A noisy stretch (tunnel, forest) lowers everyone there, so it cancels out."""
        if self.pos is None:
            return trust
        px, py = self._at(self.pos[0], t), self._at(self.pos[1], t)
        tv = pd.Series(trust)
        out = {}
        for v in self.vehicles:
            if np.isnan(tv[v]):
                out[v] = np.nan
                continue
            near = np.hypot(px - px[v], py - py[v]) <= self.cfg['norm_radius']
            ref = tv[near[near].index].dropna()
            q = np.quantile(ref, 0.75) if len(ref) >= 3 else np.nan
            out[v] = min(1.0, tv[v] / q) if q and not np.isnan(q) and q > 0 else tv[v]
        return out

    def trust_at(self, t):
        trust = self._trust_at(t)
        M = self.cfg.get('trust_memory', 0.0)
        if M > 0 and self.cfg['trust_mode'] == 'window':
            # slow recovery: a vehicle keeps the WORST windowed trust seen in the last M s,
            # so an on-off attacker cannot rebuild trust in its quiet phase
            past = [self._window_trust(t - k * self.cfg['trust_window'])
                    for k in range(1, int(M // self.cfg['trust_window']) + 1)]
            worst = pd.concat(past, axis=1).min(axis=1, skipna=True)
            trust = {v: (min(x, worst[v]) if not (np.isnan(x) or np.isnan(worst[v])) else x)
                     for v, x in trust.items()}
        if self.cfg.get('trust_norm', 'none') == 'local':
            trust = self._local_norm(trust, t)
        return trust

    def _trust_at(self, t):
        if self.cfg['trust_mode'] == 'static':
            t, start = self.times[-1], -np.inf
        else:
            start = t - self.cfg['trust_window']
        comp = {}
        for name, (num, den) in self.counters.items():
            dn = self._at(num, t) - self._at(num, start)
            dd = self._at(den, t) - self._at(den, start)
            ok = dd >= self.cfg['min_obs']
            val = (dn / dd.where(ok)).clip(0, 1)
            self.last[name] = val.where(ok, self.last[name])
            comp[name] = self.last[name]
        # route participation unknown -> fall back to forward ratio alone
        rp = comp['rp'].fillna(comp['fr'])
        trust = (ALPHA * comp['fr'] + BETA * rp).clip(0, 1)
        return trust.to_dict()                       # NaN = never observed


def otsu_threshold(values, bins=256):
    """Label-free bimodal split (Otsu 1979). Always returns a split."""
    v = np.asarray(values, dtype=float)
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu0 = np.cumsum(p * centers) / np.maximum(w0, 1e-12)
    mu1 = (np.cumsum((p * centers)[::-1])[::-1]) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.nanargmax(between))])


def gmm_threshold(values, seed=0):
    """Label-free 2-component Gaussian mixture split.

    Returns -inf (accept everyone) when BIC does not support two populations,
    so a unimodal honest distribution is never cut in half."""
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(v) < 4 or np.ptp(v) < 1e-6:
        return -np.inf
    g1 = GaussianMixture(1, random_state=seed).fit(v)
    g2 = GaussianMixture(2, random_state=seed, n_init=3).fit(v)
    if g1.bic(v) <= g2.bic(v):
        return -np.inf
    hi = int(np.argmax(g2.means_.ravel()))
    lo_m, hi_m = sorted(g2.means_.ravel())
    grid = np.linspace(lo_m, hi_m, 512).reshape(-1, 1)
    post = g2.predict_proba(grid)[:, hi]
    return float(grid[int(np.argmax(post >= 0.5)), 0])


def choose_threshold(trust_known, cfg):
    mode = cfg['trust_threshold']
    if mode == 'none':
        return -np.inf
    if mode == 'otsu':
        return otsu_threshold(trust_known)
    if mode == 'gmm':
        return gmm_threshold(trust_known, cfg['seed'])
    return float(mode)


# ------------------------------------------------------------------ load ---

def load_data(csv_path, cfg, shuffle_labels=False, fold=0, n_folds=1):
    df = pd.read_csv(csv_path)
    df, rep = sanitize(df)
    sch = schema(df)
    features = FEATURES_V3 if sch == 'v3' else FEATURES_V2

    vehicles = sorted(df.vehicle_id.unique())
    labels = df.groupby('vehicle_id').label.first()

    if shuffle_labels:
        # NEGATIVE CONTROL: attacker identity moves, behaviour stays.
        rs = np.random.RandomState(cfg['seed'] + 1)
        vals = labels.values.copy()
        rs.shuffle(vals)
        labels = pd.Series(vals, index=labels.index)
        df = df.drop(columns=['label']).merge(
            labels.rename('label'), left_on='vehicle_id', right_index=True)

    mal_vehicles = set(labels[labels == 1].index)
    print(f"\n  {csv_path}  [{sch} schema]")
    print(f"  {len(df)} rows | {len(vehicles)} vehicles | "
          f"{len(mal_vehicles)} malicious ({len(mal_vehicles)/len(vehicles)*100:.0f}%)")
    if rep:
        print(f"  sanitised: {rep}")

    # --- trust: behaviour only ----------------------------------------------
    tracker = TrustTracker(df, cfg)
    final = tracker.trust_at(tracker.times[-1]) if cfg['trust_mode'] == 'static' else None
    if final is not None:
        tracker.reset()
        known = [final[v] for v in vehicles if not np.isnan(final[v])]
        th = np.nanmean([final[v] for v in vehicles if v not in mal_vehicles])
        tm = np.nanmean([final[v] for v in vehicles if v in mal_vehicles]) if mal_vehicles else float('nan')
        print(f"  static trust: honest {th:.3f} | malicious {tm:.3f} | "
              f"threshold({cfg['trust_threshold']}) = {choose_threshold(known, cfg):.3f}")

    # --- split SUBJECT vehicles into train/test (stratified) ----------------
    # n_folds == 1: one holdout of test_vehicle_ratio. n_folds > 1: stratified
    # k-fold over vehicles; fold k tests on the k-th slice of honest and of
    # malicious vehicles, so across folds every vehicle is tested exactly once.
    rng = np.random.RandomState(cfg['seed'])
    hon = [v for v in vehicles if v not in mal_vehicles]
    mal = [v for v in vehicles if v in mal_vehicles]
    rng.shuffle(hon); rng.shuffle(mal)
    if n_folds > 1:
        test_v = set(np.array_split(hon, n_folds)[fold].tolist() +
                     (np.array_split(mal, n_folds)[fold].tolist() if mal else []))
    else:
        k_h = max(1, int(len(hon) * cfg['test_vehicle_ratio']))
        k_m = max(1, int(len(mal) * cfg['test_vehicle_ratio'])) if mal else 0
        test_v = set(hon[:k_h] + mal[:k_m])
    train_v = [v for v in vehicles if v not in test_v]

    tr = df[df.vehicle_id.isin(train_v)]
    te = df[df.vehicle_id.isin(test_v)]
    scaler = StandardScaler().fit(tr[features])
    Xte, yte = scaler.transform(te[features]), te.label.values

    # --- clients ------------------------------------------------------------
    mode = cfg['clients']
    nb_path = Path(str(csv_path).replace('_features.csv', '_neighbors.csv'))
    if mode == 'auto':
        mode = 'observed' if nb_path.exists() else 'iid'
    cfg['clients'] = mode

    clients = {}
    if mode == 'observed':
        if not nb_path.exists():
            raise SystemExit(f"--clients observed needs {nb_path} (v3 simulator output)")
        nb = pd.read_csv(nb_path)
        nb = nb[nb.neighbor_id.isin(train_v)]
        # observer v at time t sees subject u's row (t, u)
        obs = nb.merge(tr, left_on=['timestamp', 'neighbor_id'],
                       right_on=['timestamp', 'vehicle_id'], suffixes=('_obs', ''))
        groups = dict(tuple(obs.groupby('vehicle_id_obs')))
        for v in vehicles:
            g = groups.get(v)
            if g is None or len(g) == 0:
                Xc, yc = np.zeros((0, len(features))), np.zeros(0)
            else:
                Xc, yc = scaler.transform(g[features]), g.label.values.astype(float)
            clients[v] = dict(X=Xc, y=yc)
    else:
        Xtr, ytr = scaler.transform(tr[features]), tr.label.values
        perm = rng.permutation(len(Xtr))
        for v, idx in zip(vehicles, np.array_split(perm, len(vehicles))):
            clients[v] = dict(X=Xtr[idx].copy(), y=ytr[idx].astype(float).copy())

    # Who poisons the model. 'label' (default): the packet droppers themselves.
    # 'decoupled': a separate set of round(poison_frac * N) vehicles drawn from
    # those that FORWARD HONESTLY (label 0). Their routing looks clean, so the
    # trust filter has nothing to catch; droppers then train honestly.
    if cfg['poison_set'] == 'decoupled':
        k = int(round(cfg['poison_frac'] * len(vehicles)))
        if k > len(hon):
            raise SystemExit(f"--poison-frac {cfg['poison_frac']} needs {k} honest-forwarding "
                             f"vehicles, only {len(hon)} available")
        prs = np.random.RandomState(cfg['seed'] + 7)
        poisoners = set(prs.choice(sorted(hon), size=k, replace=False).tolist())
        print(f"  decoupled poisoners: {k} honest-forwarding vehicles (droppers train honestly)")
    else:
        poisoners = mal_vehicles

    for v in vehicles:
        c = clients[v]
        c['is_mal'] = v in poisoners                  # attack injection + REPORTING only
        if c['is_mal'] and cfg['attack'] == 'label_flip':
            c['y'] = 1 - c['y']
        c['n'] = len(c['X'])

    sizes = [clients[v]['n'] for v in vehicles]
    pos = [clients[v]['y'].mean() for v in vehicles if clients[v]['n']]
    print(f"  fold {fold + 1}/{n_folds} | "
          f"split: {len(train_v)} train / {len(test_v)} test subject vehicles (disjoint) | "
          f"{len(Xte)} test rows")
    print(f"  clients: {mode} | rows/client min {min(sizes)} median {int(np.median(sizes))} "
          f"max {max(sizes)} | empty {sum(s == 0 for s in sizes)} | "
          f"positive share {np.min(pos):.2f}-{np.max(pos):.2f}")

    # --- FL participation from the combined simulator (vanet-pq-trust-fl.cc) ---
    # uploads[r] = (round start time, vehicles selected, vehicles whose upload completed)
    tracker.uploads = None
    up = cfg['uploads']
    if up == 'auto':
        p = Path(str(csv_path).replace('_features.csv', '_uploads.csv'))
        up = str(p) if p.exists() else None
    if up:
        u = pd.read_csv(up)
        tracker.uploads = [(float(g.t_start.iloc[0]), set(g.vehicle_id), set(g.vehicle_id[g.done == 1]))
                           for _, g in u.groupby('round')]
        done = u[u.done == 1]
        print(f"  uploads: {up} | {len(tracker.uploads)} rounds | q_h {u[u.label == 0].done.mean():.2f} "
              f"q_m {u[u.label == 1].done.mean():.2f} | rho_sel {u.label.mean():.2f} "
              f"rho_eff {done.label.mean() if len(done) else float('nan'):.2f}")

    test_loader = DataLoader(VanetDataset(Xte, yte), batch_size=cfg['batch_size'])
    return clients, test_loader, len(features), tracker


# -------------------------------------------------------------- training ---

def train_local(model, data, cfg, device):
    loader = DataLoader(VanetDataset(data['X'], data['y']),
                        batch_size=cfg['batch_size'], shuffle=True)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    crit = nn.BCELoss()
    model.train()
    for _ in range(cfg['local_epochs']):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()
    return model.state_dict()


def predict(model, loader, device):
    model.eval()
    P, L = [], []
    with torch.no_grad():
        for X, y in loader:
            P.extend((model(X.to(device)) > 0.5).float().cpu().numpy().flatten())
            L.extend(y.numpy().flatten())
    return np.array(P), np.array(L)


def scores(P, L):
    return dict(accuracy=accuracy_score(L, P),
                precision=precision_score(L, P, zero_division=0),
                recall=recall_score(L, P, zero_division=0),
                f1=f1_score(L, P, zero_division=0),
                cm=confusion_matrix(L, P, labels=[0, 1]).tolist())


def evaluate(model, loader, device):
    return scores(*predict(model, loader, device))


# -------------------------------------------------------------- attacks ---

def apply_attack(local_w, global_w, attack, cfg):
    if attack in ('none', 'label_flip', 'alie'):
        return local_w                      # label_flip: applied to y; alie: see alie_attack
    delta = {k: local_w[k].float() - global_w[k].float() for k in local_w}
    if attack == 'sign_flip':
        lam = cfg['attack_scale']
        return {k: global_w[k].float() - lam * delta[k] for k in delta}
    if attack == 'scaled':
        lam = cfg['attack_scale']
        return {k: global_w[k].float() + lam * delta[k] for k in delta}
    if attack == 'gaussian':
        sig = cfg['attack_sigma']
        return {k: global_w[k].float() + torch.randn_like(global_w[k].float()) * sig
                for k in delta}
    raise ValueError(f'unknown attack: {attack}')


def alie_attack(updates, mal_idx, z):
    """A Little Is Enough (Baruch et al., NeurIPS 2019), non-omniscient variant.

    Attackers do not see honest updates. Each trains honestly on its own data;
    together they take the per-coordinate mean mu and std sigma of their own
    updates and ALL send mu - z * sigma. The poison is small, identical across
    attackers and sits inside the natural spread, so distance-based rules see
    a tight cluster rather than outliers."""
    if not mal_idx:
        return updates
    out = list(updates)
    poison = {}
    for k in updates[0]:
        S = torch.stack([updates[i][k].float() for i in mal_idx])
        sd = S.std(dim=0) if len(mal_idx) > 1 else torch.zeros_like(S[0])
        poison[k] = S.mean(dim=0) - z * sd
    for i in mal_idx:
        out[i] = {k: v.clone() for k, v in poison.items()}
    return out


def weighted_mean(updates, weights):
    tot = sum(weights)
    return {k: sum(u[k] * w for u, w in zip(updates, weights)) / tot
            for k in updates[0]}


# ------------------------------------------------------- aggregation rules --

def _flat(u):
    return torch.cat([v.flatten().float() for v in u.values()])


def krum_index(updates, f):
    n = len(updates)
    k = max(1, n - f - 2)
    V = torch.stack([_flat(u) for u in updates])
    D = torch.cdist(V, V) ** 2
    scores = [D[i].sort().values[1:k + 1].sum().item() for i in range(n)]
    return int(np.argmin(scores))


def agg_krum(updates, f):
    return copy.deepcopy(updates[krum_index(updates, f)])


def agg_trimmed_mean(updates, f):
    out = {}
    for k in updates[0]:
        S, _ = torch.stack([u[k].float() for u in updates]).sort(dim=0)
        lo, hi = f, len(updates) - f
        out[k] = S[lo:hi].mean(dim=0) if hi > lo else S.mean(dim=0)
    return out


def agg_median(updates, f=None):
    return {k: torch.stack([u[k].float() for u in updates]).median(dim=0).values
            for k in updates[0]}


def _byz_f(n, cfg):
    f = cfg['krum_f'] if cfg['krum_f'] is not None else n // 4
    return max(0, min(f, (n - 3) // 2))


def trust_filter(ctx, cfg):
    """Indices that pass the trust filter, plus the accept/reject breakdown.
    The breakdown reads is_mal for REPORTING only; it never steers training."""
    thr, keep, info = ctx['thr'], [], dict(acc_hon=0, acc_mal=0, rej_hon=0, rej_mal=0, unknown=0)
    for i, t in enumerate(ctx['trusts']):
        unknown = np.isnan(t)
        info['unknown'] += int(unknown)
        ok = (cfg['cold_start'] == 'accept') if unknown else t >= thr
        tag = 'mal' if ctx['is_mal'][i] else 'hon'
        info[('acc_' if ok else 'rej_') + tag] += 1
        if ok:
            keep.append(i)
    return keep, info


def _trust_weight(t):
    return 0.5 if np.isnan(t) else max(t, 1e-3)


def rule_fedavg(updates, ctx, cfg):
    return weighted_mean(updates, ctx['sizes']), {}


def make_byz_rule(fn):
    def r(updates, ctx, cfg):
        f = _byz_f(len(updates), cfg)
        if fn is agg_krum:
            i = krum_index(updates, f)
            # pick_mal: attackers inside the update Krum selected (REPORTING only)
            return copy.deepcopy(updates[i]), dict(f=f, pick_mal=int(ctx['is_mal'][i]))
        return fn(updates, f), dict(f=f)
    return r


def rule_trust_flad(updates, ctx, cfg):
    """Eq. 6: filter below threshold, then weight by trust (x sample count)."""
    keep, info = trust_filter(ctx, cfg)
    if not keep:
        return None, info
    ws = [ctx['sizes'][i] * (1.0 if cfg['uniform_after_filter'] else _trust_weight(ctx['trusts'][i]))
          for i in keep]
    return weighted_mean([updates[i] for i in keep], ws), info


def rule_trust_flad_krum(updates, ctx, cfg):
    keep, info = trust_filter(ctx, cfg)
    if not keep:
        return None, info
    kept = [updates[i] for i in keep]
    if len(kept) < 4:
        return agg_median(kept), info
    j = krum_index(kept, max(1, _byz_f(len(kept), cfg)))
    return copy.deepcopy(kept[j]), dict(info, pick_mal=int(ctx['is_mal'][keep[j]]))


def rule_trust_flad_median(updates, ctx, cfg):
    keep, info = trust_filter(ctx, cfg)
    if not keep:
        return None, info
    return agg_median([updates[i] for i in keep]), info


# ------------------------------------------------------------- bucketing ---
# Karimireddy, He & Jaggi, "Byzantine-robust learning on heterogeneous
# datasets via bucketing", ICLR 2022. Shuffle the updates, average them in
# buckets of size s, then run the robust rule on the bucket means. Averaging
# shrinks the honest heterogeneity that robust rules mistake for attacks.
# Cost: up to s times more buckets can contain an attacker, so it only helps
# when the attacker share entering the rule is well below 1/(2s).

def bucketize(updates, sizes, s, seed):
    """Returns (bucket means, member indices per bucket)."""
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(updates))
    out, groups = [], []
    for b in range(0, len(perm), s):
        idx = perm[b:b + s]
        out.append(weighted_mean([updates[i] for i in idx], [sizes[i] for i in idx]))
        groups.append(list(idx))
    return out, groups


def _bucket_report(groups, is_mal, picked=None):
    """REPORTING only: contaminated buckets, and attackers in the bucket Krum picked."""
    info = dict(bkt_dirty=sum(any(is_mal[i] for i in g) for g in groups))
    if picked is not None:
        info['pick_mal'] = sum(int(is_mal[i]) for i in groups[picked])
    return info


def _bucket_seed(ctx, cfg):
    return cfg['seed'] * 1000 + ctx['round']


def make_bucket_rule(fn):
    def r(updates, ctx, cfg):
        B, G = bucketize(updates, ctx['sizes'], cfg['bucket_size'], _bucket_seed(ctx, cfg))
        f = _byz_f(len(B), cfg)
        if fn is agg_krum and len(B) >= 4:
            j = krum_index(B, f)
            return copy.deepcopy(B[j]), dict(f=f, buckets=len(B), **_bucket_report(G, ctx['is_mal'], j))
        return agg_median(B), dict(f=f, buckets=len(B), **_bucket_report(G, ctx['is_mal']))
    return r


def make_trust_bucket_rule(fn):
    def r(updates, ctx, cfg):
        keep, info = trust_filter(ctx, cfg)
        if not keep:
            return None, info
        B, G = bucketize([updates[i] for i in keep], [ctx['sizes'][i] for i in keep],
                         cfg['bucket_size'], _bucket_seed(ctx, cfg))
        mal_kept = [ctx['is_mal'][i] for i in keep]
        if fn is agg_krum and len(B) >= 4:
            j = krum_index(B, max(1, _byz_f(len(B), cfg)))
            return copy.deepcopy(B[j]), dict(info, buckets=len(B), **_bucket_report(G, mal_kept, j))
        return agg_median(B), dict(info, buckets=len(B), **_bucket_report(G, mal_kept))
    return r


RULES = [
    ('FedAvg',            rule_fedavg),
    ('Krum',              make_byz_rule(agg_krum)),
    ('TrimmedMean',       make_byz_rule(agg_trimmed_mean)),
    ('Median',            make_byz_rule(agg_median)),
    ('Trust-FLAD',        rule_trust_flad),
    ('Trust-FLAD+Krum',   rule_trust_flad_krum),
    ('Trust-FLAD+Median', rule_trust_flad_median),
    ('Krum+Bucket',              make_bucket_rule(agg_krum)),
    ('Median+Bucket',            make_bucket_rule(agg_median)),
    ('Trust-FLAD+Krum+Bucket',   make_trust_bucket_rule(agg_krum)),
    ('Trust-FLAD+Median+Bucket', make_trust_bucket_rule(agg_median)),
]


# ----------------------------------------------------------- experiments ---

def run(name, clients, test_loader, input_dim, tracker, cfg, rule):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg['seed'])
    model = BlackholeDetector(input_dim).to(device)
    print(f"\n{'='*62}\n{name}\n{'='*62}")
    active = [v for v, d in clients.items() if d['n'] > 0]
    tracker.reset()

    history = []
    R = len(tracker.uploads) if tracker.uploads else cfg['rounds_plain']
    all_active = active
    for r in range(1, R + 1):
        part = {}
        if tracker.uploads:
            # only vehicles whose round-r upload reached the RSU; trust as of the round start
            t_r, sel, done = tracker.uploads[r - 1]
            active = [v for v in all_active if v in done]
            part = dict(sel_hon=sum(not clients[v]['is_mal'] for v in sel if v in clients),
                        sel_mal=sum(clients[v]['is_mal'] for v in sel if v in clients),
                        arr_hon=sum(not clients[v]['is_mal'] for v in active),
                        arr_mal=sum(clients[v]['is_mal'] for v in active))
        else:
            t_r = tracker.round_time(r, R)
        trust = tracker.trust_at(t_r)
        known = [trust[v] for v in all_active if not np.isnan(trust[v])]
        thr = choose_threshold(known, cfg) if known else -np.inf
        if not active:
            m = evaluate(model, test_loader, device)
            history.append(dict(round=r, t_sim=round(float(t_r), 2), thr=None,
                                **{k: v for k, v in m.items() if k != 'cm'}, **part))
            continue

        gw = model.state_dict()
        updates = []
        for cid in active:
            d = clients[cid]
            lm = BlackholeDetector(input_dim).to(device)
            lm.load_state_dict(copy.deepcopy(gw))
            w = train_local(lm, d, cfg, device)
            if d['is_mal']:
                w = apply_attack(w, gw, cfg['attack'], cfg)
            updates.append(w)
        if cfg['attack'] == 'alie':
            updates = alie_attack(updates, [i for i, c in enumerate(active) if clients[c]['is_mal']],
                                  cfg['alie_z'])

        ctx = dict(trusts=[trust[c] for c in active], sizes=[clients[c]['n'] for c in active],
                   is_mal=[clients[c]['is_mal'] for c in active], thr=thr, round=r)
        new_w, info = rule(updates, ctx, cfg)
        info = {**part, **info}
        if new_w is not None:
            model.load_state_dict(new_w)

        m = evaluate(model, test_loader, device)
        history.append(dict(round=r, t_sim=round(float(t_r), 2),
                            thr=None if np.isinf(thr) else round(thr, 4),
                            **{k: v for k, v in m.items() if k != 'cm'}, **info))
        if r % 5 == 0 or r == 1:
            extra = ' '.join(f'{k}={v}' for k, v in info.items())
            ts = 'accept-all' if np.isinf(thr) else f'{thr:.3f}'
            print(f"  round {r:2d} t={t_r:5.1f}s thr={ts}: acc={m['accuracy']:.4f} "
                  f"f1={m['f1']:.4f} {extra}")

    preds = predict(model, test_loader, device)
    final = scores(*preds)
    h = pd.DataFrame(history)
    if 'acc_mal' in h:
        n_mal = h.acc_mal + h.rej_mal
        n_hon = h.acc_hon + h.rej_hon
        final['beta'] = float((h.acc_mal / n_mal.where(n_mal > 0)).mean())
        final['gamma'] = float((h.rej_hon / n_hon.where(n_hon > 0)).mean())
        final['accepted_mean'] = float((h.acc_hon + h.acc_mal).mean())
    if 'arr_mal' in h:
        arr = h.arr_hon + h.arr_mal
        final['rho_eff'] = float(h.arr_mal.sum() / max(1, arr.sum()))
        final['bad_round_frac'] = float(((h.arr_mal >= h.arr_hon) & (h.arr_mal > 0)).mean())
        if 'acc_mal' in h:
            kept = h.acc_hon.fillna(0) + h.acc_mal.fillna(0)
            final['rho_surv'] = float(h.acc_mal.fillna(0).sum() / max(1, kept.sum()))
    if 'pick_mal' in h:
        final['pick_mal_rate'] = float((h.pick_mal.dropna() > 0).mean())   # rounds Krum chose poison
    if 'bkt_dirty' in h:
        final['bkt_dirty_frac'] = float((h.bkt_dirty / h.buckets).mean())
    print(f"  FINAL  acc={final['accuracy']:.4f} P={final['precision']:.4f} "
          f"R={final['recall']:.4f} F1={final['f1']:.4f}"
          + (f"  beta={final['beta']:.2f} gamma={final['gamma']:.2f}" if 'beta' in final else ''))
    return history, final, preds


# ------------------------------------------------------------------ main ---

def main():
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--threshold', default='gmm',
                    help="'gmm' (label-free, may accept all), 'otsu', 'none', or a float")
    ap.add_argument('--trust-mode', default='window', choices=['window', 'static'])
    ap.add_argument('--trust-window', type=float, default=10.0, help='seconds')
    ap.add_argument('--min-obs', type=int, default=5,
                    help='events needed in a window before a ratio is updated')
    ap.add_argument('--trust-memory', type=float, default=0.0,
                    help='slow recovery: keep the worst windowed trust of the last M s (0 = off)')
    ap.add_argument('--warmup', type=float, default=0.0,
                    help='seconds of trust observation before the first FL round (rounds still span the run)')
    ap.add_argument('--trust-norm', default='none', choices=['none', 'local'],
                    help='local = divide by the 75th-percentile trust of vehicles within --norm-radius')
    ap.add_argument('--norm-radius', type=float, default=300.0)
    ap.add_argument('--cold-start', default='accept', choices=['accept', 'reject'],
                    help='what the filter does with never-observed vehicles')
    ap.add_argument('--clients', default='auto', choices=['auto', 'observed', 'iid'],
                    help="auto = observed if <prefix>_neighbors.csv exists, else iid")
    ap.add_argument('--attack', default='sign_flip',
                    choices=['none', 'label_flip', 'sign_flip', 'scaled', 'gaussian', 'alie'])
    ap.add_argument('--attack-scale', type=float, default=5.0)
    ap.add_argument('--attack-sigma', type=float, default=1.0)
    ap.add_argument('--poison-set', default='label', choices=['label', 'decoupled'],
                    help="label: droppers poison; decoupled: honest-forwarding vehicles poison")
    ap.add_argument('--poison-frac', type=float, default=0.3,
                    help='fraction of all vehicles that poison when --poison-set decoupled')
    ap.add_argument('--alie-z', type=float, default=1.0,
                    help='ALIE shift in units of the attackers\' per-coordinate std')
    ap.add_argument('--uniform-after-filter', action='store_true')
    ap.add_argument('--bucket-size', type=int, default=2,
                    help='bucket size s for the *+Bucket rules (Karimireddy et al. 2022)')
    ap.add_argument('--methods', default='all',
                    help='comma-separated subset of: ' + ','.join(n for n, _ in RULES))
    ap.add_argument('--batch-size', type=int, default=32,
                    help='local batch size (128 is about 3-4x faster for this small MLP)')
    ap.add_argument('--folds', type=int, default=1,
                    help='k-fold over vehicles (stratified); test predictions are pooled')
    ap.add_argument('--uploads', default='auto',
                    help="auto = use <prefix>_uploads.csv if present (combined simulator), "
                         "'none' = everyone every round, or a path")
    ap.add_argument('--out', default='results_v3')
    ap.add_argument('--shuffle-labels', action='store_true',
                    help='NEGATIVE CONTROL: permute attacker identities.')
    a = ap.parse_args()

    cfg = dict(DEFAULTS, seed=a.seed, rounds_plain=a.rounds, trust_threshold=a.threshold,
               attack=a.attack, attack_scale=a.attack_scale, attack_sigma=a.attack_sigma, alie_z=a.alie_z,
               uniform_after_filter=a.uniform_after_filter, trust_mode=a.trust_mode,
               trust_window=a.trust_window, min_obs=a.min_obs, cold_start=a.cold_start,
               clients=a.clients, bucket_size=a.bucket_size,
               batch_size=a.batch_size, poison_set=a.poison_set, poison_frac=a.poison_frac,
               uploads=None if a.uploads == 'none' else a.uploads,
               trust_memory=a.trust_memory, trust_norm=a.trust_norm, norm_radius=a.norm_radius,
               warmup=a.warmup)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    print('=' * 62)
    print('TRUST-FLAD v3  --  windowed trust, label-free threshold, observed clients')
    print(f"attack: {a.attack}  threshold: {a.threshold}  trust: {a.trust_mode}"
          + (f" (W={a.trust_window}s)" if a.trust_mode == 'window' else ''))
    print('=' * 62)
    if a.shuffle_labels:
        print('\n  !! NEGATIVE CONTROL: attacker identities permuted')

    wanted = None if a.methods == 'all' else set(a.methods.split(','))
    per = {}          # method -> list of (history, final, preds) per fold
    for fold in range(a.folds):
        clients, test_loader, dim, tracker = load_data(
            a.csv, cfg, shuffle_labels=a.shuffle_labels, fold=fold, n_folds=a.folds)
        for name, rule in RULES:
            if wanted and name not in wanted:
                continue
            per.setdefault(name, []).append(run(name, clients, test_loader, dim, tracker, cfg, rule))

    # Pool test predictions across folds: with k-fold every vehicle is scored once.
    results = {}
    for name, runs in per.items():
        P = np.concatenate([p for _, _, (p, _) in runs])
        L = np.concatenate([l for _, _, (_, l) in runs])
        final = scores(P, L)
        for key in ('beta', 'gamma', 'accepted_mean', 'pick_mal_rate', 'bkt_dirty_frac',
                    'rho_eff', 'bad_round_frac', 'rho_surv'):
            vals = [f[key] for _, f, _ in runs if key in f]
            if vals:
                final[key] = float(np.nanmean(vals))
        final['f1_folds'] = [f['f1'] for _, f, _ in runs]
        results[name] = dict(histories=[h for h, _, _ in runs], final=final)

    print('\n' + '=' * 81)
    print(f"{'Method':<26}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'beta':>8}{'gamma':>8}{'kept':>7}")
    print('-' * 81)
    for n, d in results.items():
        f = d['final']
        tb = (f"{f['beta']:>8.2f}{f['gamma']:>8.2f}{f['accepted_mean']:>7.1f}"
              if 'beta' in f else f"{'':>8}{'':>8}{'':>7}")
        print(f"{n:<26}{f['accuracy']:>8.4f}{f['precision']:>8.4f}"
              f"{f['recall']:>8.4f}{f['f1']:>8.4f}{tb}")
    print('=' * 81)
    print("beta = attacker pass rate, gamma = honest rejection rate, kept = clients aggregated/round")

    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"{Path(a.csv).stem}_{a.attack}_{a.threshold}_{a.trust_mode}_{cfg['clients']}" \
          + (f'_z{a.alie_z:g}' if a.attack == 'alie' and a.alie_z != 1.0 else '') \
          + (f'_dec{a.poison_frac:g}' if a.poison_set == 'decoupled' else '') \
          + ('_noup' if a.uploads == 'none' else '') \
          + (f'_mem{a.trust_memory:g}' if a.trust_memory > 0 else '') \
          + ('_lnorm' if a.trust_norm == 'local' else '') \
          + ('_shuffled' if a.shuffle_labels else '')
    for n, d in results.items():
        for k, h in enumerate(d['histories']):
            suffix = f'_f{k}' if a.folds > 1 else ''
            pd.DataFrame(h).to_csv(out / f'{tag}_{n}{suffix}.csv', index=False)
    with open(out / f'{tag}_summary.json', 'w') as fh:
        json.dump({'config': cfg, 'csv': a.csv, 'folds': a.folds,
                   'results': {n: d['final'] for n, d in results.items()}},
                  fh, indent=2, default=str)
    print(f"wrote {out}/{tag}_*.csv and _summary.json")


if __name__ == '__main__':
    main()
