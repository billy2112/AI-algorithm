#!/usr/bin/env python3
"""
D4 PackNet Parameter Isolation — BASELINE (known method, not a candidate).
Iterative magnitude pruning + per-task weight freezing (Mallya & Lazebnik 2018).
Filed under baselines in FINAL_REPORT.md; serves as a strong, known-good comparator.

Uses sklearn MLPClassifier as the base learner (verified working) with magnitude-based
pruning + weight freezing between tasks.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure
from sklearn.neural_network import MLPClassifier

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "d4_packnet_baseline")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


class PackNet:
    """PackNet (Mallya & Lazebnik 2018) using sklearn MLPClassifier as base.
    After each task, prunes bottom prune_frac% of weights by magnitude and freezes them.
    Subsequent tasks only train on unpruned (active) weights."""
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3):
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.seed = seed
        self.prune_frac = prune_frac
        self.task_count = 0
        self.mask_w1 = None
        self.mask_b1 = None
        self.mask_w2 = None
        self.mask_b2 = None
        self.clf = None

    def _init_clf(self, X, y):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(self.hidden,),
            max_iter=1,
            learning_rate_init=0.05,
            momentum=0.9,
            random_state=self.seed,
            warm_start=True,
        )
        self.clf.fit(X, y)
        # Initialize masks (all trainable)
        self.mask_w1 = np.ones_like(self.clf.coefs_[0], dtype=bool)
        self.mask_b1 = np.ones_like(self.clf.intercepts_[0], dtype=bool)
        self.mask_w2 = np.ones_like(self.clf.coefs_[1], dtype=bool)
        self.mask_b2 = np.ones_like(self.clf.intercepts_[1], dtype=bool)

    def _apply_masks(self):
        """Apply pruning masks to weights before training."""
        self.clf.coefs_[0] = self.clf.coefs_[0] * self.mask_w1
        self.clf.intercepts_[0] = self.clf.intercepts_[0] * self.mask_b1
        self.clf.coefs_[1] = self.clf.coefs_[1] * self.mask_w2
        self.clf.intercepts_[1] = self.clf.intercepts_[1] * self.mask_b2

    def train_step(self, X, y, lr=0.05, epochs=1):
        if self.clf is None:
            self._init_clf(X, y)
        else:
            self.clf.max_iter = epochs
            self.clf.learning_rate_init = lr
            self.clf.random_state = self.seed
            self.clf.warm_start = True
            self.clf.fit(X, y)
        # Apply masks after training (zero out pruned weights)
        self._apply_masks()

    def prune_and_freeze(self):
        """Prune bottom prune_frac% of weights by magnitude and freeze them."""
        all_w = np.concatenate([
            np.abs(self.clf.coefs_[0]).flatten(),
            np.abs(self.clf.intercepts_[0]).flatten(),
            np.abs(self.clf.coefs_[1]).flatten(),
            np.abs(self.clf.intercepts_[1]).flatten(),
        ])
        threshold = np.percentile(all_w, self.prune_frac * 100)
        m1 = np.abs(self.clf.coefs_[0]) > threshold
        m2 = np.abs(self.clf.intercepts_[0]) > threshold
        m3 = np.abs(self.clf.coefs_[1]) > threshold
        m4 = np.abs(self.clf.intercepts_[1]) > threshold
        if m1.sum() == 0: m1[:] = True
        if m2.sum() == 0: m2[:] = True
        if m3.sum() == 0: m3[:] = True
        if m4.sum() == 0: m4[:] = True
        self.mask_w1 = self.mask_w1 & m1
        self.mask_b1 = self.mask_b1 & m2
        self.mask_w2 = self.mask_w2 & m3
        self.mask_b2 = self.mask_b2 & m4
        self.task_count += 1

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def frozen_count(self):
        total = (self.clf.coefs_[0].size + self.clf.intercepts_[0].size +
                 self.clf.coefs_[1].size + self.clf.intercepts_[1].size)
        frozen = (~self.mask_w1).sum() + (~self.mask_b1).sum() + \
                 (~self.mask_w2).sum() + (~self.mask_b2).sum()
        return int(frozen), int(total), float(frozen / total)


def eval_acc(model, X, y):
    preds = model.predict(X)
    return float(np.mean(preds == y))


def continual_packnet(seed, prune_frac=0.3, hidden=64, epochs=50, lr=0.05):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed)
    model = PackNet(64, hidden, 10, seed=seed, prune_frac=prune_frac)
    T = len(tasks)
    task_acc = np.zeros((T, T))
    with Tracker() as tr:
        for k, task in enumerate(tasks):
            Xtr, ytr = task['X_train'], task['y_train']
            model.train_step(Xtr, ytr, lr=lr, epochs=epochs)
            model.prune_and_freeze()
            for j in range(T):
                Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                task_acc[j, k] = eval_acc(model, Xt, yt)
    tr.update()
    aa = average_accuracy(task_acc)
    fm = forgetting_measure(task_acc)
    frozen, total, frac = model.frozen_count()
    return task_acc, aa, fm, tr.stats, {'frozen': frozen, 'total': total, 'frozen_frac': frac, 'task_count': model.task_count}


def stream_packnet(seed, prune_frac=0.3, hidden=64, chunks=20, chunk_size=200, epochs=3, lr=0.05):
    set_seed(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed)
    rng = np.random.default_rng(seed)
    model = PackNet(64, hidden, 10, seed=seed, prune_frac=prune_frac)
    accs = []
    total_seen = 0
    with Tracker() as tr:
        for c in range(chunks):
            idx = rng.integers(0, len(Xtr), size=chunk_size)
            Xc = Xtr[idx]; yc = ytr[idx]
            model.train_step(Xc, yc, lr=lr, epochs=epochs)
            if (c + 1) % 5 == 0:
                model.prune_and_freeze()
            total_seen += chunk_size
            if c in [0, 5, 9, 14, 19]:
                accs.append({
                    'chunk': c,
                    'examples_seen': total_seen,
                    'acc': eval_acc(model, Xte, yte),
                    'frozen_frac': model.frozen_count()[2],
                })
    tr.update()
    return accs, tr.stats


def run_continual():
    print("=== D4 PackNet Continual (split-digits-5) ===")
    results = {}
    for prune in [0.1, 0.3, 0.5]:
        name = f'packnet_prune{int(prune*100)}'
        aas, fms, walls, peaks, per_seed = [], [], [], [], []
        for s in SEEDS:
            set_seed(s)
            mat, aa, fm, st, info = continual_packnet(s, prune_frac=prune)
            aas.append(aa); fms.append(fm); walls.append(st.wall_seconds); peaks.append(st.peak_rss_mb)
            per_seed.append({'seed': s, 'AA': aa, 'FM': fm, 'matrix': mat.tolist(),
                             'frozen_frac': info['frozen_frac'], 'frozen': info['frozen'], 'total': info['total']})
        results[name] = {'prune_frac': prune, 'per_seed': per_seed,
                         'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
                         'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}
        print(f"  {name}: AA={results[name]['AA']['mean']:.4f}±{results[name]['AA']['std']:.4f} "
              f"FM={results[name]['FM']['mean']:.4f}±{results[name]['FM']['std']:.4f} "
              f"frozen_frac={results[name]['per_seed'][0]['frozen_frac']:.3f}")
    return results


def run_stream():
    print("=== D4 PackNet Stream (20 chunks) ===")
    results = {}
    for prune in [0.1, 0.3, 0.5]:
        name = f'packnet_prune{int(prune*100)}'
        per_seed, first, last, deltas = [], [], [], []
        for s in SEEDS:
            set_seed(s)
            curve, st = stream_packnet(s, prune_frac=prune)
            per_seed.append({'seed': s, 'curve': curve, 'wall_s': st.wall_seconds,
                             'peak_rss_mb': st.peak_rss_mb})
            first.append(curve[0]['acc']); last.append(curve[-1]['acc'])
            deltas.append(curve[-1]['acc'] - curve[0]['acc'])
        results[name] = {'prune_frac': prune, 'per_seed': per_seed,
                         'acc_first': bootstrap_ci(first), 'acc_last': bootstrap_ci(last),
                         'delta': bootstrap_ci(deltas)}
        print(f"  {name}: first={results[name]['acc_first']['mean']:.4f} "
              f"last={results[name]['acc_last']['mean']:.4f} "
              f"delta={results[name]['delta']['mean']:.4f}±{results[name]['delta']['std']:.4f}")
    return results


def run_ablations():
    print("=== D4 PackNet Ablations (slots/capacity sweep) ===")
    results = {}
    for hidden in [32, 64, 128]:
        for prune in [0.1, 0.3, 0.5]:
            name = f'packnet_h{hidden}_p{int(prune*100)}'
            aas, fms, walls, peaks, per_seed = [], [], [], [], []
            for s in SEEDS:
                set_seed(s)
                mat, aa, fm, st, info = continual_packnet(s, prune_frac=prune, hidden=hidden)
                aas.append(aa); fms.append(fm); walls.append(st.wall_seconds); peaks.append(st.peak_rss_mb)
                per_seed.append({'seed': s, 'AA': aa, 'FM': fm, 'frozen_frac': info['frozen_frac']})
            results[name] = {'hidden': hidden, 'prune_frac': prune, 'per_seed': per_seed,
                             'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
                             'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}
            print(f"  {name}: AA={results[name]['AA']['mean']:.4f} FM={results[name]['FM']['mean']:.4f} "
                  f"frozen_frac={results[name]['per_seed'][0]['frozen_frac']:.3f}")
    return results


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git': git, 'seeds': SEEDS, 'method': 'PackNet (Mallya & Lazebnik 2018) - BASELINE, not candidate'}
    print("CONTINUAL")
    out['continual'] = run_continual()
    print("STREAM")
    out['stream'] = run_stream()
    print("ABLATIONS")
    out['ablations'] = run_ablations()
    save_json(os.path.join(OUT, 'results.json'), out)
    print(f"\nSaved to {OUT}/results.json")
