#!/usr/bin/env python3
"""
D4 PackNet Parameter Isolation — BASELINE (known method, not a candidate).
Iterative magnitude pruning + per-task weight freezing (Mallya & Lazebnik 2018).
Filed under baselines in FINAL_REPORT.md; serves as a strong, known-good comparator.

Protocols:
- split-digits-5 continual (5 seeds): AA, FM
- non-stationary stream (5 seeds): full vs frozen_after_1 vs no_alloc
- ablations (slots/capacity sweep): prune fraction sweep
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "d4_packnet_baseline")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float64)
    v[int(y)] = 1.0
    return v


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / e.sum()


class PackNet:
    """Fixed backbone MLP with iterative magnitude pruning + per-task weight freezing.
    Mallya & Lazebnik 2018 (PackNet)."""
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3):
        self.rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.w1 = self.rng.standard_normal((in_dim, hidden)) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = self.rng.standard_normal((hidden, out_dim)) * 0.1
        self.b2 = np.zeros(out_dim, dtype=np.float64)
        self.mask1 = np.ones((in_dim, hidden), dtype=bool)
        self.mask2 = np.ones((hidden, out_dim), dtype=bool)
        self.prune_frac = prune_frac
        self.task_count = 0
        # Momentum buffers
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            h = np.maximum(0, X @ self.w1 + self.b1)
            return h @ self.w2 + self.b2
        h = np.maximum(0, X @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict_class(self, x):
        return int(np.argmax(self.predict(x)))

    def train_step(self, x, y, lr, momentum=0.9):
        y_onehot = np.zeros(self.out_dim, dtype=np.float64)
        y_onehot[int(y)] = 1.0
        h = np.maximum(0, x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        probs = softmax(logits)
        err = probs - y_onehot
        d_w2 = np.outer(h, err)
        d_b2 = err
        d_h = err @ self.w2.T * (h > 0)
        d_w1 = np.outer(x, d_h)
        d_b1 = d_h
        # Momentum updates with masking
        self.v_w2 = momentum * self.v_w2 - lr * d_w2 * self.mask2
        self.v_b2 = momentum * self.v_b2 - lr * d_b2
        self.v_w1 = momentum * self.v_w1 - lr * d_w1 * self.mask1
        self.v_b1 = momentum * self.v_b1 - lr * d_b1
        self.w2 += self.v_w2
        self.b2 += self.v_b2
        self.w1 += self.v_w1
        self.b1 += self.v_b1
        return float(-np.sum(y_onehot * np.log(probs + 1e-12)))

    def prune_and_freeze(self):
        all_w = np.concatenate([self.w1.flatten(), self.w2.flatten()])
        threshold = np.percentile(np.abs(all_w), self.prune_frac * 100)
        m1 = np.abs(self.w1) > threshold
        m2 = np.abs(self.w2) > threshold
        if m1.sum() == 0: m1[:] = True
        if m2.sum() == 0: m2[:] = True
        self.mask1 = self.mask1 & m1
        self.mask2 = self.mask2 & m2
        # Zero out momentum for pruned weights
        self.v_w1 *= self.mask1
        self.v_w2 *= self.mask2
        self.task_count += 1

    def frozen_count(self):
        total = self.w1.size + self.w2.size
        frozen = (~self.mask1).sum() + (~self.mask2).sum()
        return int(frozen), int(total), float(frozen / total)


def eval_acc(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float(np.mean(np.array(preds) == y))


def continual_packnet(seed, prune_frac=0.3, hidden=64, epochs=50, lr=0.05):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed)
    model = PackNet(64, hidden, 10, seed=seed, prune_frac=prune_frac)
    T = len(tasks)
    task_acc = np.zeros((T, T))
    with Tracker() as tr:
        for k, task in enumerate(tasks):
            Xtr, ytr = task['X_train'], task['y_train']
            for ep in range(epochs):
                idx = np.random.default_rng(seed * 1000 + k * 10 + ep).permutation(len(Xtr))
                for i in idx:
                    model.train_step(Xtr[i], ytr[i], lr)
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
            for ep in range(epochs):
                for i in idx:
                    model.train_step(Xtr[i], ytr[i], lr)
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
