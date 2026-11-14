#!/usr/bin/env python3
"""
D4 PackNet Parameter Isolation — BASELINE (known method, not a candidate).
Iterative magnitude pruning + per-task weight freezing (Mallya & Lazebnik 2018).
Filed under baselines in FINAL_REPORT.md; serves as a strong, known-good comparator.

Key implementation details (PackNet semantics):
- Weights carry forward between tasks (not re-initialized)
- After each task, prune bottom prune_frac% of weights by magnitude and FREEZE them
- Next task trains only on unpruned weights
- At eval time for task i: use only weights unpruned at task i's time, with their
  values as they were right after task i finished training
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


def softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum()


class PackNet:
    """PackNet (Mallya & Lazebnik 2018) using sklearn MLPClassifier as base.

    Per-task: train on unpruned weights, then prune bottom prune_frac% by magnitude
    and freeze them. Next task trains only on remaining unpruned weights, starting
    from where they were after pruning.

    At evaluation for task i: restore weights to their state right after task i,
    using only weights unpruned at task i's time (later pruned weights are ignored).
    """
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3):
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.seed = seed
        self.prune_frac = prune_frac
        self.task_count = 0

        # Current model weights
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

        # Per-task snapshots: for each task, store (w1, b1, w2, b2, mask_w1, mask_w2)
        # as they were right after that task finished training
        self.task_snapshots = []

        # Current masks (which weights are still trainable)
        self.mask_w1 = None
        self.mask_w2 = None

    def _init_weights(self, X, y):
        """Initialize weights randomly."""
        self.rng = np.random.default_rng(self.seed)
        self.w1 = self.rng.standard_normal((self.in_dim, self.hidden)) * 0.1
        self.b1 = np.zeros(self.hidden, dtype=np.float64)
        self.w2 = self.rng.standard_normal((self.hidden, self.out_dim)) * 0.1
        self.b2 = np.zeros(self.out_dim, dtype=np.float64)
        self.mask_w1 = np.ones_like(self.w1, dtype=bool)
        self.mask_w2 = np.ones_like(self.w2, dtype=bool)
        # Momentum buffers
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)

    def train_step(self, X, y, lr=0.05, epochs=50):
        """Train on unpruned weights only using mini-batch SGD with momentum."""
        if self.w1 is None:
            self._init_weights(X, y)

        n = len(X)
        for ep in range(epochs):
            idx = np.random.default_rng(self.seed + ep).permutation(n)
            for i in idx:
                x = X[i]
                yi = y[i]
                # Forward pass
                h = np.maximum(0, x @ self.w1 + self.b1)
                logits = h @ self.w2 + self.b2
                probs = softmax(logits)
                y_onehot = np.zeros(self.out_dim, dtype=np.float64)
                y_onehot[int(yi)] = 1.0
                err = probs - y_onehot
                # Gradients
                d_w2 = np.outer(h, err)
                d_b2 = err
                d_h = err @ self.w2.T * (h > 0)
                d_w1 = np.outer(x, d_h)
                d_b1 = d_h
                # Momentum updates (masked: only unpruned weights change)
                self.v_w2 = 0.9 * self.v_w2 - lr * d_w2 * self.mask_w2
                self.v_b2 = 0.9 * self.v_b2 - lr * d_b2
                self.v_w1 = 0.9 * self.v_w1 - lr * d_w1 * self.mask_w1
                self.v_b1 = 0.9 * self.v_b1 - lr * d_b1
                self.w2 += self.v_w2
                self.b2 += self.v_b2
                self.w1 += self.v_w1
                self.b1 += self.v_b1

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            h = np.maximum(0, X @ self.w1 + self.b1)
            return h @ self.w2 + self.b2
        h = np.maximum(0, X @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict_proba(self, X):
        logits = self.predict(X)
        probs = softmax(logits)
        return probs

    def predict_class(self, x):
        return int(np.argmax(self.predict(x)))

    def prune_and_freeze(self):
        """Prune bottom prune_frac% of weights by magnitude and freeze them."""
        all_w = np.concatenate([np.abs(self.w1).flatten(), np.abs(self.w2).flatten()])
        threshold = np.percentile(all_w, self.prune_frac * 100)
        m1 = np.abs(self.w1) > threshold
        m2 = np.abs(self.w2) > threshold
        if m1.sum() == 0: m1[:] = True
        if m2.sum() == 0: m2[:] = True
        self.mask_w1 = self.mask_w1 & m1
        self.mask_w2 = self.mask_w2 & m2
        self.task_count += 1

    def save_snapshot(self, task_idx):
        """Save weights and masks right after task task_idx finishes training."""
        self.task_snapshots.append({
            'w1': self.w1.copy(),
            'b1': self.b1.copy(),
            'w2': self.w2.copy(),
            'b2': self.b2.copy(),
            'mask_w1': self.mask_w1.copy(),
            'mask_w2': self.mask_w2.copy(),
        })

    def restore_snapshot(self, task_idx):
        """Restore weights and masks to their state right after task task_idx."""
        snap = self.task_snapshots[task_idx]
        self.w1 = snap['w1'].copy()
        self.b1 = snap['b1'].copy()
        self.w2 = snap['w2'].copy()
        self.b2 = snap['b2'].copy()
        self.mask_w1 = snap['mask_w1'].copy()
        self.mask_w2 = snap['mask_w2'].copy()

    def frozen_count(self):
        total = self.w1.size + self.w2.size
        frozen = (~self.mask_w1).sum() + (~self.mask_w2).sum()
        return int(frozen), int(total), float(frozen / total)


def eval_acc(model, X, y):
    logits = model.predict(X)
    preds = np.argmax(logits, axis=1)
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
            model.save_snapshot(k)
            # Evaluate ALL tasks seen so far using their respective snapshots
            for j in range(k + 1):
                model.restore_snapshot(j)
                Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                task_acc[j, k] = eval_acc(model, Xt, yt)
    tr.update()
    aa = average_accuracy(task_acc)
    fm = forgetting_measure(task_acc)
    frozen, total, frac = model.frozen_count()
    return task_acc, aa, fm, tr.stats, {'frozen': frozen, 'total': total, 'frozen_frac': frac, 'task_count': model.task_count}


def stream_packnet(seed, prune_frac=0.3, hidden=64, chunks=20, chunk_size=200, epochs=3, lr=0.05):
    """Stream over training data in chunks, evaluate on fixed held-out test set."""
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
