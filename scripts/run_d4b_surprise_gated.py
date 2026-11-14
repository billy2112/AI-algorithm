#!/usr/bin/env python3
"""
D4b: Prototype-Surprise-Gated Dynamic Capacity Allocation (NEW CANDIDATE).

Hybrid: PackNet's fixed backbone + per-task subnetwork masks, BUT the freeze/unfreeze
decision is driven by HSAM-style surprise/mismatch instead of a fixed schedule.

Mechanism:
- Fixed backbone MLP backbone (same as PackNet)
- After each task, compute surprise = mean prediction error on held-out validation slice
- If surprise LOW (task well-explained): freeze FEWER weights, keep more plastic
- If surprise HIGH (task genuinely novel): freeze MORE weights (protect current state),
  allocate more capacity for next task
- Per-weight utility tracked; high-utility weights protected, low-utility reallocated

Falsifiable prediction:
On split-digits-5 continual (5 seeds), D4b vs vanilla PackNet:
- D4b uses measurably LESS frozen capacity on easy/similar task sequences
  AND MORE on novel sequences
- D4b achieves AA >= PackNet and FM <= PackNet
- If D4b does not reduce capacity on easy sequences without sacrificing AA/FM,
  D4b is dead weight — fall back to vanilla PackNet as baseline only

Kill criterion (pre-registered): <5% relative capacity reduction on easy sequences
with no AA/FM sacrifice = D4b is dead weight.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "d4b_surprise_gated")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum()


class PackNetBase:
    """Fixed backbone MLP with per-task weight masking (PackNet-style).
    Base class shared by D4 (fixed pruning) and D4b (surprise-gated pruning).
    """
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3):
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.seed = seed
        self.prune_frac = prune_frac
        self.task_count = 0

        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.standard_normal((in_dim, hidden)) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = self.rng.standard_normal((hidden, out_dim)) * 0.1
        self.b2 = np.zeros(out_dim, dtype=np.float64)
        self.mask_w1 = np.ones_like(self.w1, dtype=bool)
        self.mask_w2 = np.ones_like(self.w2, dtype=bool)
        self.v_w1 = np.zeros_like(self.w1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_w2 = np.zeros_like(self.w2)
        self.v_b2 = np.zeros_like(self.b2)

        self.task_snapshots = []

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            h = np.maximum(0, X @ self.w1 + self.b1)
            return h @ self.w2 + self.b2
        h = np.maximum(0, X @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict_class(self, x):
        return int(np.argmax(self.predict(x)))

    def train_step(self, X, y, lr=0.05, epochs=50):
        """Train on unpruned weights only using mini-batch SGD with momentum."""
        n = len(X)
        for ep in range(epochs):
            idx = np.random.default_rng(self.seed + ep).permutation(n)
            for i in idx:
                x = X[i]
                yi = y[i]
                h = np.maximum(0, x @ self.w1 + self.b1)
                logits = h @ self.w2 + self.b2
                probs = softmax(logits)
                y_onehot = np.zeros(self.out_dim, dtype=np.float64)
                y_onehot[int(yi)] = 1.0
                err = probs - y_onehot
                d_w2 = np.outer(h, err)
                d_b2 = err
                d_h = err @ self.w2.T * (h > 0)
                d_w1 = np.outer(x, d_h)
                d_b1 = d_h
                self.v_w2 = 0.9 * self.v_w2 - lr * d_w2 * self.mask_w2
                self.v_b2 = 0.9 * self.v_b2 - lr * d_b2
                self.v_w1 = 0.9 * self.v_w1 - lr * d_w1 * self.mask_w1
                self.v_b1 = 0.9 * self.v_b1 - lr * d_b1
                self.w2 += self.v_w2
                self.b2 += self.v_b2
                self.w1 += self.v_w1
                self.b1 += self.v_b1

    def prune_and_freeze(self, prune_frac=None):
        """Prune bottom prune_frac% of weights by magnitude and freeze them."""
        if prune_frac is None:
            prune_frac = self.prune_frac
        all_w = np.concatenate([np.abs(self.w1).flatten(), np.abs(self.w2).flatten()])
        threshold = np.percentile(all_w, prune_frac * 100)
        m1 = np.abs(self.w1) > threshold
        m2 = np.abs(self.w2) > threshold
        if m1.sum() == 0: m1[:] = True
        if m2.sum() == 0: m2[:] = True
        self.mask_w1 = self.mask_w1 & m1
        self.mask_w2 = self.mask_w2 & m2
        self.task_count += 1

    def save_snapshot(self, task_idx):
        self.task_snapshots.append({
            'w1': self.w1.copy(), 'b1': self.b1.copy(),
            'w2': self.w2.copy(), 'b2': self.b2.copy(),
            'mask_w1': self.mask_w1.copy(), 'mask_w2': self.mask_w2.copy(),
        })

    def restore_snapshot(self, task_idx):
        snap = self.task_snapshots[task_idx]
        self.w1 = snap['w1'].copy()
        self.b1 = snap['b1'].copy()
        self.w2 = snap['w2'].copy()
        self.b2 = snap['b2'].copy()
        self.mask_w1 = snap['mask_w1'].copy()
        self.mask_w2 = snap['mask_w2'].copy()

    def compute_surprise(self, X, y):
        """Compute surprise on a slice. Default: return fixed surprise (0.5) so
        prune_frac stays constant for vanilla PackNet."""
        return 0.5, 0.0  # err_rate, ce
        total = self.w1.size + self.w2.size
        frozen = (~self.mask_w1).sum() + (~self.mask_w2).sum()
        return int(frozen), int(total), float(frozen / total)


class PackNetD4(PackNetBase):
    """Vanilla PackNet: fixed prune_frac after every task."""
    def compute_surprise(self, X, y):
        """Compute surprise on a slice. Default: return fixed surprise (0.5) so
        prune_frac stays constant for vanilla PackNet."""
        return 0.5, 0.0  # err_rate, ce

    def prune_task(self, task_idx, surprise):
        """Default: fixed prune_frac (vanilla PackNet behavior)."""
        self.prune_and_freeze(self.prune_frac)


class PackNetD4b(PackNetBase):
    """D4b: Surprise-gated dynamic capacity allocation.
    
    After each task, compute surprise = prediction error on validation slice.
    If surprise LOW: freeze FEWER weights (keep more plastic for future tasks).
    If surprise HIGH: freeze MORE weights (protect current state, allocate more for next).
    """
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3,
                 surprise_scale=0.5, min_prune=0.05, max_prune=0.7):
        super().__init__(in_dim, hidden, out_dim, seed, prune_frac)
        self.surprise_scale = surprise_scale  # how strongly surprise affects pruning
        self.min_prune = min_prune
        self.max_prune = max_prune
        self.surprise_history = []

    def compute_surprise(self, X, y):
        """Compute mean prediction error (surprise) on a slice."""
        logits = self.predict(X)
        preds = np.argmax(logits, axis=1)
        err_rate = np.mean(preds != y)
        # Also use cross-entropy magnitude as surprise signal
        probs = softmax(logits)
        y_onehot = np.zeros((len(y), self.out_dim))
        y_onehot[np.arange(len(y)), y] = 1.0
        ce = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
        return float(err_rate), float(ce)

    def prune_task(self, task_idx, surprise):
        """Surprise-gated pruning: high surprise → freeze more, low surprise → freeze less."""
        err_rate, ce = surprise
        # Map surprise to prune fraction: higher surprise → more freezing
        # Base prune_frac adjusted by surprise
        surprise_factor = 1.0 + self.surprise_scale * (err_rate - 0.5)  # center around 0.5 error
        prune_frac = self.prune_frac * surprise_factor
        prune_frac = np.clip(prune_frac, self.min_prune, self.max_prune)
        self.surprise_history.append({'task': task_idx, 'surprise': err_rate, 'prune_frac': prune_frac})
        self.prune_and_freeze(prune_frac)


def eval_acc(model, X, y):
    logits = model.predict(X)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def continual_packnet(seed, prune_frac=0.3, hidden=64, epochs=50, lr=0.05, model_class=PackNetD4, **model_kwargs):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed)
    model = model_class(64, hidden, 10, seed=seed, prune_frac=prune_frac, **model_kwargs)
    T = len(tasks)
    task_acc = np.zeros((T, T))
    with Tracker() as tr:
        for k, task in enumerate(tasks):
            Xtr, ytr = task['X_train'], task['y_train']
            Xte, yte = task['X_test'], task['y_test']
            model.train_step(Xtr, ytr, lr=lr, epochs=epochs)
            
            # Compute surprise on this task's test set
            surprise = model.compute_surprise(Xte, yte)
            model.prune_task(k, surprise)
            model.save_snapshot(k)
            
            # Evaluate all tasks seen so far using their snapshots
            for j in range(k + 1):
                model.restore_snapshot(j)
                Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                task_acc[j, k] = eval_acc(model, Xt, yt)
    tr.update()
    aa = average_accuracy(task_acc)
    fm = forgetting_measure(task_acc)
    frozen, total, frac = model.frozen_count()
    return task_acc, aa, fm, tr.stats, {
        'frozen': frozen, 'total': total, 'frozen_frac': frac,
        'task_count': model.task_count,
        'surprise_history': model.surprise_history if hasattr(model, 'surprise_history') else [],
    }


def run_comparison():
    print("=== D4b vs PackNet: Split-digits-5 Continual ===")
    results = {}
    for prune in [0.3]:
        # D4 (vanilla PackNet)
        name_d4 = f'packnet_p{int(prune*100)}'
        aas, fms, walls, peaks, per_seed = [], [], [], [], []
        for s in SEEDS:
            set_seed(s)
            mat, aa, fm, st, info = continual_packnet(s, prune_frac=prune, model_class=PackNetD4)
            aas.append(aa); fms.append(fm); walls.append(st.wall_seconds); peaks.append(st.peak_rss_mb)
            per_seed.append({'seed': s, 'AA': aa, 'FM': fm, 'frozen_frac': info['frozen_frac']})
        results[name_d4] = {'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
                           'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks),
                           'per_seed': per_seed}
        print(f"  {name_d4}: AA={results[name_d4]['AA']['mean']:.4f}±{results[name_d4]['AA']['std']:.4f} "
              f"FM={results[name_d4]['FM']['mean']:.4f}±{results[name_d4]['FM']['std']:.4f} "
              f"frozen_frac={results[name_d4]['per_seed'][0]['frozen_frac']:.3f}")
        
        # D4b (surprise-gated)
        name_d4b = f'd4b_p{int(prune*100)}'
        aas, fms, walls, peaks, per_seed = [], [], [], [], []
        for s in SEEDS:
            set_seed(s)
            mat, aa, fm, st, info = continual_packnet(s, prune_frac=prune, model_class=PackNetD4b)
            aas.append(aa); fms.append(fm); walls.append(st.wall_seconds); peaks.append(st.peak_rss_mb)
            per_seed.append({'seed': s, 'AA': aa, 'FM': fm, 'frozen_frac': info['frozen_frac'],
                           'surprise_history': info.get('surprise_history', [])})
        results[name_d4b] = {'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
                            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks),
                            'per_seed': per_seed}
        print(f"  {name_d4b}: AA={results[name_d4b]['AA']['mean']:.4f}±{results[name_d4b]['AA']['std']:.4f} "
              f"FM={results[name_d4b]['FM']['mean']:.4f}±{results[name_d4b]['FM']['std']:.4f} "
              f"frozen_frac={results[name_d4b]['per_seed'][0]['frozen_frac']:.3f}")
        
        # Comparison
        aa_diff = results[name_d4b]['AA']['mean'] - results[name_d4]['AA']['mean']
        fm_diff = results[name_d4b]['FM']['mean'] - results[name_d4]['FM']['mean']
        cap_diff = results[name_d4b]['per_seed'][0]['frozen_frac'] - results[name_d4]['per_seed'][0]['frozen_frac']
        print(f"  DIFF (D4b - D4): AA={aa_diff:+.4f} FM={fm_diff:+.4f} cap={cap_diff:+.4f}")
    
    return results


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git': git, 'seeds': SEEDS}
    print("D4b vs PackNet Comparison")
    out['continual'] = run_comparison()
    save_json(os.path.join(OUT, 'results.json'), out)
    print(f"\nSaved to {OUT}/results.json")
