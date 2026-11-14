#!/usr/bin/env python3
"""
Half-day sanity pass for D1-D4 candidates.
Tests on: overfit-32 (toy), supervised digits (sklearn load_digits, 5 seeds).
Kills anything that can't beat simple baselines (ridge, small MLP).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from src.utils import set_seed, bootstrap_ci


# ============ D1: Learned Optimizer (meta-learned update rule) ============
class D1LearnedOptimizer:
    """Small meta-learned MLP that takes (grad, loss, lr) and outputs update multiplier.
    Simplified sanity version: meta-net maps 3 features -> 1 scalar multiplier.
    """
    def __init__(self, in_dim, hidden, out_dim, seed=0, meta_lr=1e-3):
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.standard_normal((in_dim, hidden)) * 0.1
        self.w2 = self.rng.standard_normal((hidden, out_dim)) * 0.1
        # meta-net: 3 inputs -> 1 output (update multiplier)
        self.mw1 = self.rng.standard_normal((3, 8)) * 0.1
        self.mw2 = self.rng.standard_normal((8, 1)) * 0.1

    def _relu(self, x): return np.maximum(0, x)

    def predict(self, x):
        h = self._relu(x @ self.w1)
        return h @ self.w2

    def meta_forward(self, g, loss_val, lr):
        inp = np.array([g, loss_val, lr])
        h = self._relu(inp @ self.mw1)
        val = h @ self.mw2
        return float(np.asarray(val).ravel()[0])

    def train_step(self, x, y, lr):
        # Forward
        h = self._relu(x @ self.w1)
        out = h @ self.w2
        # Loss (MSE for classification target one-hot)
        y_onehot = np.zeros(self.w2.shape[1])
        y_onehot[int(y)] = 1.0
        err = out - y_onehot
        loss = float(np.sum(err**2))
        # Gradients
        d_out = err
        d_w2 = np.outer(h, d_out)
        d_h = d_out @ self.w2.T * (h > 0)
        d_w1 = x[:, None] @ d_h[None, :]
        # Meta-learned multiplier per parameter block
        g_w2 = float(np.sum(d_w2**2)**0.5)
        g_w1 = float(np.sum(d_w1**2)**0.5)
        m_w2 = self.meta_forward(g_w2, loss, lr)
        m_w1 = self.meta_forward(g_w1, loss, lr)
        # Apply updates with meta multiplier
        self.w2 -= lr * m_w2 * d_w2
        self.w1 -= lr * m_w1 * d_w1
        return loss


# ============ D2: Predictive Coding (iPC-style, minimal 2-layer) ============
class D2PredictiveCoding:
    """2-layer generative model. Inference: settle hidden activity to minimize
    energy = ||x - h@w2.T||^2 + ||h - x@w1||^2. Learn: local Hebbian updates."""
    def __init__(self, in_dim, hidden, out_dim, seed=0, infer_iters=20, infer_lr=0.1):
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.standard_normal((in_dim, hidden)) * 0.1  # input -> hidden
        self.w2 = self.rng.standard_normal((hidden, out_dim)) * 0.1  # hidden -> output
        self.infer_iters = infer_iters
        self.infer_lr = infer_lr

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            h = self._infer(X)
            return h @ self.w2
        return np.array([self._infer(xi) @ self.w2 for xi in X])

    def _infer(self, x):
        h = np.zeros(self.w1.shape[1])
        for _ in range(self.infer_iters):
            h_pred_from_x = x @ self.w1
            e_h = h - h_pred_from_x
            h = h - self.infer_lr * e_h
            # stability clamp
            h = np.clip(h, -10, 10)
        return h

    def train_step(self, x, y, lr):
        h = self._infer(x)
        # Local Hebbian update on w1: h <- x
        d_w1 = np.outer(x, (h - x @ self.w1))  # (in_dim, hidden)
        self.w1 -= lr * 0.1 * d_w1
        # Output: classify h via w2, update w2 on classification error
        out = h @ self.w2
        y_onehot = np.zeros(self.w2.shape[1])
        y_onehot[int(y)] = 1.0
        err = out - y_onehot
        self.w2 -= lr * np.outer(h, err)
        return float(np.sum(err**2))


# ============ D3: Sparse MoE Routing Networks (minimal) ============
class D3SparseMoE:
    """Fixed backbone with N experts + router. Router selects top-1 expert per input.
    Co-training: unused experts also get gradient steps."""
    def __init__(self, in_dim, hidden, out_dim, n_experts=4, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_experts = n_experts
        self.experts = []
        for _ in range(n_experts):
            self.experts.append({
                'w1': self.rng.standard_normal((in_dim, hidden)) * 0.1,
                'w2': self.rng.standard_normal((hidden, out_dim)) * 0.1
            })
        self.router_w = self.rng.standard_normal((in_dim, n_experts)) * 0.1

    def _expert_forward(self, x, e):
        h = np.maximum(0, x @ self.experts[e]['w1'])
        return h @ self.experts[e]['w2']

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            logits = X @ self.router_w
            best = int(np.argmax(logits))
            return self._expert_forward(X, best)
        logits = X @ self.router_w  # (batch, n_experts)
        bests = np.argmax(logits, axis=1)
        outs = []
        for xi, b in zip(X, bests):
            outs.append(self._expert_forward(xi, b))
        return np.array(outs)

    def train_step(self, x, y, lr):
        y_onehot = np.zeros(self.experts[0]['w2'].shape[1])
        y_onehot[int(y)] = 1.0
        logits = x @ self.router_w
        # Router probabilities (softmax)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        best = int(np.argmax(logits))

        # Forward all experts, compute loss for each
        losses = []
        for e in range(self.n_experts):
            out = self._expert_forward(x, e)
            losses.append(float(np.sum((out - y_onehot)**2)))

        # Update selected expert
        out_best = self._expert_forward(x, best)
        err_best = out_best - y_onehot
        h_best = np.maximum(0, x @ self.experts[best]['w1'])
        self.experts[best]['w2'] -= lr * np.outer(h_best, err_best)
        d_h_best = err_best @ self.experts[best]['w2'].T * (h_best > 0)
        self.experts[best]['w1'] -= lr * np.outer(x, d_h_best)

        # Co-training: update unused experts lightly
        for e in range(self.n_experts):
            if e == best:
                continue
            out_e = self._expert_forward(x, e)
            err_e = out_e - y_onehot
            h_e = np.maximum(0, x @ self.experts[e]['w1'])
            self.experts[e]['w2'] -= lr * 0.1 * np.outer(h_e, err_e)
            d_h_e = err_e @ self.experts[e]['w2'].T * (h_e > 0)
            self.experts[e]['w1'] -= lr * 0.1 * np.outer(x, d_h_e)

        # Update router (reinforce best expert if it had low loss)
        router_grad = np.zeros_like(self.router_w)
        # Simple: router gradient pushes logits of best expert up if its loss is low
        if losses[best] < np.mean(losses):
            for e in range(self.n_experts):
                if e == best:
                    self.router_w[:, e] += lr * 0.05 * x
                else:
                    self.router_w[:, e] -= lr * 0.05 * x
        return losses[best]


# ============ D4: PackNet Parameter Isolation ============
class D4PackNet:
    """Fixed backbone MLP. After each task, prune bottom p% weights by magnitude,
    freeze them. Next task trains only unpruned weights."""
    def __init__(self, in_dim, hidden, out_dim, seed=0, prune_frac=0.3):
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.standard_normal((in_dim, hidden)) * 0.1
        self.w2 = self.rng.standard_normal((hidden, out_dim)) * 0.1
        self.mask1 = np.ones_like(self.w1, dtype=bool)
        self.mask2 = np.ones_like(self.w2, dtype=bool)
        self.prune_frac = prune_frac

    def predict(self, x):
        h = np.maximum(0, x @ self.w1)
        return h @ self.w2

    def train_step(self, x, y, lr):
        y_onehot = np.zeros(self.w2.shape[1])
        y_onehot[int(y)] = 1.0
        h = np.maximum(0, x @ self.w1)
        out = h @ self.w2
        err = out - y_onehot
        # Gradients
        d_w2 = np.outer(h, err)
        d_h = err @ self.w2.T * (h > 0)
        d_w1 = np.outer(x, d_h)
        # Apply mask (only update unpruned)
        self.w2 -= lr * d_w2 * self.mask2
        self.w1 -= lr * d_w1 * self.mask1
        return float(np.sum(err**2))

    def prune_and_freeze(self):
        """Prune bottom prune_frac of weights across both layers."""
        all_w = np.concatenate([self.w1.flatten(), self.w2.flatten()])
        threshold = np.percentile(np.abs(all_w), self.prune_frac * 100)
        m1 = np.abs(self.w1) > threshold
        m2 = np.abs(self.w2) > threshold
        # Ensure at least some weights remain
        if m1.sum() == 0: m1[:] = True
        if m2.sum() == 0: m2[:] = True
        self.mask1 = self.mask1 & m1
        self.mask2 = self.mask2 & m2


# ============ Baselines ============
def ridge_acc(Xtr, ytr, Xte, yte):
    clf = RidgeClassifier(alpha=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))

def mlp_acc(Xtr, ytr, Xte, yte):
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=0)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


# ============ Tests ============
def test_overfit32():
    """Can the candidate overfit 32 random samples?"""
    set_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 16)).astype(np.float32)
    y = rng.integers(0, 4, size=32)
    results = {}

    # D1
    set_seed(1)
    m = D1LearnedOptimizer(16, 32, 4, seed=1)
    for ep in range(200):
        idx = rng.permutation(32)
        for i in idx:
            m.train_step(X[i], y[i], lr=0.01)
    preds = np.argmax(m.predict(X), axis=1)
    results['D1'] = float(np.mean(preds == y))

    # D2
    set_seed(2)
    m = D2PredictiveCoding(16, 32, 4, seed=2, infer_iters=10)
    for ep in range(200):
        idx = rng.permutation(32)
        for i in idx:
            m.train_step(X[i], y[i], lr=0.01)
    preds = np.argmax(m.predict(X), axis=1)
    results['D2'] = float(np.mean(preds == y))

    # D3
    set_seed(3)
    m = D3SparseMoE(16, 32, 4, n_experts=4, seed=3)
    for ep in range(200):
        idx = rng.permutation(32)
        for i in idx:
            m.train_step(X[i], y[i], lr=0.01)
    preds = np.argmax(m.predict(X), axis=1)
    results['D3'] = float(np.mean(preds == y))

    # D4
    set_seed(4)
    m = D4PackNet(16, 32, 4, seed=4, prune_frac=0.3)
    for ep in range(100):
        idx = rng.permutation(32)
        for i in idx:
            m.train_step(X[i], y[i], lr=0.01)
        if ep % 25 == 24:
            m.prune_and_freeze()
    preds = np.argmax(m.predict(X), axis=1)
    results['D4'] = float(np.mean(preds == y))

    # Baselines
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0)
    clf.fit(X, y)
    results['MLP_baseline'] = float(clf.score(X, y))

    return results


def test_digits():
    """Supervised digits, 5 seeds."""
    results = {}
    for name, fn in [('D1', lambda: D1LearnedOptimizer(64, 64, 10, seed=s)),
                     ('D2', lambda: D2PredictiveCoding(64, 64, 10, seed=s)),
                     ('D3', lambda: D3SparseMoE(64, 64, 10, n_experts=4, seed=s)),
                     ('D4', lambda: D4PackNet(64, 64, 10, seed=s, prune_frac=0.3))]:
        accs = []
        for s in range(5):
            set_seed(s)
            m = fn()
            d = load_digits()
            X = d.data / 16.0
            y = d.target.astype(int)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
            for ep in range(15):
                idx = np.random.default_rng(s * 1000 + ep).permutation(len(Xtr))
                for i in idx:
                    m.train_step(Xtr[i], ytr[i], lr=0.01)
            preds = np.argmax(m.predict(Xte), axis=1)
            accs.append(float(np.mean(preds == yte)))
        results[name] = (float(np.mean(accs)), float(np.std(accs)))

    # Baselines (5 seeds)
    ridge_accs, mlp_accs = [], []
    for s in range(5):
        set_seed(s)
        d = load_digits()
        X = d.data / 16.0
        y = d.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        ridge_accs.append(ridge_acc(Xtr, ytr, Xte, yte))
        mlp_accs.append(mlp_acc(Xtr, ytr, Xte, yte))
    results['Ridge'] = (float(np.mean(ridge_accs)), float(np.std(ridge_accs)))
    results['MLP'] = (float(np.mean(mlp_accs)), float(np.std(mlp_accs)))
    return results


if __name__ == '__main__':
    print("=== OVERFIT-32 SANITY ===")
    r32 = test_overfit32()
    for k, v in r32.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("=== SUPERVISED DIGITS (5 seeds) ===")
    rd = test_digits()
    for k, v in rd.items():
        print(f"  {k}: {v[0]:.4f} ± {v[1]:.4f}")
