#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.candidates import HSAM, RLS_SelfGate
from src.utils import bootstrap_ci, git_commit_hash, save_json, set_seed

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "sanity")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def test_overfit_32_hsam(seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(32, 16)).astype(np.float32)
    y = rng.integers(0, 4, size=(32,))
    m = HSAM(
        input_dim=16,
        output_dim=4,
        n_slots=64,
        n_winners=1,
        seed=seed,
        proj_dim=16,
        eta_k=0.1,
        eta_v=0.8,
        self_tune=True,
    )
    for ep in range(50):
        idx = rng.permutation(32)
        for i in idx:
            m.learn(X[i], one_hot(y[i], 4))
    preds = [int(np.argmax(m.predict(X[i])[0])) for i in range(32)]
    return float((np.array(preds) == y).mean())


def test_overfit_32_rls(seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(32, 16)).astype(np.float32)
    y = rng.integers(0, 4, size=(32,))
    m = RLS_SelfGate(input_dim=16, output_dim=4, n_features=64, seed=seed, lam=1.0, forget=0.995)
    for ep in range(20):
        idx = rng.permutation(32)
        for i in idx:
            m.learn(X[i], one_hot(y[i], 4))
    preds = [m.predict_class(X[i]) for i in range(32)]
    return float((np.array(preds) == y).mean())


if __name__ == "__main__":
    git = git_commit_hash(ROOT)
    res = {}
    for name, fn in [("hsam_overfit32", test_overfit_32_hsam), ("rls_overfit32", test_overfit_32_rls)]:
        vals = [fn(s) for s in SEEDS]
        res[name] = {
            "per_seed": [{"seed": s, "acc": v} for s, v in zip(SEEDS, vals)],
            "acc": bootstrap_ci(vals),
        }
        print(name, res[name]["acc"])
    save_json(os.path.join(OUT, "overfit32_results.json"), {"git": git, "seeds": SEEDS, "results": res})
