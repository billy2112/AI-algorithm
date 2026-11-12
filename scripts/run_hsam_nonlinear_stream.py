#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.candidates import HSAM
from src.data import get_digits_split
from src.utils import Tracker, bootstrap_ci, git_commit_hash, save_json, set_seed

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "hsam_nonlinear_stream")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_probe(model, Xte, yte):
    preds = [int(np.argmax(model.predict(Xte[i])[0])) for i in range(len(Xte))]
    return float((np.array(preds) == yte).mean())


def corrupt_x(x, rng, level=0.3):
    x = np.array(x, dtype=np.float32).copy()
    noise = rng.normal(0, level, size=x.shape).astype(np.float32)
    x = x + noise
    x = np.clip(x, 0, 1)
    # random dropout of pixels
    mask = rng.random(size=x.shape) > 0.2
    x = x * mask
    return x


def run_variant(seed, mode="full", chunks=60, chunk_size=200):
    set_seed(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    model = HSAM(
        input_dim=64,
        output_dim=10,
        n_slots=200,
        n_winners=3,
        seed=seed,
        proj_dim=64,
        eta_k=0.08,
        eta_v=0.25,
        self_tune=(mode in ["full", "no_alloc"]),
    )
    accs = []
    total = 0
    with Tracker() as tr:
        for c in range(chunks):
            idx = rng.integers(0, len(Xtr), size=chunk_size)
            if mode == "frozen_after_1" and c >= 1:
                pass
            else:
                for j, i in enumerate(idx):
                    x = Xtr[i]
                    # introduce distribution shifts at different points
                    if c >= 10 and c < 30:  # noise regime
                        x = corrupt_x(x, rng, level=0.15 + 0.1 * ((c - 10) / 20.0))
                    elif c >= 30 and c < 40:  # stronger corruption + class shift preference
                        x = corrupt_x(x, rng, level=0.25)
                        # bias towards different classes
                        if rng.random() < 0.6:
                            y_shift = (int(ytr[i]) + 5) % 10
                            y = one_hot(y_shift, 10)
                        else:
                            y = one_hot(ytr[i], 10)
                    elif c >= 40:  # new corruption pattern + rotation-like mixing
                        x = x * rng.uniform(0.7, 1.3)
                        x = np.clip(x + rng.normal(0, 0.05, size=x.shape), 0, 1).astype(np.float32)
                        y = one_hot(ytr[i], 10)
                    else:
                        y = one_hot(ytr[i], 10)
                    if mode == "no_alloc" and np.any(model.filled):
                        old = model.v_high
                        model.v_high = 1e9
                        model.learn(x, y)
                        model.v_high = old
                    elif mode == "no_tune":
                        old = model.self_tune
                        model.self_tune = False
                        model.learn(x, y)
                        model.self_tune = old
                    else:
                        model.learn(x, y)
            total += chunk_size
            if c in [0, 5, 9, 14, 19, 29, 34, 39, 44, 49, 59]:
                accs.append(
                    {
                        "chunk": c,
                        "examples_seen": total,
                        "acc": eval_probe(model, Xte, yte),
                        "err_ema": float(model.err_ema),
                        "allocations": int(model.allocations),
                        "filled": int(model.filled.sum()),
                    }
                )
    tr.update()
    return accs, tr.stats


if __name__ == "__main__":
    git = git_commit_hash(ROOT)
    results = {}
    for mode in ["full", "frozen_after_1", "no_tune", "no_alloc"]:
        per_seed = []
        first = []
        last = []
        deltas = []
        peak_accs = []
        for s in SEEDS:
            curve, st = run_variant(s, mode=mode, chunks=60, chunk_size=200)
            per_seed.append(
                {
                    "seed": s,
                    "curve": curve,
                    "wall_s": st.wall_seconds,
                    "peak_rss_mb": st.peak_rss_mb,
                }
            )
            first.append(curve[0]["acc"])
            last.append(curve[-1]["acc"])
            deltas.append(curve[-1]["acc"] - curve[0]["acc"])
            peak_accs.append(max([c["acc"] for c in curve]))
        results[mode] = {
            "per_seed": per_seed,
            "acc_first": bootstrap_ci(first),
            "acc_last": bootstrap_ci(last),
            "delta": bootstrap_ci(deltas),
            "acc_peak": bootstrap_ci(peak_accs),
        }
    save_json(os.path.join(OUT, "results.json"), {"git": git, "seeds": SEEDS, "results": results})
    for k, v in results.items():
        print(k, "first", v["acc_first"]["mean"], "last", v["acc_last"]["mean"], "delta", v["delta"]["mean"], "peak", v["acc_peak"]["mean"])
