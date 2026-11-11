#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.candidates import HSAM
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure
from src.utils import Tracker, bootstrap_ci, git_commit_hash, save_json, set_seed

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "hsam_robustness")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def continual_with_pairs(seed, pairs, n_slots=200, n_winners=3, passes=2):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed, pairs=pairs)
    model = HSAM(input_dim=64, output_dim=10, n_slots=n_slots, n_winners=n_winners, seed=seed,
                 proj_dim=64, eta_k=0.08, eta_v=0.3, self_tune=True)
    T = len(tasks)
    task_acc = np.zeros((T, T))
    with Tracker() as tr:
        for k, task in enumerate(tasks):
            Xtr, ytr = task["X_train"], task["y_train"]
            for p in range(passes):
                idx = np.random.default_rng(seed * 1000 + k * 10 + p).permutation(len(Xtr))
                for i in idx:
                    model.learn(Xtr[i], one_hot(ytr[i], 10))
            for j in range(T):
                Xt, yt = tasks[j]["X_test"], tasks[j]["y_test"]
                task_acc[j, k] = eval_acc(model, Xt, yt)
    tr.update()
    return task_acc, average_accuracy(task_acc), forgetting_measure(task_acc), tr.stats


def eval_probe(model, Xte, yte):
    preds = [int(np.argmax(model.predict(Xte[i])[0])) for i in range(len(Xte))]
    return float((np.array(preds) == yte).mean())


def run_stream(seed, mode="full", chunks=100, chunk_size=200, frozen_after=None):
    set_seed(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed)
    rng = np.random.default_rng(seed)
    model = HSAM(input_dim=64, output_dim=10, n_slots=200, n_winners=3, seed=seed,
                 proj_dim=64, eta_k=0.08, eta_v=0.25, self_tune=(mode in ["full", "no_alloc"]))
    accs = []
    total = 0
    with Tracker() as tr:
        for c in range(chunks):
            idx = rng.integers(0, len(Xtr), size=chunk_size)
            frozen = (frozen_after is not None and c >= frozen_after)
            if not frozen:
                for i in idx:
                    x = Xtr[i]
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
            if c in [0, 4, 9, 19, 49, 99] or c == chunks - 1:
                accs.append({
                    "chunk": c,
                    "examples_seen": total,
                    "acc": eval_probe(model, Xte, yte),
                    "err_ema": float(model.err_ema),
                    "allocations": int(model.allocations),
                    "filled": int(model.filled.sum()),
                })
    tr.update()
    return accs, tr.stats


if __name__ == "__main__":
    git = git_commit_hash(ROOT)
    out = {"git": git, "seeds": SEEDS}

    # Task-order robustness for continual learning
    orders = {
        "default_01_23_45_67_89": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
        "reverse_89_67_45_23_01": [(8, 9), (6, 7), (4, 5), (2, 3), (0, 1)],
        "interleaved_05_16_27_38_49": [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)],
    }
    cont = {}
    for name, pairs in orders.items():
        aas = []
        fms = []
        walls = []
        per_seed = []
        for s in SEEDS:
            mat, aa, fm, st = continual_with_pairs(s, pairs=pairs)
            aas.append(aa)
            fms.append(fm)
            walls.append(st.wall_seconds)
            per_seed.append({"seed": s, "AA": aa, "FM": fm, "wall_s": st.wall_seconds, "matrix": mat.tolist()})
        cont[name] = {"pairs": pairs, "per_seed": per_seed, "AA": bootstrap_ci(aas), "FM": bootstrap_ci(fms), "wall_s": bootstrap_ci(walls)}
        print("ORDER", name, "AA", cont[name]["AA"]["mean"], "FM", cont[name]["FM"]["mean"])
    out["continual_task_orders"] = cont

    # Longer stream probe (100 chunks = 20000 examples, 10x prior chunks)
    stream = {}
    variants = [
        ("full", "full", None),
        ("frozen_after_1", "full", 1),
        ("frozen_after_5", "full", 5),
        ("no_tune", "no_tune", None),
        ("no_alloc_after_fill_behavior", "no_alloc", None),
    ]
    for name, mode, frozen_after in variants:
        per_seed = []
        first = []
        last = []
        deltas = []
        for s in SEEDS:
            curve, st = run_stream(s, mode=mode, chunks=100, chunk_size=200, frozen_after=frozen_after)
            per_seed.append({"seed": s, "curve": curve, "wall_s": st.wall_seconds, "peak_rss_mb": st.peak_rss_mb})
            first.append(curve[0]["acc"])
            last.append(curve[-1]["acc"])
            deltas.append(curve[-1]["acc"] - curve[0]["acc"])
        stream[name] = {
            "mode": mode,
            "frozen_after": frozen_after,
            "per_seed": per_seed,
            "acc_first": bootstrap_ci(first),
            "acc_last": bootstrap_ci(last),
            "delta": bootstrap_ci(deltas),
        }
        print("STREAM", name, "first", stream[name]["acc_first"]["mean"], "last", stream[name]["acc_last"]["mean"], "delta", stream[name]["delta"]["mean"])
    out["stream_long"] = stream

    save_json(os.path.join(OUT, "results.json"), out)
    print("saved", OUT)
