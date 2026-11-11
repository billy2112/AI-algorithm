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
OUT = os.path.join(ROOT, "research", "03_experiments", "hsam_digits")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def supervised(seed, n_slots=400, n_winners=1, passes=2, proj_dim=64, eta_k=0.05, eta_v=0.5, self_tune=True):
    set_seed(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed)
    model = HSAM(
        input_dim=64,
        output_dim=10,
        n_slots=n_slots,
        n_winners=n_winners,
        seed=seed,
        proj_dim=proj_dim,
        eta_k=eta_k,
        eta_v=eta_v,
        self_tune=self_tune,
    )
    with Tracker() as tr:
        for p in range(passes):
            idx = np.random.default_rng(seed * 1000 + p).permutation(len(Xtr))
            for i in idx:
                model.learn(Xtr[i], one_hot(ytr[i], 10))
        acc = eval_acc(model, Xte, yte)
    tr.update()
    info = {
        "allocations": int(model.allocations),
        "filled": int(model.filled.sum()),
        "err_ema": float(model.err_ema),
        "step": int(model.step),
    }
    return acc, tr.stats, info


def continual(seed, n_slots=200, n_winners=3, passes=2, proj_dim=64, eta_k=0.08, eta_v=0.3, self_tune=True):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed)
    model = HSAM(
        input_dim=64,
        output_dim=10,
        n_slots=n_slots,
        n_winners=n_winners,
        seed=seed,
        proj_dim=proj_dim,
        eta_k=eta_k,
        eta_v=eta_v,
        self_tune=self_tune,
    )
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
    aa = average_accuracy(task_acc)
    fm = forgetting_measure(task_acc)
    info = {
        "allocations": int(model.allocations),
        "filled": int(model.filled.sum()),
        "err_ema": float(model.err_ema),
    }
    return task_acc, aa, fm, tr.stats, info


def run_supervised_suite():
    configs = {
        "hsam_s400_w1_p2_tune": dict(n_slots=400, n_winners=1, passes=2, self_tune=True),
        "hsam_s400_w3_p2_tune": dict(n_slots=400, n_winners=3, passes=2, self_tune=True),
        "hsam_s200_w3_p2_tune": dict(n_slots=200, n_winners=3, passes=2, self_tune=True),
        "hsam_s400_w1_p2_notune": dict(n_slots=400, n_winners=1, passes=2, self_tune=False),
    }
    out = {}
    for name, cfg in configs.items():
        accs = []
        walls = []
        peaks = []
        per_seed = []
        for s in SEEDS:
            acc, st, info = supervised(s, **cfg)
            accs.append(acc)
            walls.append(st.wall_seconds)
            peaks.append(st.peak_rss_mb)
            per_seed.append(
                {
                    "seed": s,
                    "acc": acc,
                    "wall_s": st.wall_seconds,
                    "peak_rss_mb": st.peak_rss_mb,
                    "info": info,
                }
            )
        out[name] = {
            "config": cfg,
            "per_seed": per_seed,
            "acc": bootstrap_ci(accs),
            "wall_s": bootstrap_ci(walls),
            "peak_rss_mb": bootstrap_ci(peaks),
        }
    return out


def run_continual_suite():
    configs = {
        "hsam_s200_w3_p2_tune": dict(n_slots=200, n_winners=3, passes=2, self_tune=True),
        "hsam_s200_w3_p2_notune": dict(n_slots=200, n_winners=3, passes=2, self_tune=False),
        "hsam_s100_w3_p2_tune": dict(n_slots=100, n_winners=3, passes=2, self_tune=True),
        "hsam_s400_w3_p2_tune": dict(n_slots=400, n_winners=3, passes=2, self_tune=True),
    }
    out = {}
    for name, cfg in configs.items():
        aas = []
        fms = []
        walls = []
        peaks = []
        per_seed = []
        for s in SEEDS:
            mat, aa, fm, st, info = continual(s, **cfg)
            aas.append(aa)
            fms.append(fm)
            walls.append(st.wall_seconds)
            peaks.append(st.peak_rss_mb)
            per_seed.append(
                {
                    "seed": s,
                    "AA": aa,
                    "FM": fm,
                    "matrix": mat.tolist(),
                    "wall_s": st.wall_seconds,
                    "peak_rss_mb": st.peak_rss_mb,
                    "info": info,
                }
            )
        out[name] = {
            "config": cfg,
            "per_seed": per_seed,
            "AA": bootstrap_ci(aas),
            "FM": bootstrap_ci(fms),
            "wall_s": bootstrap_ci(walls),
            "peak_rss_mb": bootstrap_ci(peaks),
        }
    return out


if __name__ == "__main__":
    git = git_commit_hash(ROOT)
    sup = run_supervised_suite()
    save_json(os.path.join(OUT, "supervised_results.json"), {"git": git, "seeds": SEEDS, "results": sup})
    cont = run_continual_suite()
    save_json(os.path.join(OUT, "continual_results.json"), {"git": git, "seeds": SEEDS, "results": cont})
    print("SUPERVISED")
    for k, v in sup.items():
        print(k, v["acc"]["mean"], "+/-", v["acc"]["std"], "wall", v["wall_s"]["mean"])
    print("CONTINUAL")
    for k, v in cont.items():
        print(k, "AA", v["AA"]["mean"], "FM", v["FM"]["mean"], "wall", v["wall_s"]["mean"])
