#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure
from src.candidates import HSAM, HSAM_Simplified
from src.baselines import (
    FuzzyART, GNGU, StreamingLDA, ReplayMLPTuned,
    ridge_readout_baseline, mlp_sklearn_baseline, random_feature_ridge,
    finetune_train_split_digits, ewc_train_split_digits, replay_train_split_digits
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "simplified_benchmark")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc_hsam(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def eval_acc_sklearn(model, X, y):
    try:
        p = model.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))
    except Exception:
        return float(model.score(X, y))


# ---------------- HSAM / HSAM_Simplified supervised ----------------
def hsam_supervised(cls, cfg):
    accs = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        (Xtr, ytr), (Xte, yte) = get_digits_split(seed=s)
        model = cls(input_dim=64, output_dim=10, seed=s, proj_dim=64, **cfg)
        with Tracker() as tr:
            for p in range(cfg.get('passes', 2)):
                idx = np.random.default_rng(s * 1000 + p).permutation(len(Xtr))
                for i in idx:
                    model.learn(Xtr[i], one_hot(ytr[i], 10))
            acc = eval_acc_hsam(model, Xte, yte)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'acc': acc})
    return {'per_seed': per_seed, 'acc': bootstrap_ci(accs),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


def hsam_continual(cls, cfg):
    aas = []; fms = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        model = cls(input_dim=64, output_dim=10, seed=s, proj_dim=64, **cfg)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                Xtr, ytr = task['X_train'], task['y_train']
                for p in range(cfg.get('passes', 2)):
                    idx = np.random.default_rng(s * 1000 + k * 10 + p).permutation(len(Xtr))
                    for i in idx:
                        model.learn(Xtr[i], one_hot(ytr[i], 10))
                for j in range(T):
                    Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                    task_acc[j, k] = eval_acc_hsam(model, Xt, yt)
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'AA': aa, 'FM': fm})
    return {'per_seed': per_seed, 'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


# ---------------- sklearn-style baselines supervised ----------------
def sklearn_supervised(fn):
    accs = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        (Xtr, ytr), (Xte, yte) = get_digits_split(seed=s)
        with Tracker() as tr:
            acc = fn(Xtr, ytr, Xte, yte, s)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'acc': acc})
    return {'per_seed': per_seed, 'acc': bootstrap_ci(accs),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


# ---------------- mechanistic baselines supervised ----------------
def proto_supervised(cls, kwargs):
    accs = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        (Xtr, ytr), (Xte, yte) = get_digits_split(seed=s)
        model = cls(input_dim=64, seed=s, **kwargs)
        with Tracker() as tr:
            model.fit(Xtr, ytr)
            acc = model.score(Xte, yte)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'acc': acc})
    return {'per_seed': per_seed, 'acc': bootstrap_ci(accs),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


# ---------------- continual baselines ----------------
def replay_continual(buffer_size):
    aas = []; fms = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        model = ReplayMLPTuned(in_dim=64, hidden=64, out_dim=10,
                               buffer_per_task=buffer_size, seed=s, epochs=15)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                model.train_task(task['X_train'], task['y_train'])
                for j in range(T):
                    Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                    task_acc[j, k] = float(np.mean(model.predict(Xt) == yt))
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'AA': aa, 'FM': fm})
    return {'per_seed': per_seed, 'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


def torch_continual(fn):
    aas = []; fms = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        try:
            import torch
            torch.manual_seed(s)
        except Exception:
            pass
        tasks = split_digits_tasks(seed=s)
        mat = None
        with Tracker() as tr:
            mat = fn(tasks, s)
        tr.update()
        aa = average_accuracy(mat); fm = forgetting_measure(mat)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'AA': aa, 'FM': fm})
    return {'per_seed': per_seed, 'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


def proto_continual(cls, kwargs):
    aas = []; fms = []; walls = []; peaks = []; per_seed = []
    for s in SEEDS:
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        model = cls(input_dim=64, seed=s, **kwargs)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                model.fit(task['X_train'], task['y_train'])
                for j in range(T):
                    Xt, yt = tasks[j]['X_test'], tasks[j]['y_test']
                    task_acc[j, k] = model.score(Xt, yt)
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed': s, 'AA': aa, 'FM': fm})
    return {'per_seed': per_seed, 'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms),
            'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git': git, 'seeds': SEEDS}

    sup = {}
    cont = {}

    # HSAM (original with tuner)
    print('=== HSAM original (self_tune=True) ===')
    sup['hsam_s200_w3_p2_tune'] = hsam_supervised(HSAM,
        dict(n_slots=200, n_winners=3, self_tune=True, eta_k=0.08, eta_v=0.3))
    cont['hsam_s200_w3_p2_tune'] = hsam_continual(HSAM,
        dict(n_slots=200, n_winners=3, self_tune=True, eta_k=0.08, eta_v=0.3))
    sup['hsam_s400_w3_p2_tune'] = hsam_supervised(HSAM,
        dict(n_slots=400, n_winners=3, self_tune=True, eta_k=0.05, eta_v=0.5))
    cont['hsam_s400_w3_p2_tune'] = hsam_continual(HSAM,
        dict(n_slots=400, n_winners=3, self_tune=True, eta_k=0.05, eta_v=0.5))

    # HSAM_Simplified (no tuner)
    print('=== HSAM_Simplified (no tuner) ===')
    sup['hsam_simplified_s200_w3'] = hsam_supervised(HSAM_Simplified,
        dict(n_slots=200, n_winners=3, eta_k=0.08, eta_v=0.3))
    cont['hsam_simplified_s200_w3'] = hsam_continual(HSAM_Simplified,
        dict(n_slots=200, n_winners=3, eta_k=0.08, eta_v=0.3))
    sup['hsam_simplified_s400_w3'] = hsam_supervised(HSAM_Simplified,
        dict(n_slots=400, n_winners=3, eta_k=0.05, eta_v=0.5))
    cont['hsam_simplified_s400_w3'] = hsam_continual(HSAM_Simplified,
        dict(n_slots=400, n_winners=3, eta_k=0.05, eta_v=0.5))

    # sklearn baselines supervised
    print('=== sklearn baselines supervised ===')
    sup['ridge'] = sklearn_supervised(lambda Xtr, ytr, Xte, yte, s:
        ridge_readout_baseline(Xtr, ytr, Xte, yte, seed=s, alpha=1.0)[0])
    sup['rf_ridge_256'] = sklearn_supervised(lambda Xtr, ytr, Xte, yte, s:
        random_feature_ridge(Xtr, ytr, Xte, yte, seed=s, n_features=256, alpha=1.0)[0])
    sup['mlp_sklearn_64'] = sklearn_supervised(lambda Xtr, ytr, Xte, yte, s:
        mlp_sklearn_baseline(Xtr, ytr, Xte, yte, seed=s, hidden=64, max_iter=200)[0])

    # continual baselines
    print('=== continual baselines ===')
    cont['finetune_mlp'] = torch_continual(lambda tasks, s:
        finetune_train_split_digits(tasks, epochs=15, lr=1e-3, seed=s))
    cont['ewc_mlp'] = torch_continual(lambda tasks, s:
        ewc_train_split_digits(tasks, epochs=15, lr=1e-3, ewc_lambda=50.0, seed=s))
    cont['replay_mlp_buf20'] = torch_continual(lambda tasks, s:
        replay_train_split_digits(tasks, epochs=15, lr=1e-3, buffer_per_task=20, seed=s))
    cont['replay_mlp_buf100'] = replay_continual(100)
    cont['replay_mlp_buf200'] = replay_continual(200)

    # mechanistic baselines supervised + continual
    print('=== mechanistic baselines ===')
    sup['fuzzy_art_rho07'] = proto_supervised(FuzzyART, dict(rho=0.7, beta=1.0))
    cont['fuzzy_art_rho07'] = proto_continual(FuzzyART, dict(rho=0.7, beta=1.0))
    sup['gngu'] = proto_supervised(GNGU, dict(max_nodes=200))
    cont['gngu'] = proto_continual(GNGU, dict(max_nodes=200))
    sup['slda'] = proto_supervised(StreamingLDA, dict())
    cont['slda'] = proto_continual(StreamingLDA, dict())

    out['supervised'] = sup
    out['continual'] = cont
    save_json(os.path.join(OUT, 'results.json'), out)

    print('\n=== SUPERVISED ===')
    for k, v in sup.items():
        print(f'{k}: acc={v["acc"]["mean"]:.4f}±{v["acc"]["std"]:.4f} wall={v["wall_s"]["mean"]:.3f}s rss={v["peak_rss_mb"]["mean"]:.1f}MB')
    print('\n=== CONTINUAL ===')
    for k, v in cont.items():
        print(f'{k}: AA={v["AA"]["mean"]:.4f}±{v["AA"]["std"]:.4f} FM={v["FM"]["mean"]:.4f}±{v["FM"]["std"]:.4f} wall={v["wall_s"]["mean"]:.3f}s rss={v["peak_rss_mb"]["mean"]:.1f}MB')
