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
OUT = os.path.join(ROOT, "research", "03_experiments", "hsam_ablations")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def supervised(seed, **cfg):
    set_seed(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed)
    model = HSAM(seed=seed, **cfg)
    with Tracker() as tr:
        passes = cfg.get("passes", 2)
        for p in range(passes):
            idx = np.random.default_rng(seed * 1000 + p).permutation(len(Xtr))
            for i in idx:
                model.learn(Xtr[i], one_hot(ytr[i], 10))
        acc = eval_acc(model, Xte, yte)
    tr.update()
    info = {"allocations": int(model.allocations), "filled": int(model.filled.sum()), "err_ema": float(model.err_ema)}
    return acc, tr.stats, info


def continual(seed, **cfg):
    set_seed(seed)
    tasks = split_digits_tasks(seed=seed)
    model = HSAM(seed=seed, **cfg)
    T = len(tasks)
    task_acc = np.zeros((T, T))
    with Tracker() as tr:
        passes = cfg.get("passes", 2)
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
    return task_acc, average_accuracy(task_acc), forgetting_measure(task_acc), tr.stats, {"allocations": int(model.allocations), "filled": int(model.filled.sum())}


def summarize_supervised(name, cfg):
    accs=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        # remove passes from HSAM ctor if present
        ctor = {k:v for k,v in cfg.items() if k != 'passes'}
        passes = cfg.get('passes', 2)
        set_seed(s)
        (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed if False else s)
        model = HSAM(seed=s, **ctor)
        with Tracker() as tr:
            for p in range(passes):
                idx = np.random.default_rng(s*1000+p).permutation(len(Xtr))
                for i in idx:
                    model.learn(Xtr[i], one_hot(ytr[i],10))
            acc = eval_acc(model, Xte, yte)
        tr.update()
        info={"allocations":int(model.allocations),"filled":int(model.filled.sum()),"err_ema":float(model.err_ema)}
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed":s,"acc":acc,"wall_s":tr.stats.wall_seconds,"peak_rss_mb":tr.stats.peak_rss_mb,"info":info})
    return {"config":cfg,"per_seed":per_seed,"acc":bootstrap_ci(accs),"wall_s":bootstrap_ci(walls),"peak_rss_mb":bootstrap_ci(peaks)}


def summarize_continual(name, cfg):
    aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        ctor = {k:v for k,v in cfg.items() if k != 'passes'}
        passes = cfg.get('passes', 2)
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        model = HSAM(seed=s, **ctor)
        T=len(tasks)
        task_acc=np.zeros((T,T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                Xtr,ytr=task['X_train'],task['y_train']
                for p in range(passes):
                    idx=np.random.default_rng(s*1000+k*10+p).permutation(len(Xtr))
                    for i in idx:
                        model.learn(Xtr[i], one_hot(ytr[i],10))
                for j in range(T):
                    Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                    task_acc[j,k]=eval_acc(model,Xt,yt)
        tr.update()
        aa=average_accuracy(task_acc); fm=forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed":s,"AA":aa,"FM":fm,"matrix":task_acc.tolist(),"wall_s":tr.stats.wall_seconds,"peak_rss_mb":tr.stats.peak_rss_mb,"info":{"allocations":int(model.allocations),"filled":int(model.filled.sum())}})
    return {"config":cfg,"per_seed":per_seed,"AA":bootstrap_ci(aas),"FM":bootstrap_ci(fms),"wall_s":bootstrap_ci(walls),"peak_rss_mb":bootstrap_ci(peaks)}


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    base = dict(input_dim=64, output_dim=10, proj_dim=64, eta_k=0.08, eta_v=0.3, n_slots=200, n_winners=3, passes=2, self_tune=True)
    sup_configs = {
        'base_s200_w3': dict(base),
        'no_tune_s200_w3': {**base, 'self_tune': False},
        'w1_s200': {**base, 'n_winners': 1},
        'slots100_w3': {**base, 'n_slots': 100},
        'slots400_w3': {**base, 'n_slots': 400},
        'eta_low_s200_w3': {**base, 'eta_k': 0.03, 'eta_v': 0.15},
    }
    sup = {}
    for name,cfg in sup_configs.items():
        sup[name]=summarize_supervised(name,cfg)
        print('SUP', name, sup[name]['acc']['mean'])
    cont_configs = {
        'base_s200_w3': dict(base),
        'no_tune_s200_w3': {**base, 'self_tune': False},
        'w1_s200': {**base, 'n_winners': 1},
        'slots100_w3': {**base, 'n_slots': 100},
        'slots400_w3': {**base, 'n_slots': 400},
    }
    cont = {}
    for name,cfg in cont_configs.items():
        cont[name]=summarize_continual(name,cfg)
        print('CONT', name, cont[name]['AA']['mean'], cont[name]['FM']['mean'])
    save_json(os.path.join(OUT,'results.json'), {'git':git,'seeds':SEEDS,'supervised':sup,'continual':cont})
    print('done')
