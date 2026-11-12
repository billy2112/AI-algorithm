#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure
from src.candidates import HSAM

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "hsam_grid")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]  # 3 seeds


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def run_supervised():
    results = {}
    slots_list = [100, 200, 400, 800]
    winners_list = [1, 3, 5]
    for slots in slots_list:
        for winners in winners_list:
            for tune_mode in ['full', 'no_tune', 'no_alloc']:
                cfg_name = f's{slots}_w{winners}_{tune_mode}'
                accs = []; walls=[]; peaks=[]
                per_seed = []
                for s in SEEDS:
                    set_seed(s)
                    (Xtr_s, ytr_s), (Xte_s, yte_s) = get_digits_split(seed=s)
                    self_tune = tune_mode in ['full']
                    model = HSAM(input_dim=64, output_dim=10, n_slots=slots, n_winners=winners, seed=s, 
                                proj_dim=64, eta_k=0.08, eta_v=0.3, self_tune=self_tune)
                    with Tracker() as tr:
                        for p in range(2):
                            idx = np.random.default_rng(s*1000+p).permutation(len(Xtr_s))
                            for i in idx:
                                x = Xtr_s[i]
                                y = one_hot(ytr_s[i],10)
                                if tune_mode == 'no_alloc' and np.any(model.filled):
                                    old = model.v_high
                                    model.v_high = 1e9
                                    model.learn(x, y)
                                    model.v_high = old
                                elif tune_mode == 'no_tune':
                                    old = model.self_tune
                                    model.self_tune = False
                                    model.learn(x, y)
                                    model.self_tune = old
                                else:
                                    model.learn(x, y)
                        acc = eval_acc(model, Xte_s, yte_s)
                    tr.update()
                    accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
                    per_seed.append({'seed':s,'acc':acc,'allocations':int(model.allocations),'filled':int(model.filled.sum())})
                results[cfg_name] = {'slots':slots,'winners':winners,'mode':tune_mode,'per_seed':per_seed, 
                                     'acc':bootstrap_ci(accs), 'wall_s':bootstrap_ci(walls), 'peak_rss_mb':bootstrap_ci(peaks)}
                print(cfg_name, results[cfg_name]['acc']['mean'])
    return results


def run_continual():
    results = {}
    slots_list = [100, 200, 400, 800]
    winners_list = [1, 3, 5]
    for slots in slots_list:
        for winners in winners_list:
            for tune_mode in ['full', 'no_tune', 'no_alloc']:
                cfg_name = f's{slots}_w{winners}_{tune_mode}'
                aas=[]; fms=[]; walls=[]; peaks=[]
                per_seed = []
                for s in SEEDS:
                    set_seed(s)
                    tasks = split_digits_tasks(seed=s)
                    self_tune = tune_mode in ['full']
                    model = HSAM(input_dim=64, output_dim=10, n_slots=slots, n_winners=winners, seed=s, 
                                proj_dim=64, eta_k=0.08, eta_v=0.3, self_tune=self_tune)
                    T=len(tasks); task_acc=np.zeros((T,T))
                    with Tracker() as tr:
                        for k,task in enumerate(tasks):
                            Xtr_t,ytr_t=task['X_train'],task['y_train']
                            for p in range(2):
                                idx=np.random.default_rng(s*1000+k*10+p).permutation(len(Xtr_t))
                                for i in idx:
                                    x = Xtr_t[i]; y = one_hot(ytr_t[i],10)
                                    if tune_mode == 'no_alloc' and np.any(model.filled):
                                        old = model.v_high
                                        model.v_high = 1e9
                                        model.learn(x, y)
                                        model.v_high = old
                                    elif tune_mode == 'no_tune':
                                        old = model.self_tune
                                        model.self_tune = False
                                        model.learn(x, y)
                                        model.self_tune = old
                                    else:
                                        model.learn(x, y)
                            for j in range(T):
                                Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                                preds=[int(np.argmax(model.predict(Xt[i])[0])) for i in range(len(Xt))]
                                task_acc[j,k]=float((np.array(preds)==yt).mean())
                    tr.update()
                    aa=average_accuracy(task_acc); fm=forgetting_measure(task_acc)
                    aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
                    per_seed.append({'seed':s,'AA':aa,'FM':fm})
                results[cfg_name] = {'slots':slots,'winners':winners,'mode':tune_mode,'per_seed':per_seed,
                                     'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
                print('C', cfg_name, results[cfg_name]['AA']['mean'], results[cfg_name]['FM']['mean'])
    return results


def make_plots(sup, cont):
    # Plot for winners=3 (common case)
    winners = 3
    slots_list = [100,200,400,800]
    modes = ['full','no_tune','no_alloc']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for mi, mode in enumerate(modes):
        # supervised accuracy
        ax = axes[0, mi]
        for slots in slots_list:
            key = f's{slots}_w{winners}_{mode}'
            if key in sup:
                ax.scatter(sup[key]['wall_s']['mean'], sup[key]['acc']['mean'], label=f's{slots}')
        ax.set_xlabel('wall_s'); ax.set_ylabel('acc'); ax.set_title(f'W{winners} {mode} supervised')
        ax.legend()
        
        # continual AA
        ax = axes[1, mi]
        for slots in slots_list:
            key = f's{slots}_w{winners}_{mode}'
            if key in cont:
                ax.scatter(cont[key]['wall_s']['mean'], cont[key]['AA']['mean'], label=f's{slots}')
        ax.set_xlabel('wall_s'); ax.set_ylabel('AA'); ax.set_title(f'W{winners} {mode} continual')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'grid_w3.png'), dpi=150)
    plt.close()
    
    # Plot by slots
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    for si, slots in enumerate(slots_list):
        for mi, mode in enumerate(modes):
            key = f's{slots}_w{winners}_{mode}'  # use w3
            ax = axes[0, si] if mi==0 else None
        # supervised by slots
        ax = axes[0, si]
        for winners in [1,3,5]:
            for mode in ['full']:
                key = f's{slots}_w{winners}_{mode}'
                if key in sup:
                    ax.scatter(winners, sup[key]['acc']['mean'], label=f'W{winners}')
        ax.set_xlabel('winners'); ax.set_ylabel('acc'); ax.set_title(f'S{slots}')
        ax.legend()
        
        ax = axes[1, si]
        for winners in [1,3,5]:
            for mode in ['full']:
                key = f's{slots}_w{winners}_{mode}'
                if key in cont:
                    ax.scatter(winners, cont[key]['AA']['mean'], label=f'W{winners}')
        ax.set_xlabel('winners'); ax.set_ylabel('AA'); ax.set_title(f'S{slots} continual')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'grid_by_slots.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git':git, 'seeds':SEEDS}
    print('SUP')
    out['supervised'] = run_supervised()
    print('CONT')
    out['continual'] = run_continual()
    make_plots(out['supervised'], out['continual'])
    save_json(os.path.join(OUT,'results.json'), out)
    print('done')
