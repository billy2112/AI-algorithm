#!/usr/bin/env python3
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.baselines import ridge_readout_baseline, mlp_sklearn_baseline, random_feature_ridge, TorchMLP, train_torch_classifier, finetune_train_split_digits, ewc_train_split_digits, replay_train_split_digits
from src.metrics import average_accuracy, forgetting_measure

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'research', '03_experiments', 'baselines_digits')
os.makedirs(OUT, exist_ok=True)

SEEDS = [0,1,2,3,4]

def supervised_baselines():
    results = {}
    configs = [
        ('ridge', lambda Xtr,ytr,Xte,yte,s: ridge_readout_baseline(Xtr,ytr,Xte,yte,seed=s,alpha=1.0)),
        ('rf_ridge_256', lambda Xtr,ytr,Xte,yte,s: random_feature_ridge(Xtr,ytr,Xte,yte,seed=s,n_features=256,alpha=1.0)),
        ('mlp_sklearn_64', lambda Xtr,ytr,Xte,yte,s: mlp_sklearn_baseline(Xtr,ytr,Xte,yte,seed=s,hidden=64,max_iter=200)),
    ]
    for name, fn in configs:
        accs = []; walls=[]; peaks=[]
        per_seed=[]
        for s in SEEDS:
            set_seed(s)
            (Xtr,ytr),(Xte,yte) = get_digits_split(seed=s)
            with Tracker() as tr:
                acc, _ = fn(Xtr,ytr,Xte,yte,s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc,'wall_s':tr.stats.wall_seconds,'peak_rss_mb':tr.stats.peak_rss_mb})
        results[name] = {'per_seed': per_seed, 'acc': bootstrap_ci(accs), 'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}
    # torch MLP baseline (small, few epochs) for fairness as "standard backprop baseline"
    name = 'torch_mlp_64_e20'
    accs=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        set_seed(s); torch.manual_seed(s)
        (Xtr,ytr),(Xte,yte) = get_digits_split(seed=s)
        model = TorchMLP(64, hidden=64, out_dim=10)
        with Tracker() as tr:
            model, pred_fn = train_torch_classifier(model, Xtr, ytr, epochs=20, lr=1e-3, batch_size=64, seed=s)
            acc = float((pred_fn(Xte)==yte).mean())
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed':s,'acc':acc,'wall_s':tr.stats.wall_seconds,'peak_rss_mb':tr.stats.peak_rss_mb})
    results[name] = {'per_seed': per_seed, 'acc': bootstrap_ci(accs), 'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}
    return results

def continual_baselines():
    results = {}
    methods = {
        'finetune_mlp': lambda tasks,s: finetune_train_split_digits(tasks, epochs=15, lr=1e-3, seed=s),
        'ewc_mlp': lambda tasks,s: ewc_train_split_digits(tasks, epochs=15, lr=1e-3, ewc_lambda=50.0, seed=s),
        'replay_mlp_buf20': lambda tasks,s: replay_train_split_digits(tasks, epochs=15, lr=1e-3, buffer_per_task=20, seed=s),
    }
    for name, fn in methods.items():
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s); torch.manual_seed(s)
            tasks = split_digits_tasks(seed=s)
            with Tracker() as tr:
                mat = fn(tasks, s)
            tr.update()
            aa = average_accuracy(mat); fm = forgetting_measure(mat)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm,'matrix':mat.tolist(),'wall_s':tr.stats.wall_seconds,'peak_rss_mb':tr.stats.peak_rss_mb})
        results[name] = {'per_seed': per_seed, 'AA': bootstrap_ci(aas), 'FM': bootstrap_ci(fms), 'wall_s': bootstrap_ci(walls), 'peak_rss_mb': bootstrap_ci(peaks)}
    return results

if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    sup = supervised_baselines()
    save_json(os.path.join(OUT, 'supervised_results.json'), {'git':git,'seeds':SEEDS,'results':sup})
    cont = continual_baselines()
    save_json(os.path.join(OUT, 'continual_results.json'), {'git':git,'seeds':SEEDS,'results':cont})
    print('SUPERVISED SUMMARY')
    for k,v in sup.items():
        print(k, v['acc']['mean'], '+/-', v['acc']['std'], 'wall', v['wall_s']['mean'])
    print('CONTINUAL SUMMARY')
    for k,v in cont.items():
        print(k, 'AA', v['AA']['mean'], 'FM', v['FM']['mean'], 'wall', v['wall_s']['mean'])
