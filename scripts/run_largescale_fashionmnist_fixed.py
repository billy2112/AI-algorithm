#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.candidates import HSAM
from src.baselines import (
    ridge_readout_baseline, mlp_sklearn_baseline, random_feature_ridge,
    finetune_train_split_digits, ewc_train_split_digits, replay_train_split_digits,
    FuzzyART, GNGU, StreamingLDA, AGEM, ReplayMLPTuned
)
from src.metrics import average_accuracy, forgetting_measure

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "largescale_fashionmnist")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32)
    v[int(y)] = 1.0
    return v


def eval_acc_hsam(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float((np.array(preds) == y).mean())


def load_fashion_mnist_pca(n_components=64, seed=0, max_train=6000, max_test=1200):
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        y = y.astype(int)
    except Exception as e:
        raise RuntimeError(f"Fashion-MNIST load failed: {e}")
    
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(X), size=min(max_train, len(X)), replace=False)
    remaining = np.setdiff1d(np.arange(len(X)), train_idx)
    test_idx = rng.choice(remaining, size=min(max_test, len(remaining)), replace=False)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    pca = PCA(n_components=n_components, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)
    return (Xtr_p, ytr), (Xte_p, yte), {
        'explained_var': float(pca.explained_variance_ratio_.sum()), 
        'n_components': n_components,
        'n_train': len(Xtr_p),
        'n_test': len(Xte_p),
        'total_samples_used': len(Xtr_p) + len(Xte_p)
    }


def run_supervised():
    results = {}
    try:
        (Xtr, ytr), (Xte, yte), meta = load_fashion_mnist_pca(n_components=64, seed=123)
    except Exception as e:
        results['error'] = str(e)
        return results
    
    results['meta'] = meta
    input_dim = Xtr.shape[1]
    
    for cfg_name, cfg in [('hsam_s400_w3_p2_tune', dict(n_slots=400, n_winners=3, passes=2, self_tune=True, eta_k=0.05, eta_v=0.5)),
                          ('hsam_s200_w3_p2_tune', dict(n_slots=200, n_winners=3, passes=2, self_tune=True, eta_k=0.08, eta_v=0.3))]:
        accs = []; walls=[]; peaks=[]
        per_seed = []
        for s in SEEDS:
            set_seed(s)
            try:
                (Xtr_s, ytr_s), (Xte_s, yte_s), _ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s, ytr_s, Xte_s, yte_s = Xtr, ytr, Xte, yte
            model = HSAM(input_dim=input_dim, output_dim=10, seed=s, proj_dim=min(input_dim,64), n_slots=cfg['n_slots'], n_winners=cfg['n_winners'], 
                        eta_k=cfg['eta_k'], eta_v=cfg['eta_v'], self_tune=cfg['self_tune'])
            with Tracker() as tr:
                for p in range(cfg['passes']):
                    idx = np.random.default_rng(s*1000+p).permutation(len(Xtr_s))
                    for i in idx:
                        model.learn(Xtr_s[i], one_hot(ytr_s[i],10))
                acc = eval_acc_hsam(model, Xte_s, yte_s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc})
        results[cfg_name] = {'per_seed':per_seed, 'acc':bootstrap_ci(accs), 'wall_s':bootstrap_ci(walls), 'peak_rss_mb':bootstrap_ci(peaks)}
    
    for name, fn in [('ridge', lambda Xtr,ytr,Xte,yte,s: ridge_readout_baseline(Xtr,ytr,Xte,yte,seed=s,alpha=1.0)[0]),
                     ('rf_ridge_256', lambda Xtr,ytr,Xte,yte,s: random_feature_ridge(Xtr,ytr,Xte,yte,seed=s,n_features=256,alpha=1.0)[0]),
                     ('mlp_sklearn_64', lambda Xtr,ytr,Xte,yte,s: mlp_sklearn_baseline(Xtr,ytr,Xte,yte,seed=s,hidden=64,max_iter=200)[0])]:
        accs=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            with Tracker() as tr:
                acc = fn(Xtr_s,ytr_s,Xte_s,yte_s,s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc})
        results[name] = {'per_seed':per_seed,'acc':bootstrap_ci(accs),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    for name, Cls, kwargs in [('fuzzy_art_rho07', FuzzyART, dict(rho=0.7,beta=1.0)),
                               ('gngu', GNGU, dict(max_nodes=300)),
                               ('slda', StreamingLDA, dict())]:
        accs=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            with Tracker() as tr:
                if name == 'slda':
                    model = Cls(input_dim=input_dim, seed=s)
                else:
                    model = Cls(input_dim=input_dim, seed=s, **kwargs)
                model.fit(Xtr_s,ytr_s)
                acc = model.score(Xte_s,yte_s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc})
        results[name] = {'per_seed':per_seed,'acc':bootstrap_ci(accs),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    return results


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git':git, 'seeds':SEEDS, 'dataset': 'Fashion-MNIST (OpenML, PCA features)'}
    print('SUP')
    out['supervised'] = run_supervised()
    save_json(os.path.join(OUT,'results.json'), out)
    print('done')
