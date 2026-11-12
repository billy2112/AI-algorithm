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


def eval_acc_sklearn(model, X, y):
    try:
        p = model.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))
    except Exception:
        return float(model.score(X, y))


def load_fashion_mnist_pca(n_components=64, seed=0, max_train=5000, max_test=1000):
    try:
        from sklearn.datasets import fetch_openml
        # Try to fetch Fashion-MNIST
        X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False, parser='auto')
        X = X.astype(np.float32) / 255.0
        y = y.astype(int)
    except Exception as e:
        print(f"Failed to load Fashion-MNIST: {e}", file=sys.stderr)
        raise
    
    rng = np.random.default_rng(seed)
    # Subsample for budget
    train_idx = rng.choice(len(X), size=min(max_train, len(X)), replace=False)
    test_idx = rng.choice(np.setdiff1d(np.arange(len(X)), train_idx), size=min(max_test, len(np.setdiff1d(np.arange(len(X)), train_idx))), replace=False)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    pca = PCA(n_components=n_components, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)
    Xte_p = pca.transform(Xte_s).astype(np.float32)
    return (Xtr_p, ytr), (Xte_p, yte), {'explained_var': float(pca.explained_variance_ratio_.sum()), 'n_components': n_components}


def run_supervised():
    results = {}
    try:
        (Xtr, ytr), (Xte, yte), meta = load_fashion_mnist_pca(n_components=64, seed=123)
    except Exception as e:
        results['error'] = str(e)
        return results
    
    results['meta'] = meta
    input_dim = Xtr.shape[1]
    
    # HSAM best configs
    for cfg_name, cfg in [('hsam_s400_w3_p2_tune', dict(n_slots=400, n_winners=3, passes=2, self_tune=True, eta_k=0.05, eta_v=0.5)),
                          ('hsam_s200_w3_p2_tune', dict(n_slots=200, n_winners=3, passes=2, self_tune=True, eta_k=0.08, eta_v=0.3))]:
        accs = []; walls=[]; peaks=[]
        per_seed = []
        for s in SEEDS:
            set_seed(s)
            # regenerate split with different seed
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
    
    # Baselines
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
    
    # ART, GNG-U, SLDA
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


def run_continual():
    results = {}
    try:
        (Xtr, ytr), (Xte, yte), meta = load_fashion_mnist_pca(n_components=64, seed=123)
    except Exception as e:
        results['error'] = str(e)
        return results
    results['meta'] = meta
    input_dim = Xtr.shape[1]
    
    # create split tasks
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    def make_tasks(Xtr,ytr,Xte,yte,pairs):
        tasks = []
        for a,b in pairs:
            mtr = (ytr==a)|(ytr==b)
            mte = (yte==a)|(yte==b)
            tasks.append({'X_train':Xtr[mtr],'y_train':ytr[mtr],'X_test':Xte[mte],'y_test':yte[mte],'classes':(a,b)})
        return tasks
    
    for cfg_name, cfg in [('hsam_s200_w3_p2_tune', dict(n_slots=200,n_winners=3,passes=2,self_tune=True)),
                          ('hsam_s400_w3_p2_tune', dict(n_slots=400,n_winners=3,passes=2,self_tune=True))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            tasks = make_tasks(Xtr_s,ytr_s,Xte_s,yte_s,pairs)
            model = HSAM(input_dim=input_dim, output_dim=10, seed=s, proj_dim=min(input_dim,64), n_slots=cfg['n_slots'], n_winners=cfg['n_winners'], eta_k=0.08, eta_v=0.3, self_tune=cfg['self_tune'])
            T=len(tasks); task_acc=np.zeros((T,T))
            with Tracker() as tr:
                for k,task in enumerate(tasks):
                    Xtr_t,ytr_t=task['X_train'],task['y_train']
                    for p in range(cfg['passes']):
                        idx=np.random.default_rng(s*1000+k*10+p).permutation(len(Xtr_t))
                        for i in idx:
                            model.learn(Xtr_t[i], one_hot(ytr_t[i],10))
                    for j in range(T):
                        Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                        preds=[int(np.argmax(model.predict(Xt[i])[0])) for i in range(len(Xt))]
                        task_acc[j,k]=float((np.array(preds)==yt).mean())
            tr.update()
            aa=average_accuracy(task_acc); fm=forgetting_measure(task_acc)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[cfg_name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # Baselines
    for name in ['finetune_mlp', 'ewc_mlp', 'replay_mlp_buf20']:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            tasks = make_tasks(Xtr_s,ytr_s,Xte_s,yte_s,pairs)
            try:
                import torch
                torch.manual_seed(s)
            except:
                pass
            with Tracker() as tr:
                if name == 'finetune_mlp':
                    mat = finetune_train_split_digits(tasks, in_dim=input_dim, epochs=10, lr=1e-3, seed=s)
                elif name == 'ewc_mlp':
                    mat = ewc_train_split_digits(tasks, in_dim=input_dim, epochs=10, lr=1e-3, ewc_lambda=50.0, seed=s)
                else:
                    mat = replay_train_split_digits(tasks, in_dim=input_dim, epochs=10, lr=1e-3, buffer_per_task=20, seed=s)
            tr.update()
            aa=average_accuracy(mat); fm=forgetting_measure(mat)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    for name, buffer_size in [('replay_mlp_buf100', 100), ('replay_mlp_buf200', 200)]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            tasks = make_tasks(Xtr_s,ytr_s,Xte_s,yte_s,pairs)
            with Tracker() as tr:
                model = ReplayMLPTuned(in_dim=input_dim,hidden=64,out_dim=10,buffer_per_task=buffer_size,seed=s,epochs=10)
                T=len(tasks); task_acc=np.zeros((T,T))
                for k,task in enumerate(tasks):
                    model.train_task(task['X_train'], task['y_train'])
                    for j in range(T):
                        Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                        task_acc[j,k]=float(np.mean(model.predict(Xt)==yt))
                mat=task_acc
            tr.update()
            aa=average_accuracy(mat); fm=forgetting_measure(mat)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    for name, Cls, kwargs in [('fuzzy_art_rho07_cont', FuzzyART, dict(rho=0.7,beta=1.0)),
                               ('gngu_cont', GNGU, dict(max_nodes=300))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            try:
                (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
            except:
                Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
            tasks = make_tasks(Xtr_s,ytr_s,Xte_s,yte_s,pairs)
            with Tracker() as tr:
                model = Cls(input_dim=input_dim, seed=s, **kwargs)
                T=len(tasks); task_acc=np.zeros((T,T))
                for k,task in enumerate(tasks):
                    model.fit(task['X_train'], task['y_train'])
                    for j in range(T):
                        Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                        task_acc[j,k]=float(np.mean(model.predict(Xt)==yt))
                mat=task_acc
            tr.update()
            aa=average_accuracy(mat); fm=forgetting_measure(mat)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        try:
            (Xtr_s,ytr_s),(Xte_s,yte_s),_ = load_fashion_mnist_pca(n_components=64, seed=s)
        except:
            Xtr_s,ytr_s,Xte_s,yte_s = Xtr,ytr,Xte,yte
        tasks = make_tasks(Xtr_s,ytr_s,Xte_s,yte_s,pairs)
        with Tracker() as tr:
            model = StreamingLDA(input_dim=input_dim, seed=s)
            T=len(tasks); task_acc=np.zeros((T,T))
            for k,task in enumerate(tasks):
                model.fit(task['X_train'], task['y_train'])
                for j in range(T):
                    Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                    task_acc[j,k]=model.score(Xt,yt)
            mat=task_acc
        tr.update()
        aa=average_accuracy(mat); fm=forgetting_measure(mat)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed':s,'AA':aa,'FM':fm})
    results['slda_cont'] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    return results


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git':git, 'seeds':SEEDS, 'dataset': 'Fashion-MNIST (PCA features)'}
    print('SUP')
    out['supervised'] = run_supervised()
    print('CONT')
    out['continual'] = run_continual()
    save_json(os.path.join(OUT,'results.json'), out)
    print('done')
