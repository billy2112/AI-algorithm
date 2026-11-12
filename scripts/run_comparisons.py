#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import get_digits_split, split_digits_tasks
from src.metrics import average_accuracy, forgetting_measure
from src.candidates import HSAM
from src.baselines import (
    FuzzyART, GNGU, StreamingLDA, AGEM, ReplayMLPTuned,
    ridge_readout_baseline, mlp_sklearn_baseline, random_feature_ridge,
    TorchMLP, train_torch_classifier, finetune_train_split_digits,
    ewc_train_split_digits, replay_train_split_digits
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "comparisons")
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


def run_supervised():
    results = {}
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=0)
    # HSAM best configs
    for cfg_name in ['hsam_s400_w3_p2_tune', 'hsam_s200_w3_p2_tune']:
        if cfg_name == 'hsam_s400_w3_p2_tune':
            cfg = dict(n_slots=400, n_winners=3, passes=2, self_tune=True)
        else:
            cfg = dict(n_slots=200, n_winners=3, passes=2, self_tune=True)
        accs = []; walls=[]; peaks=[]
        per_seed = []
        for s in SEEDS:
            set_seed(s)
            (Xtr_s, ytr_s), (Xte_s, yte_s) = get_digits_split(seed=s)
            model = HSAM(input_dim=64, output_dim=10, seed=s, proj_dim=64, eta_k=0.05, eta_v=0.5, **cfg)
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
            set_seed(s)
            (Xtr_s,ytr_s),(Xte_s,yte_s)=get_digits_split(seed=s)
            with Tracker() as tr:
                acc = fn(Xtr_s,ytr_s,Xte_s,yte_s,s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc})
        results[name] = {'per_seed':per_seed,'acc':bootstrap_ci(accs),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # ART and GNG-U
    for name, Cls, kwargs in [('fuzzy_art_rho07', FuzzyART, dict(rho=0.7,beta=1.0)),
                               ('fuzzy_art_rho05', FuzzyART, dict(rho=0.5,beta=0.5)),
                               ('gngu', GNGU, dict(max_nodes=200))]:
        accs=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s)
            (Xtr_s,ytr_s),(Xte_s,yte_s)=get_digits_split(seed=s)
            with Tracker() as tr:
                if name.startswith('fuzzy_art'):
                    model = Cls(input_dim=64, seed=s, **kwargs)
                    model.fit(Xtr_s,ytr_s)
                else:
                    model = Cls(input_dim=64, seed=s, **kwargs)
                    model.fit(Xtr_s,ytr_s)
                acc = eval_acc_sklearn(model, Xte_s, yte_s)
            tr.update()
            accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'acc':acc,'n_cats':getattr(model,'n_learned',len(getattr(model,'units',[])))})
        results[name] = {'per_seed':per_seed,'acc':bootstrap_ci(accs),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # SLDA
    accs=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        set_seed(s)
        (Xtr_s,ytr_s),(Xte_s,yte_s)=get_digits_split(seed=s)
        with Tracker() as tr:
            model = StreamingLDA(input_dim=64, seed=s)
            model.fit(Xtr_s,ytr_s)
            acc = model.score(Xte_s,yte_s)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({'seed':s,'acc':acc})
    results['slda'] = {'per_seed':per_seed,'acc':bootstrap_ci(accs),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    return results


def run_continual():
    results = {}
    # HSAM best
    for cfg_name, cfg in [('hsam_s200_w3_p2_tune', dict(n_slots=200,n_winners=3,passes=2,self_tune=True)),
                          ('hsam_s400_w3_p2_tune', dict(n_slots=400,n_winners=3,passes=2,self_tune=True))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s)
            tasks = split_digits_tasks(seed=s)
            model = HSAM(input_dim=64, output_dim=10, seed=s, proj_dim=64, eta_k=0.08, eta_v=0.3, **cfg)
            T=len(tasks); task_acc=np.zeros((T,T))
            with Tracker() as tr:
                for k,task in enumerate(tasks):
                    Xtr,ytr=task['X_train'],task['y_train']
                    for p in range(cfg['passes']):
                        idx=np.random.default_rng(s*1000+k*10+p).permutation(len(Xtr))
                        for i in idx:
                            model.learn(Xtr[i], one_hot(ytr[i],10))
                    for j in range(T):
                        Xt,yt=tasks[j]['X_test'],tasks[j]['y_test']
                        preds=[int(np.argmax(model.predict(Xt[i])[0])) for i in range(len(Xt))]
                        task_acc[j,k]=float((np.array(preds)==yt).mean())
            tr.update()
            aa=average_accuracy(task_acc); fm=forgetting_measure(task_acc)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[cfg_name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # MLP baselines
    for name, fn in [('finetune_mlp', lambda tasks,s: finetune_train_split_digits(tasks, epochs=15, lr=1e-3, seed=s)),
                     ('ewc_mlp', lambda tasks,s: ewc_train_split_digits(tasks, epochs=15, lr=1e-3, ewc_lambda=50.0, seed=s)),
                     ('replay_mlp_buf20', lambda tasks,s: replay_train_split_digits(tasks, epochs=15, lr=1e-3, buffer_per_task=20, seed=s))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s)
            try:
                import torch
                torch.manual_seed(s)
            except:
                pass
            tasks = split_digits_tasks(seed=s)
            with Tracker() as tr:
                mat = fn(tasks,s)
            tr.update()
            aa=average_accuracy(mat); fm=forgetting_measure(mat)
            aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
            per_seed.append({'seed':s,'AA':aa,'FM':fm})
        results[name] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # Tuned replay, AGEM
    for name, Cls, kwargs in [('replay_mlp_buf100', ReplayMLPTuned, dict(buffer_per_task=100,epochs=15)),
                               ('replay_mlp_buf200', ReplayMLPTuned, dict(buffer_per_task=200,epochs=15))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s)
            tasks = split_digits_tasks(seed=s)
            with Tracker() as tr:
                model = Cls(in_dim=64,hidden=64,out_dim=10,seed=s,**kwargs)
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
    
    # AGEM
    aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        with Tracker() as tr:
            model = AGEM(in_dim=64,hidden=64,out_dim=10,memory_size=100,seed=s,epochs=10)
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
    results['agem_mem100'] = {'per_seed':per_seed,'AA':bootstrap_ci(aas),'FM':bootstrap_ci(fms),'wall_s':bootstrap_ci(walls),'peak_rss_mb':bootstrap_ci(peaks)}
    
    # ART, GNG-U, SLDA in continual setting
    for name, Cls, kwargs in [('fuzzy_art_rho07_cont', FuzzyART, dict(rho=0.7,beta=1.0)),
                               ('gngu_cont', GNGU, dict(max_nodes=200))]:
        aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
        for s in SEEDS:
            set_seed(s)
            tasks = split_digits_tasks(seed=s)
            with Tracker() as tr:
                if 'art' in name:
                    model = Cls(input_dim=64, seed=s, **kwargs)
                else:
                    model = Cls(input_dim=64, seed=s, **kwargs)
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
    
    # SLDA
    aas=[]; fms=[]; walls=[]; peaks=[]; per_seed=[]
    for s in SEEDS:
        set_seed(s)
        tasks = split_digits_tasks(seed=s)
        with Tracker() as tr:
            model = StreamingLDA(input_dim=64, seed=s)
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


def run_stream_probe():
    results = {}
    from src.data import get_digits_split
    def one_hot(y,n=10): v=np.zeros(n,dtype=np.float32); v[int(y)]=1.0; return v
    def eval_probe(model,Xte,yte): preds=[int(np.argmax(model.predict(Xte[i])[0])) for i in range(len(Xte))]; return float((np.array(preds)==yte).mean())
    for name, Cls, kwargs, is_hsam in [('hsam_s200_w3_p2_tune', HSAM, dict(n_slots=200,n_winners=3,self_tune=True), True),
                                       ('fuzzy_art_rho07', FuzzyART, dict(rho=0.7,beta=1.0), False),
                                       ('gngu', GNGU, dict(max_nodes=200), False),
                                       ('slda', StreamingLDA, dict(), False)]:
        per_seed=[]; first=[]; last=[]; deltas=[]
        for s in SEEDS:
            set_seed(s)
            (Xtr,ytr),(Xte,yte)=get_digits_split(seed=s)
            rng=np.random.default_rng(s)
            if is_hsam:
                model = Cls(input_dim=64,output_dim=10,n_slots=kwargs['n_slots'],n_winners=kwargs['n_winners'],seed=s,proj_dim=64,eta_k=0.08,eta_v=0.25,self_tune=kwargs['self_tune'])
            elif name.startswith('fuzzy_art'):
                model = Cls(input_dim=64,seed=s,**kwargs)
            elif name=='gngu':
                model = Cls(input_dim=64,seed=s,**kwargs)
            else:
                model = Cls(input_dim=64,seed=s,**kwargs)
            accs=[]; total=0
            with Tracker() as tr:
                for c in range(20):
                    idx=rng.integers(0,len(Xtr),size=200)
                    for i in idx:
                        if is_hsam:
                            model.learn(Xtr[i], one_hot(ytr[i],10))
                        else:
                            model.partial_fit(Xtr[i], ytr[i]) if hasattr(model,'partial_fit') else model.fit([Xtr[i]],[ytr[i]])
                    total+=200
                    if is_hsam:
                        accs.append(eval_probe(model,Xte,yte))
                    else:
                        accs.append(eval_acc_sklearn(model,Xte,yte))
            tr.update()
            per_seed.append({'seed':s,'curve_accs':accs,'wall_s':tr.stats.wall_seconds,'peak_rss_mb':tr.stats.peak_rss_mb})
            first.append(accs[0]); last.append(accs[-1]); deltas.append(accs[-1]-accs[0])
        results[name] = {'per_seed':per_seed,'acc_first':bootstrap_ci(first),'acc_last':bootstrap_ci(last),'delta':bootstrap_ci(deltas)}
    return results


if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    out = {'git':git, 'seeds':SEEDS}
    print('SUP')
    out['supervised'] = run_supervised()
    print('CONT')
    out['continual'] = run_continual()
    print('STREAM')
    out['stream'] = run_stream_probe()
    save_json(os.path.join(OUT,'results.json'), out)
    print('done')
