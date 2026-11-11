#!/usr/bin/env python3
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.data import make_tiny_char_stream
from src.candidates import CSM

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'research', '03_experiments', 'char_csm')
os.makedirs(OUT, exist_ok=True)
SEEDS = [0,1,2,3,4]

def run(seed, length=8000, adapt=True):
    set_seed(seed)
    seq, vocab, ch2i = make_tiny_char_stream(length=length, seed=seed)
    n = len(vocab)
    model = CSM(n_symbols=n, max_context=6 if adapt else 2, init_context=1 if adapt else 2)
    if not adapt:
        model.context = 2; model.max_context = 2
    history = []
    corrects = []
    surprises = []
    with Tracker() as tr:
        for i in range(len(seq)-1):
            h = seq[max(0,i-32):i].tolist()
            pred = model.predict(h)
            corrects.append(int(pred==seq[i]))
            info = model.learn_step(h, int(seq[i]))
            surprises.append(info['surprise'])
            # disable adaptation if requested by forcing context
            if not adapt:
                model.context = 2
    tr.update()
    # report overall and last-20% accuracy (self-improvement proxy)
    c = np.array(corrects)
    split = int(0.8*len(c))
    return {
        'acc_overall': float(c.mean()),
        'acc_last20': float(c[split:].mean()),
        'acc_first20': float(c[:int(0.2*len(c))].mean()),
        'mean_surprise': float(np.mean(surprises)),
        'final_context': int(model.context),
        'wall_s': tr.stats.wall_seconds,
        'peak_rss_mb': tr.stats.peak_rss_mb,
    }

if __name__ == '__main__':
    git = git_commit_hash(ROOT)
    res = {}
    for name, adapt in [('csm_adaptive', True), ('csm_fixed_ctx2', False)]:
        rows=[]; accs=[]; last=[]; first=[]; walls=[]
        for s in SEEDS:
            r = run(s, length=6000, adapt=adapt)
            rows.append({'seed':s, **r})
            accs.append(r['acc_overall']); last.append(r['acc_last20']); first.append(r['acc_first20']); walls.append(r['wall_s'])
        res[name] = {'per_seed': rows, 'acc_overall': bootstrap_ci(accs), 'acc_last20': bootstrap_ci(last), 'acc_first20': bootstrap_ci(first), 'wall_s': bootstrap_ci(walls)}
    save_json(os.path.join(OUT,'results.json'), {'git':git,'seeds':SEEDS,'results':res})
    for k,v in res.items():
        print(k, v['acc_overall']['mean'], 'last20', v['acc_last20']['mean'], 'first20', v['acc_first20']['mean'])
