#!/usr/bin/env python3
"""Fashion-MNIST via raw IDX (numpy+gzip+struct). PCA-64 features.
Benchmarks HSAM_Simplified vs HSAM vs SLDA vs tuned replay vs GNG-U vs FuzzyART.
"""
import os, sys, gzip, struct, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.utils import set_seed, Tracker, save_json, git_commit_hash, bootstrap_ci
from src.candidates import HSAM, HSAM_Simplified
from src.baselines import FuzzyART, GNGU, StreamingLDA, ReplayMLPTuned, ridge_readout_baseline, mlp_sklearn_baseline, random_feature_ridge
from src.metrics import average_accuracy, forgetting_measure

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "research", "03_experiments", "simplified_benchmark_fmnist")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        zero, data_type, ndim = struct.unpack(">HBB", f.read(4))
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims).astype(np.float32)


def load_fmnist(seed=0, n_train=6000, n_test=1200, n_comp=64):
    base = "/tmp/fmnist"
    Xtr = _read_idx(f"{base}/train-images-idx3-ubyte.gz").reshape(-1, 28*28) / 255.0
    ytr = _read_idx(f"{base}/train-labels-idx1-ubyte.gz").astype(int)
    Xte = _read_idx(f"{base}/t10k-images-idx3-ubyte.gz").reshape(-1, 28*28) / 255.0
    yte = _read_idx(f"{base}/t10k-labels-idx1-ubyte.gz").astype(int)
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(Xtr), n_train, replace=False)
    te = rng.choice(len(Xte), n_test, replace=False)
    Xtr, ytr = Xtr[tr], ytr[tr]
    Xte, yte = Xte[te], yte[te]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtr_s = (Xtr - mu) / sd; Xte_s = (Xte - mu) / sd
    _, S, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
    W = Vt[:n_comp]
    ev = float((S[:n_comp]**2).sum() / (S**2).sum())
    return (Xtr_s @ W.T).astype(np.float32), ytr, (Xte_s @ W.T).astype(np.float32), yte, {
        "explained_var_pca64": ev, "n_train": n_train, "n_test": n_test, "input_dim": n_comp}


def one_hot(y, n=10):
    v = np.zeros(n, dtype=np.float32); v[int(y)] = 1.0; return v


def eval_acc_hsam(model, X, y):
    preds = [int(np.argmax(model.predict(X[i])[0])) for i in range(len(X))]
    return float(np.mean(np.array(preds) == np.asarray(y)))


def eval_acc_sklearn(model, X, y):
    try:
        p = model.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))
    except Exception:
        return float(model.score(X, y))


def hsam_sup(cls, cfg, Xtr, ytr, Xte, yte):
    accs, walls, peaks, per_seed = [], [], [], []
    for s in SEEDS:
        set_seed(s)
        model = cls(input_dim=Xtr.shape[1], output_dim=10, seed=s, proj_dim=min(Xtr.shape[1], 64), **cfg)
        with Tracker() as tr:
            for p in range(cfg.get("passes", 2)):
                idx = np.random.default_rng(s*1000+p).permutation(len(Xtr))
                for i in idx:
                    model.learn(Xtr[i], one_hot(ytr[i], 10))
            acc = eval_acc_hsam(model, Xte, yte)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "acc": acc})
    return {"per_seed": per_seed, "acc": bootstrap_ci(accs),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


def hsam_cont(cls, cfg, Xtr, ytr, Xte, yte):
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    aas, fms, walls, peaks, per_seed = [], [], [], [], []
    for s in SEEDS:
        set_seed(s)
        rng = np.random.default_rng(s)
        tr_i = rng.choice(len(Xtr), 2000, replace=False)
        te_i = rng.choice(len(Xte), 400, replace=False)
        tasks = []
        for a, b in pairs:
            mtr = (ytr[tr_i] == a) | (ytr[tr_i] == b)
            mte = (yte[te_i] == a) | (yte[te_i] == b)
            tasks.append({"X_train": Xtr[tr_i][mtr], "y_train": ytr[tr_i][mtr],
                          "X_test": Xte[te_i][mte], "y_test": yte[te_i][mte], "classes": (a, b)})
        model = cls(input_dim=Xtr.shape[1], output_dim=10, seed=s, proj_dim=min(Xtr.shape[1], 64), **cfg)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                Xt_, yt_ = task["X_train"], task["y_train"]
                for p in range(cfg.get("passes", 2)):
                    idx = np.random.default_rng(s*1000+k*10+p).permutation(len(Xt_))
                    for i in idx:
                        model.learn(Xt_[i], one_hot(yt_[i], 10))
                for j in range(T):
                    Xq, yq = tasks[j]["X_test"], tasks[j]["y_test"]
                    task_acc[j, k] = eval_acc_hsam(model, Xq, yq)
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "AA": aa, "FM": fm})
    return {"per_seed": per_seed, "AA": bootstrap_ci(aas), "FM": bootstrap_ci(fms),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


def sklearn_sup(fn, Xtr, ytr, Xte, yte):

    accs, walls, peaks, per_seed = [], [], [], []
    for s in SEEDS:
        set_seed(s)
        with Tracker() as tr:
            acc = fn(Xtr, ytr, Xte, yte, s)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "acc": acc})
    return {"per_seed": per_seed, "acc": bootstrap_ci(accs),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


def proto_sup(cls, kwargs, Xtr, ytr, Xte, yte):

    accs, walls, peaks, per_seed = [], [], [], []
    for s in SEEDS:
        set_seed(s)
        model = cls(input_dim=Xtr.shape[1], seed=s, **kwargs)
        with Tracker() as tr:
            model.fit(Xtr, ytr); acc = model.score(Xte, yte)
        tr.update()
        accs.append(acc); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "acc": acc})
    return {"per_seed": per_seed, "acc": bootstrap_ci(accs),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


def replay_cont(buffer_size, Xtr, ytr, Xte, yte):

    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    aas, fms, walls, peaks, per_seed = [], [], [], [], []
    for s in SEEDS:
        set_seed(s)
        rng = np.random.default_rng(s)
        tr_i = rng.choice(len(Xtr), 2000, replace=False)
        te_i = rng.choice(len(Xte), 400, replace=False)
        tasks = []
        for a, b in pairs:
            mtr = (ytr[tr_i] == a) | (ytr[tr_i] == b)
            mte = (yte[te_i] == a) | (yte[te_i] == b)
            tasks.append({"X_train": Xtr[tr_i][mtr], "y_train": ytr[tr_i][mtr],
                          "X_test": Xte[te_i][mte], "y_test": yte[te_i][mte], "classes": (a, b)})
        model = ReplayMLPTuned(in_dim=Xtr.shape[1], hidden=64, out_dim=10,
                               buffer_per_task=buffer_size, seed=s, epochs=10)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                model.train_task(task["X_train"], task["y_train"])
                for j in range(T):
                    Xq, yq = tasks[j]["X_test"], tasks[j]["y_test"]
                    task_acc[j, k] = float(np.mean(model.predict(Xq) == yq))
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "AA": aa, "FM": fm})
    return {"per_seed": per_seed, "AA": bootstrap_ci(aas), "FM": bootstrap_ci(fms),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


def proto_cont(cls, kwargs, Xtr, ytr, Xte, yte):

    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    aas, fms, walls, peaks, per_seed = [], [], [], [], []
    for s in SEEDS:
        set_seed(s)
        rng = np.random.default_rng(s)
        tr_i = rng.choice(len(Xtr), 2000, replace=False)
        te_i = rng.choice(len(Xte), 400, replace=False)
        tasks = []
        for a, b in pairs:
            mtr = (ytr[tr_i] == a) | (ytr[tr_i] == b)
            mte = (yte[te_i] == a) | (yte[te_i] == b)
            tasks.append({"X_train": Xtr[tr_i][mtr], "y_train": ytr[tr_i][mtr],
                          "X_test": Xte[te_i][mte], "y_test": yte[te_i][mte], "classes": (a, b)})
        model = cls(input_dim=Xtr.shape[1], seed=s, **kwargs)
        T = len(tasks); task_acc = np.zeros((T, T))
        with Tracker() as tr:
            for k, task in enumerate(tasks):
                model.fit(task["X_train"], task["y_train"])
                for j in range(T):
                    Xq, yq = tasks[j]["X_test"], tasks[j]["y_test"]
                    task_acc[j, k] = model.score(Xq, yq)
        tr.update()
        aa = average_accuracy(task_acc); fm = forgetting_measure(task_acc)
        aas.append(aa); fms.append(fm); walls.append(tr.stats.wall_seconds); peaks.append(tr.stats.peak_rss_mb)
        per_seed.append({"seed": s, "AA": aa, "FM": fm})
    return {"per_seed": per_seed, "AA": bootstrap_ci(aas), "FM": bootstrap_ci(fms),
            "wall_s": bootstrap_ci(walls), "peak_rss_mb": bootstrap_ci(peaks)}


if __name__ == "__main__":
    git = git_commit_hash(ROOT)
    out = {"git": git, "seeds": SEEDS}
    Xtr, ytr, Xte, yte, meta = load_fmnist(seed=123)
    out["meta"] = meta
    print("meta:", meta)

    sup, cont = {}, {}

    print("=== HSAM original ===")
    sup["hsam_s400_w3_p2_tune"] = hsam_sup(HSAM, dict(n_slots=400, n_winners=3, self_tune=True, eta_k=0.05, eta_v=0.5), Xtr, ytr, Xte, yte)
    cont["hsam_s400_w3_p2_tune"] = hsam_cont(HSAM, dict(n_slots=400, n_winners=3, self_tune=True, eta_k=0.05, eta_v=0.5), Xtr, ytr, Xte, yte)

    print("=== HSAM_Simplified ===")
    sup["hsam_simplified_s400_w3"] = hsam_sup(HSAM_Simplified, dict(n_slots=400, n_winners=3, eta_k=0.05, eta_v=0.5), Xtr, ytr, Xte, yte)
    cont["hsam_simplified_s400_w3"] = hsam_cont(HSAM_Simplified, dict(n_slots=400, n_winners=3, eta_k=0.05, eta_v=0.5), Xtr, ytr, Xte, yte)

    print("=== sklearn baselines ===")
    sup["ridge"] = sklearn_sup(lambda tr_X, tr_y, te_X, te_y, s: ridge_readout_baseline(tr_X, tr_y, te_X, te_y, seed=s, alpha=1.0)[0], Xtr, ytr, Xte, yte)
    sup["rf_ridge_256"] = sklearn_sup(lambda tr_X, tr_y, te_X, te_y, s: random_feature_ridge(tr_X, tr_y, te_X, te_y, seed=s, n_features=256, alpha=1.0)[0], Xtr, ytr, Xte, yte)
    sup["mlp_sklearn_64"] = sklearn_sup(lambda tr_X, tr_y, te_X, te_y, s: mlp_sklearn_baseline(tr_X, tr_y, te_X, te_y, seed=s, hidden=64, max_iter=200)[0], Xtr, ytr, Xte, yte)
    sup["fuzzy_art_rho07"] = proto_sup(FuzzyART, dict(rho=0.7, beta=1.0), Xtr, ytr, Xte, yte)
    sup["gngu"] = proto_sup(GNGU, dict(max_nodes=200), Xtr, ytr, Xte, yte)
    sup["slda"] = proto_sup(StreamingLDA, dict(), Xtr, ytr, Xte, yte)

    print("=== continual baselines ===")
    cont["replay_mlp_buf100"] = replay_cont(100, Xtr, ytr, Xte, yte)
    cont["replay_mlp_buf200"] = replay_cont(200, Xtr, ytr, Xte, yte)
    cont["fuzzy_art_rho07"] = proto_cont(FuzzyART, dict(rho=0.7, beta=1.0), Xtr, ytr, Xte, yte)
    cont["gngu"] = proto_cont(GNGU, dict(max_nodes=200), Xtr, ytr, Xte, yte)
    cont["slda"] = proto_cont(StreamingLDA, dict(), Xtr, ytr, Xte, yte)

    out["supervised"] = sup
    out["continual"] = cont
    save_json(os.path.join(OUT, "results.json"), out)

    print("\n=== SUPERVISED ===")
    for k, v in sup.items():
        print(f'{k}: acc={v["acc"]["mean"]:.4f}±{v["acc"]["std"]:.4f} wall={v["wall_s"]["mean"]:.3f}s rss={v["peak_rss_mb"]["mean"]:.1f}MB')
    print("\n=== CONTINUAL ===")
    for k, v in cont.items():
        print(f'{k}: AA={v["AA"]["mean"]:.4f}±{v["AA"]["std"]:.4f} FM={v["FM"]["mean"]:.4f}±{v["FM"]["std"]:.4f} wall={v["wall_s"]["mean"]:.3f}s rss={v["peak_rss_mb"]["mean"]:.1f}MB')
