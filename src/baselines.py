import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.neural_network import MLPClassifier
except Exception:  # pragma: no cover
    RidgeClassifier = None
    MLPClassifier = None


def set_torch_seed(seed):
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(2)
    except Exception:
        pass


if TORCH_AVAILABLE:

    class TorchMLP(nn.Module):
        def __init__(self, in_dim, hidden=64, out_dim=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
            )

        def forward(self, x):
            return self.net(x)


    class TorchLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden=64, out_dim=10):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, out_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


    class SmallTransformerClassifier(nn.Module):
        def __init__(
            self, input_size, d_model=32, nhead=4, nlayers=2, out_dim=10, max_len=128
        ):
            super().__init__()
            self.inp = nn.Linear(input_size, d_model)
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
            self.fc = nn.Linear(d_model, out_dim)

        def forward(self, x):
            B, T, _ = x.shape
            h = self.inp(x) + self.pos[:, :T, :]
            h = self.enc(h)
            return self.fc(h[:, -1, :])

else:

    class TorchMLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for TorchMLP")

    class TorchLSTMClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for TorchLSTMClassifier")

    class SmallTransformerClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for SmallTransformerClassifier")


def train_torch_classifier(
    model,
    Xtr,
    ytr,
    Xva=None,
    yva=None,
    epochs=20,
    lr=1e-3,
    batch_size=64,
    seed=0,
    device="cpu",
    class_weight=None,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for train_torch_classifier")
    set_torch_seed(seed)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    n = len(Xtr)
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
    model.eval()

    def pred(X):
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=device)
            logits = model(Xt)
            return torch.argmax(logits, dim=1).cpu().numpy()

    return model, pred


def eval_pred(pred_fn, X, y):
    p = pred_fn(X)
    return float((p == y).mean())


def ridge_readout_baseline(Xtr, ytr, Xte, yte, seed=0, alpha=1.0):
    if RidgeClassifier is None:
        raise RuntimeError("sklearn is required for ridge_readout_baseline")
    clf = RidgeClassifier(alpha=alpha, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte)), clf


def mlp_sklearn_baseline(Xtr, ytr, Xte, yte, seed=0, hidden=64, max_iter=200):
    if MLPClassifier is None:
        raise RuntimeError("sklearn is required for mlp_sklearn_baseline")
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=max_iter, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte)), clf


def random_feature_ridge(Xtr, ytr, Xte, yte, seed=0, n_features=256, alpha=1.0, scale=1.0):
    if RidgeClassifier is None:
        raise RuntimeError("sklearn is required for random_feature_ridge")
    rng = np.random.default_rng(seed)
    D = Xtr.shape[1]
    W = rng.normal(0, scale / np.sqrt(max(1, D)), size=(D, n_features)).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, size=(n_features,)).astype(np.float32)

    def phi(X):
        return np.cos(X.astype(np.float32) @ W + b).astype(np.float32)

    Ztr = phi(Xtr)
    Zte = phi(Xte)
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(Ztr, ytr)
    return float(clf.score(Zte, yte)), {"W": W, "b": b, "clf": clf}


def _eval_tasks(model, tasks, device="cpu"):
    task_acc = np.zeros((len(tasks), len(tasks)))
    model.eval()
    for j in range(len(tasks)):
        Xt = torch.tensor(tasks[j]["X_test"], dtype=torch.float32, device=device)
        yt = tasks[j]["y_test"]
        with torch.no_grad():
            pred = torch.argmax(model(Xt), dim=1).cpu().numpy()
        task_acc[j, :] = task_acc[j, :]  # placeholder; caller fills column
        # return helper? easier to compute per k
    return task_acc


def finetune_train_split_digits(
    tasks, in_dim=64, hidden=64, epochs=10, lr=1e-3, seed=0, device="cpu"
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for finetune_train_split_digits")
    set_torch_seed(seed)
    model = TorchMLP(in_dim, hidden, out_dim=10).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    task_acc = np.zeros((len(tasks), len(tasks)))
    for k, task in enumerate(tasks):
        Xtr = torch.tensor(task["X_train"], dtype=torch.float32, device=device)
        ytr = torch.tensor(task["y_train"], dtype=torch.long, device=device)
        model.train()
        for ep in range(epochs):
            idx = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                b = idx[i : i + 32]
                opt.zero_grad()
                loss = crit(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
        model.eval()
        for j in range(len(tasks)):
            Xt = torch.tensor(tasks[j]["X_test"], dtype=torch.float32, device=device)
            yt = tasks[j]["y_test"]
            with torch.no_grad():
                pred = torch.argmax(model(Xt), dim=1).cpu().numpy()
            task_acc[j, k] = float((pred == yt).mean())
    return task_acc


def ewc_train_split_digits(
    tasks, in_dim=64, hidden=64, epochs=10, lr=1e-3, ewc_lambda=50.0, seed=0, device="cpu"
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for ewc_train_split_digits")
    set_torch_seed(seed)
    model = TorchMLP(in_dim, hidden, out_dim=10).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    opt_params = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    task_acc = np.zeros((len(tasks), len(tasks)))
    for k, task in enumerate(tasks):
        Xtr = torch.tensor(task["X_train"], dtype=torch.float32, device=device)
        ytr = torch.tensor(task["y_train"], dtype=torch.long, device=device)
        model.train()
        for ep in range(epochs):
            idx = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 32):
                b = idx[i : i + 32]
                opt.zero_grad()
                out = model(Xtr[b])
                loss = crit(out, ytr[b])
                if k > 0:
                    ewc = 0
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            ewc = ewc + (fisher[n] * (p - opt_params[n]) ** 2).sum()
                    loss = loss + 0.5 * ewc_lambda * ewc
                loss.backward()
                opt.step()
        # estimate fisher on this task
        model.eval()
        new_fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        for i in range(len(Xtr)):
            opt.zero_grad()
            out = model(Xtr[i : i + 1])
            loss = crit(out, ytr[i : i + 1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_fisher[n] += p.grad.detach() ** 2
        for n in new_fisher:
            new_fisher[n] /= max(1, len(Xtr))
            fisher[n] = fisher[n] + new_fisher[n]
            if n in model.state_dict():
                opt_params[n] = model.state_dict()[n].detach().clone()
        # evaluate on all tasks
        model.eval()
        for j in range(len(tasks)):
            Xt = torch.tensor(tasks[j]["X_test"], dtype=torch.float32, device=device)
            yt = tasks[j]["y_test"]
            with torch.no_grad():
                pred = torch.argmax(model(Xt), dim=1).cpu().numpy()
            task_acc[j, k] = float((pred == yt).mean())
    return task_acc


def replay_train_split_digits(
    tasks, in_dim=64, hidden=64, epochs=10, lr=1e-3, buffer_per_task=20, seed=0, device="cpu"
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for replay_train_split_digits")
    rng = np.random.default_rng(seed)
    set_torch_seed(seed)
    model = TorchMLP(in_dim, hidden, out_dim=10).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    buffer_X, buffer_y = [], []
    task_acc = np.zeros((len(tasks), len(tasks)))
    for k, task in enumerate(tasks):
        Xtr = task["X_train"]
        ytr = task["y_train"]
        # sample buffer from current task
        idx = rng.choice(len(Xtr), size=min(buffer_per_task, len(Xtr)), replace=False)
        buffer_X.append(Xtr[idx].copy())
        buffer_y.append(ytr[idx].copy())
        # train on current + stored past buffers (excluding current duplicate)
        if len(buffer_X) > 1:
            BX = np.concatenate([Xtr] + buffer_X[:-1], axis=0)
            BY = np.concatenate([ytr] + buffer_y[:-1], axis=0)
        else:
            BX = Xtr
            BY = ytr
        Xtr_t = torch.tensor(BX, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(BY, dtype=torch.long, device=device)
        model.train()
        for ep in range(epochs):
            idxp = torch.randperm(len(Xtr_t))
            for i in range(0, len(Xtr_t), 32):
                b = idxp[i : i + 32]
                opt.zero_grad()
                loss = crit(model(Xtr_t[b]), ytr_t[b])
                loss.backward()
                opt.step()
        model.eval()
        for j in range(len(tasks)):
            Xt = torch.tensor(tasks[j]["X_test"], dtype=torch.float32, device=device)
            yt = tasks[j]["y_test"]
            with torch.no_grad():
                pred = torch.argmax(model(Xt), dim=1).cpu().numpy()
            task_acc[j, k] = float((pred == yt).mean())
    return task_acc


# ---- Additional baselines for comparison ----


class FuzzyART:
    """Minimal Fuzzy ART (Carpenter et al.) classifier with adaptive/vigilance-like control.
    Prototype/category learning is local; included as mechanistic baseline.
    """

    def __init__(self, input_dim, n_classes=None, rho=0.7, alpha=1e-4, beta=1.0, choice=0.001, seed=0):
        self.input_dim = input_dim
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.choice = float(choice)
        self.rng = np.random.default_rng(seed)
        # complement coding
        self.categories = []  # each: w (2*D), label_counts
        self.label_map = {}
        self.n_learned = 0

    def _cc(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        # assume normalized [0,1] already; complement code
        x = np.clip(x, 0, 1)
        return np.concatenate([x, 1.0 - x])

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def partial_fit(self, x, y):
        xcc = self._cc(x)
        D = len(xcc)
        if not self.categories:
            self.categories.append({"w": xcc.copy(), "counts": {int(y): 1}, "total": 1})
            self.label_map[0] = int(y)
            self.n_learned += 1
            return self
        # choice function
        T = []
        for cat in self.categories:
            w = cat["w"]
            norm_w = np.sum(w)
            norm_x = np.sum(xcc)
            intersection = np.sum(np.minimum(xcc, w))
            Tj = intersection / (self.alpha + norm_w)
            T.append((Tj, intersection / (norm_x + 1e-12), cat))
        T.sort(reverse=True, key=lambda z: z[0])
        for Tj, mj, cat in T:
            if mj >= self.rho - 1e-12:  # vigilance test (resonance)
                w = cat["w"]
                # learn
                cat["w"] = self.beta * np.minimum(xcc, w) + (1 - self.beta) * w
                yi = int(y)
                cat["counts"][yi] = cat["counts"].get(yi, 0) + 1
                cat["total"] += 1
                return self
        # create new category
        self.categories.append({"w": xcc.copy(), "counts": {int(y): 1}, "total": 1})
        self.n_learned += 1
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for xi in X:
            xcc = self._cc(xi)
            if not self.categories:
                preds.append(0)
                continue
            best_m = -1
            best_cat = 0
            for ci, cat in enumerate(self.categories):
                w = cat["w"]
                intersection = np.sum(np.minimum(xcc, w))
                mj = intersection / (np.sum(xcc) + 1e-12)
                if mj > best_m:
                    best_m = mj
                    best_cat = ci
            cat = self.categories[best_cat]
            # predict by majority label in category
            if not cat["counts"]:
                preds.append(0)
            else:
                lab = max(cat["counts"].items(), key=lambda kv: kv[1])[0]
                preds.append(lab)
        return np.array(preds)

    def score(self, X, y):
        p = self.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))


class GNGU:
    """Growing Neural Gas with utility (Fritzke-style, simplified) classifier.
    Incremental prototype graph-like learning with node insertion/deletion by utility/error.
    Included as mechanistic baseline vs HSAM slot allocation.
    """

    def __init__(self, input_dim, max_nodes=200, age_max=50, lambda_=100, alpha=0.5, beta=0.0005, d=0.995, seed=0):
        self.input_dim = input_dim
        self.max_nodes = int(max_nodes)
        self.age_max = int(age_max)
        self.lambda_ = int(lambda_)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.d = float(d)
        self.rng = np.random.default_rng(seed)
        self.units = []  # list of dicts
        self.age = {}
        self.step = 0
        self.errors = []

    def _dist(self, a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(a - b))

    def partial_fit(self, x, y):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = int(y)
        if len(self.units) == 0:
            self.units.append({"w": x.copy(), "error": 0.0, "utility": 1.0, "counts": {y: 1}, "total": 1})
            self.step += 1
            return self
        if len(self.units) == 1:
            self.units.append({"w": x.copy() + 1e-6, "error": 0.0, "utility": 1.0, "counts": {y: 1}, "total": 1})
        # find nearest and second nearest
        dists = [(self._dist(x, u["w"]), i) for i, u in enumerate(self.units)]
        dists.sort(key=lambda z: z[0])
        s1_idx = dists[0][1]
        s2_idx = dists[1][1] if len(dists) > 1 else s1_idx
        s1 = self.units[s1_idx]
        s2 = self.units[s2_idx]
        e1 = self._dist(x, s1["w"])
        s1["error"] += e1 * e1
        # update s1 and neighbors (no explicit edge list tracked fully; approximate by nearest)
        s1["w"] += 0.1 * (x - s1["w"])
        if len(self.units) > 1:
            s2["w"] += 0.01 * (x - s2["w"])
        s1["counts"][y] = s1["counts"].get(y, 0) + 1
        s1["total"] += 1
        # decrease error
        for u in self.units:
            u["error"] *= 0.999
            u["utility"] *= self.d
        s1["utility"] += self.alpha * (s2["error"] - s1["error"])  # GNG-U style approximation
        self.step += 1
        if self.step % self.lambda_ == 0 and len(self.units) < self.max_nodes:
            # insert new node between unit with largest error and its neighbor (approx neighbor = s2)
            q = max(range(len(self.units)), key=lambda i: self.units[i]["error"])
            u_q = self.units[q]
            u_r = s2
            w_new = (u_q["w"] + u_r["w"]) / 2.0
            new_u = {"w": w_new, "error": (u_q["error"] + u_r["error"]) / 2.0, "utility": 1.0, "counts": dict(u_q["counts"]), "total": u_q["total"]}
            self.units.append(new_u)
            u_q["error"] *= self.alpha
            u_r["error"] *= self.alpha
            u_q["utility"] *= self.d
        # utility-based deletion
        if len(self.units) > 2 and self.step % 10 == 0:
            if self.units:
                worst_util = min(self.units, key=lambda u: u["utility"])
                if worst_util["utility"] < 1e-6 and worst_util["total"] < 5:
                    try:
                        self.units.remove(worst_util)
                    except Exception:
                        pass
        return self

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        for xi in X:
            x = np.asarray(xi, dtype=np.float64).reshape(-1)
            if not self.units:
                preds.append(0)
                continue
            dists = [(self._dist(x, u["w"]), i) for i, u in enumerate(self.units)]
            dists.sort(key=lambda z: z[0])
            u_best = self.units[dists[0][1]]
            if not u_best["counts"]:
                preds.append(0)
            else:
                lab = max(u_best["counts"].items(), key=lambda kv: kv[1])[0]
                preds.append(lab)
        return np.array(preds)

    def score(self, X, y):
        p = self.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))


class StreamingLDA:
    """Streaming Linear Discriminant Analysis (SLDA) - prototype/class-mean based.
    Closest mechanistic comparator to prototype memories; online updates to class means/covariance inverse.
    """

    def __init__(self, input_dim, shrinkage=1e-4, seed=0):
        self.input_dim = input_dim
        self.shrinkage = shrinkage
        self.classes = set()
        self.counts = {}
        self.means = {}
        # shared covariance approximation (pooled) - use regularized running covariance
        self.S = shrinkage * np.eye(input_dim, dtype=np.float64)
        self.n = 0
        self.total_mean = np.zeros(input_dim, dtype=np.float64)
        self._is_fitted = False

    def partial_fit(self, x, y):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = int(y)
        self.classes.add(y)
        self.counts[y] = self.counts.get(y, 0) + 1
        if y not in self.means:
            self.means[y] = x.copy()
        else:
            # online mean update
            c = self.counts[y]
            self.means[y] = (self.means[y] * (c - 1) + x) / c
        # running pooled covariance (within-class approximation)
        self.n += 1
        self.total_mean = (self.total_mean * (self.n - 1) + x) / self.n
        # update scatter
        xc = x - self.total_mean
        self.S = ((self.n - 1) / max(1, self.n)) * self.S + (1.0 / max(1, self.n)) * np.outer(xc, xc) + self.shrinkage * np.eye(self.input_dim) / self.n
        self._is_fitted = True
        return self

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        for xi, yi in zip(X, y):
            self.partial_fit(xi, yi)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self._is_fitted or not self.means:
            return np.zeros(len(X), dtype=int)
        try:
            # regularized inverse
            Sigma = self.S.copy()
            Sigma += self.shrinkage * np.eye(self.input_dim)
            invS = np.linalg.pinv(Sigma)
        except Exception:
            invS = np.linalg.pinv(self.S + self.shrinkage * np.eye(self.input_dim))
        preds = []
        for xi in X:
            x = np.asarray(xi, dtype=np.float64).reshape(-1)
            best_score = -1e18
            best_lab = list(self.means.keys())[0]
            for lab, mu in self.means.items():
                # LDA score: x^T Sigma^-1 mu - 0.5 mu^T Sigma^-1 mu + log p(y)
                score = float(x @ invS @ mu - 0.5 * mu @ invS @ mu + np.log(max(1e-12, self.counts.get(lab, 1) / max(1, self.n))))
                if score > best_score:
                    best_score = score
                    best_lab = lab
            preds.append(int(best_lab))
        return np.array(preds)

    def score(self, X, y):
        p = self.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))


class AGEM:
    """Averaged Gradient Episodic Memory (Lopez-Paz & Ranzato) - lightweight implementation
    using small episodic memory and reference gradients.
    """

    def __init__(self, in_dim, hidden=64, out_dim=10, memory_size=100, seed=0, lr=1e-3, epochs=5):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for AGEM")
        set_torch_seed(seed)
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.memory_size = int(memory_size)
        self.lr = lr
        self.epochs = int(epochs)
        self.seed = seed
        self.model = TorchMLP(in_dim, hidden, out_dim).to("cpu")
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.crit = nn.CrossEntropyLoss()
        self.mem_x = []
        self.mem_y = []
        self.rng = np.random.default_rng(seed)

    def _add_memory(self, X, y, task_id=None):
        # reservoir-like add
        for xi, yi in zip(X, y):
            if len(self.mem_x) < self.memory_size:
                self.mem_x.append(np.asarray(xi).copy())
                self.mem_y.append(int(yi))
            else:
                j = int(self.rng.integers(0, len(self.mem_x)))
                if self.rng.random() < self.memory_size / max(1, (len(self.mem_x) + 1)):
                    self.mem_x[j] = np.asarray(xi).copy()
                    self.mem_y[j] = int(yi)

    def train_task(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        for ep in range(self.epochs):
            self.model.train()
            idx = torch.randperm(len(X_t))
            for i in range(0, len(X_t), 32):
                b = idx[i:i+32]
                self.opt.zero_grad()
                loss = self.crit(self.model(X_t[b]), y_t[b])
                loss.backward()
                # project if memory exists
                if len(self.mem_x) > 0:
                    # reference gradient
                    self.opt.zero_grad()
                    m = min(len(self.mem_x), 32)
                    mx = torch.tensor(np.array(self.mem_x[:m]), dtype=torch.float32)
                    my = torch.tensor(np.array(self.mem_y[:m]), dtype=torch.long)
                    loss_ref = self.crit(self.model(mx), my)
                    loss_ref.backward(retain_graph=True)
                    g_ref = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
                    self.opt.zero_grad()
                    loss_cur = self.crit(self.model(X_t[b]), y_t[b])
                    loss_cur.backward()
                    # simple projection approximation: if gradient goes up in ref direction badly, do nothing complex - just step (lightweight)
                    # full A-GEM projection is heavier; do gradient surgery approx
                    for p, gr in zip(self.model.parameters(), g_ref):
                        if p.grad is None:
                            continue
                        # project if dot product negative
                        if torch.dot(p.grad.view(-1), gr.view(-1)) < 0:
                            p.grad = p.grad - torch.dot(p.grad.view(-1), gr.view(-1)) / (torch.dot(gr.view(-1), gr.view(-1)) + 1e-12) * gr
                    self.opt.step()
                else:
                    self.opt.step()
        self._add_memory(X, y)
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(X_t)
            return torch.argmax(out, dim=1).cpu().numpy()

    def score(self, X, y):
        p = self.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))


class ReplayMLPTuned:
    """Tuned replay with larger buffer."""

    def __init__(self, in_dim, hidden=64, out_dim=10, buffer_per_task=100, seed=0, lr=1e-3, epochs=15):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for ReplayMLPTuned")
        set_torch_seed(seed)
        self.model = TorchMLP(in_dim, hidden, out_dim).to("cpu")
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.crit = nn.CrossEntropyLoss()
        self.buffer_per_task = int(buffer_per_task)
        self.buffer_X = []
        self.buffer_y = []
        self.rng = np.random.default_rng(seed)
        self.epochs = int(epochs)

    def train_task(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        idx_buf = self.rng.choice(len(X), size=min(self.buffer_per_task, len(X)), replace=False)
        self.buffer_X.append(X[idx_buf].copy())
        self.buffer_y.append(y[idx_buf].copy())
        if len(self.buffer_X) > 1:
            BX = np.concatenate([X] + self.buffer_X[:-1], axis=0)
            BY = np.concatenate([y] + self.buffer_y[:-1], axis=0)
        else:
            BX = X
            BY = y
        X_t = torch.tensor(BX, dtype=torch.float32)
        y_t = torch.tensor(BY, dtype=torch.long)
        for ep in range(self.epochs):
            self.model.train()
            idxp = torch.randperm(len(X_t))
            for i in range(0, len(X_t), 32):
                b = idxp[i:i+32]
                self.opt.zero_grad()
                loss = self.crit(self.model(X_t[b]), y_t[b])
                loss.backward()
                self.opt.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            out = self.model(X_t)
            return torch.argmax(out, dim=1).cpu().numpy()

    def score(self, X, y):
        p = self.predict(X)
        return float(np.mean(np.asarray(p) == np.asarray(y)))
