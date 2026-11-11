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
