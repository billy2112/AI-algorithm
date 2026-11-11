import numpy as np

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    load_digits = None
    train_test_split = None


def get_digits_split(seed=0, test_size=0.2):
    if load_digits is None or train_test_split is None:
        raise RuntimeError("sklearn is required for digits dataset")
    d = load_digits()
    X = d.data.astype(np.float32)
    # normalize to [0,1]
    X = X / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return (Xtr, ytr), (Xte, yte)


def split_digits_tasks(seed=0, test_size=0.2, pairs=None):
    if pairs is None:
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed, test_size=test_size)
    tasks = []
    for a, b in pairs:
        mtr = (ytr == a) | (ytr == b)
        mte = (yte == a) | (yte == b)
        tasks.append(
            {
                "classes": (a, b),
                "X_train": Xtr[mtr].copy(),
                "y_train": ytr[mtr].copy(),
                "X_test": Xte[mte].copy(),
                "y_test": yte[mte].copy(),
            }
        )
    return tasks


def permuted_digits_task(seed=0, n_perms=3, test_size=0.2):
    rng = np.random.default_rng(seed)
    (Xtr, ytr), (Xte, yte) = get_digits_split(seed=seed, test_size=test_size)
    tasks = []
    D = Xtr.shape[1]
    for i in range(n_perms):
        perm = rng.permutation(D)
        tasks.append(
            {
                "perm": perm,
                "X_train": Xtr[:, perm].copy(),
                "y_train": ytr.copy(),
                "X_test": Xte[:, perm].copy(),
                "y_test": yte.copy(),
            }
        )
    return tasks


def make_tiny_char_stream(length=8000, seed=0):
    rng = np.random.default_rng(seed)
    base = (
        "to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of troubles "
        "and by opposing end them to die to sleep no more "
        "the quick brown fox jumps over the lazy dog pack my box with five dozen liquor jugs "
        "self learning systems adapt online from streams of experience with local updates "
    )
    text = base * 400
    vocab = sorted(list(set(text)))
    ch2i = {c: i for i, c in enumerate(vocab)}
    ids = np.array([ch2i[c] for c in text], dtype=np.int64)
    if len(ids) < length:
        reps = int(np.ceil(length / max(1, len(ids))))
        ids = np.tile(ids, reps)
    max_start = max(1, len(ids) - length + 1)
    start = int(rng.integers(0, max_start))
    seq = ids[start : start + length].copy()
    return seq, vocab, ch2i


def synthetic_copy_recall(n_samples=1000, seq_len=5, n_symbols=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_symbols, size=(n_samples, seq_len))
    y = X.copy()
    return X.astype(np.int64), y.astype(np.int64)


def synthetic_parity(n_samples=1000, n_bits=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.float32)
    y = (np.sum(X, axis=1) % 2).astype(np.int64)
    return X, y
