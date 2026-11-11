import json
import os
import time
import subprocess
import random
from dataclasses import dataclass

import numpy as np

try:
    import psutil
except Exception:
    psutil = None
try:
    import resource
except Exception:
    resource = None


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except Exception:
        pass


@dataclass
class Stats:
    wall_seconds: float = 0.0
    peak_rss_mb: float = 0.0


class Tracker:
    """Simple wall-time and peak-RSS tracker."""

    def __init__(self):
        self.stats = Stats()
        self._t0 = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._peak = 0.0

    def _rss_mb(self):
        if self._proc is not None:
            try:
                return float(self._proc.memory_info().rss) / (1024.0 * 1024.0)
            except Exception:
                pass
        if resource is not None:
            try:
                # Linux: ru_maxrss in KB
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return float(rss) / 1024.0
            except Exception:
                pass
        return 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._peak = self._rss_mb()
        return self

    def update(self):
        if self._t0 is not None:
            self.stats.wall_seconds = float(time.perf_counter() - self._t0)
        self._peak = max(self._peak, self._rss_mb())
        self.stats.peak_rss_mb = float(self._peak)

    def __exit__(self, exc_type, exc, tb):
        self.update()
        return False


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit_hash(root="."):
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    vals = np.asarray(values, dtype=float)
    n = int(len(vals))
    rng = np.random.default_rng(seed)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(vals[idx]))
    boots.sort()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n": n}


def paired_bootstrap_ci(a, b, n_boot=2000, ci=0.95, seed=0):
    # CI for mean(a-b)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    diffs = a - b
    out = bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)
    out["diff_mean"] = out["mean"]
    return out
