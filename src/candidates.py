import numpy as np


class HSAM:
    """Homeostatic Surprise-Gated Associative Memory (C1).

    Backprop-free online prototype/associative memory with local delta updates,
    surprise/mismatch-gated allocation, utility-based replacement, and autonomous
    homeostatic tuning of plasticity/thresholds from endogenous error statistics.

    Novelty assessment context: closely related to prototype/nearest-neighbor
    learners, ART-family vigilance/allocation mechanisms, GWR/SOINN/GNG-style
    utility/growth ideas, and Hebbian associative memories. Any novelty claim is
    mechanistic/incremental and must be validated against prior art; this is not
    claimed as fundamentally new without evidence.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        n_slots=128,
        n_winners=1,
        seed=0,
        proj_dim=None,
        eta_k=0.05,
        eta_v=0.3,
        decay=0.999,
        init_vigilance_low=0.8,
        init_vigilance_high=1.5,
        u0=1.0,
        self_tune=True,
        sim_alloc_threshold=0.2,
        per_slot_adaptive=True,
    ):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.proj_dim = proj_dim or min(input_dim, 64)
        self.output_dim = output_dim
        self.n_slots = n_slots
        self.n_winners = n_winners
        self.eta_k = eta_k
        self.eta_v = eta_v
        self.decay = decay
        self.v_low = init_vigilance_low
        self.v_high = init_vigilance_high
        self.u0 = u0
        self.self_tune = self_tune
        self.sim_alloc_threshold = sim_alloc_threshold
        self.per_slot_adaptive = per_slot_adaptive
        # random projection (fixed)
        self.P = self.rng.normal(
            0, 1.0 / np.sqrt(max(1, input_dim)), size=(input_dim, self.proj_dim)
        ).astype(np.float32)
        self.keys = self.rng.normal(0, 0.01, size=(n_slots, self.proj_dim)).astype(np.float32)
        self.keys /= (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-8)
        self.values = np.zeros((n_slots, output_dim), dtype=np.float32) if output_dim else None
        self.scalar_values = np.zeros(n_slots, dtype=np.float32) if output_dim is None else None
        self.utility = np.zeros(n_slots, dtype=np.float32)
        self.filled = np.zeros(n_slots, dtype=bool)
        # per-slot adaptive learning rates based on local error history
        self.slot_eta_k = np.ones(n_slots, dtype=np.float32) * eta_k
        self.slot_eta_v = np.ones(n_slots, dtype=np.float32) * eta_v
        self.slot_err_ema = np.ones(n_slots, dtype=np.float32)
        self.slot_count = np.zeros(n_slots, dtype=np.float32)
        # per-class/per-cluster adaptive vigilance
        self.class_vigilance = {}
        self.class_err = {}
        # endogenous stats
        self.err_ema = 1.0
        self.err_sq_ema = 1.0
        self.improve_ema = 0.0
        self.last_err = None
        self.step = 0
        self.allocations = 0

    def encode(self, x):
        z = np.asarray(x, dtype=np.float32).reshape(-1) @ self.P
        n = np.linalg.norm(z) + 1e-8
        return (z / n).astype(np.float32)

    def _ensure_values(self, y):
        if self.values is None and self.scalar_values is None:
            y_arr = np.asarray(y)
            if y_arr.ndim == 0:
                self.output_dim = 1
                self.scalar_values = np.zeros(self.n_slots, dtype=np.float32)
            else:
                self.output_dim = int(y_arr.shape[0])
                self.values = np.zeros((self.n_slots, self.output_dim), dtype=np.float32)

    def _sims_masked(self, z):
        sims = self.keys @ z
        sims_masked = sims.copy()
        sims_masked[~self.filled] = -1e9
        return sims, sims_masked

    def predict(self, x):
        z = self.encode(x)
        sims, sims_masked = self._sims_masked(z)
        if not np.any(self.filled):
            if self.values is not None:
                return np.zeros(self.output_dim, dtype=np.float32), 0.0, np.array([], dtype=int), z
            return 0.0, 0.0, np.array([], dtype=int), z
        k = min(self.n_winners, int(np.sum(self.filled)))
        widx = np.argsort(sims_masked)[-k:][::-1]
        ws = sims_masked[widx]
        ex = np.exp(ws - np.max(ws))
        w = ex / (ex.sum() + 1e-8)
        conf = float(np.max(ws))
        if self.values is not None:
            pred = (w[:, None] * self.values[widx]).sum(axis=0)
        else:
            pred = float((w * self.scalar_values[widx]).sum())
        return pred, conf, widx, z

    def _allocate(self, z, y_arr):
        empty = np.where(~self.filled)[0]
        if len(empty) > 0:
            j = int(self.rng.choice(empty))
        else:
            j = int(np.argmin(self.utility))
        self.keys[j] = z
        if self.values is not None:
            self.values[j] = y_arr
        else:
            self.scalar_values[j] = float(y_arr)
        self.utility[j] = self.u0
        self.filled[j] = True
        self.allocations += 1
        # initialize per-slot tracking for new slot
        if getattr(self, 'per_slot_adaptive', False):
            self.slot_eta_k[j] = self.eta_k
            self.slot_eta_v[j] = self.eta_v
            self.slot_err_ema[j] = 1.0
            self.slot_count[j] = 0.0
        return j

    def learn(self, x, y):
        self._ensure_values(y)
        pred, conf, widx, z = self.predict(x)
        y_arr = np.asarray(y, dtype=np.float32)
        if self.values is not None and y_arr.ndim == 0:
            y_arr = np.array([y_arr], dtype=np.float32)
        pred_arr = np.asarray(pred, dtype=np.float32)
        err = (
            float(np.linalg.norm(y_arr - pred_arr))
            if self.values is not None
            else float(abs(float(y_arr) - float(pred_arr)))
        )
        # endogenous statistics
        self.err_ema = 0.99 * self.err_ema + 0.01 * err
        self.err_sq_ema = 0.99 * self.err_sq_ema + 0.01 * (err * err)
        surprise = err / (self.err_ema + 1e-6)
        if self.last_err is not None:
            self.improve_ema = 0.99 * self.improve_ema + 0.01 * (self.last_err - err)
        self.last_err = err
        # autonomous homeostatic tuning (bounded) - redesigned with per-slot adaptive rates
        if self.self_tune and self.step % 50 == 0:
            if self.improve_ema > 0:
                self.eta_k = max(0.01, self.eta_k * 0.995)
                self.eta_v = max(0.01, self.eta_v * 0.995)
                self.v_high = min(3.0, self.v_high * 1.002)
            else:
                self.eta_k = min(0.5, self.eta_k * 1.005)
                self.eta_v = min(0.5, self.eta_v * 1.005)
                self.v_low = max(0.2, self.v_low * 0.998)
                self.v_high = max(self.v_low + 0.2, self.v_high * 0.998)
            # Per-slot adaptive learning rates based on local error history
            if getattr(self, 'per_slot_adaptive', True) and np.any(self.filled):
                for si in np.where(self.filled)[0]:
                    if self.slot_count[si] > 0 and self.slot_err_ema[si] > 0:
                        # If slot has high local error, increase its learning rate; if low/stable, decrease
                        if self.slot_err_ema[si] > self.err_ema * 1.2:
                            self.slot_eta_k[si] = min(0.5, self.slot_eta_k[si] * 1.01)
                            self.slot_eta_v[si] = min(0.5, self.slot_eta_v[si] * 1.01)
                        elif self.slot_err_ema[si] < self.err_ema * 0.8:
                            self.slot_eta_k[si] = max(0.01, self.slot_eta_k[si] * 0.99)
                            self.slot_eta_v[si] = max(0.01, self.slot_eta_v[si] * 0.99)
            # Per-class/per-cluster adaptive vigilance
            if np.any(self.filled) and self.values is not None and self.output_dim and self.output_dim > 1:
                try:
                    pred_label = int(np.argmax(pred_arr))
                    target_label = int(np.argmax(y_arr))
                    for c in [pred_label, target_label]:
                        if c not in self.class_err:
                            self.class_err[c] = 1.0
                            self.class_vigilance[c] = self.v_high
                        self.class_err[c] = 0.99 * self.class_err[c] + 0.01 * err
                        if self.class_err[c] > self.err_ema * 1.1:
                            self.class_vigilance[c] = min(3.0, self.class_vigilance.get(c, self.v_high) * 1.005)
                        else:
                            self.class_vigilance[c] = max(self.v_low, self.class_vigilance.get(c, self.v_high) * 0.995)
                except Exception:
                    pass
        # mismatch / novelty conditions
        mismatch = False
        if np.any(self.filled) and self.values is not None and self.output_dim and self.output_dim > 1:
            try:
                pred_label = int(np.argmax(pred_arr))
                target_label = int(np.argmax(y_arr))
                mismatch = pred_label != target_label
            except Exception:
                mismatch = False
        sims, sims_masked = self._sims_masked(z)
        max_sim = float(np.max(sims_masked)) if np.any(self.filled) else -1e9
        low_sim = np.any(self.filled) and (max_sim < self.sim_alloc_threshold)
        # Use per-class vigilance if available
        alloc_thresh = self.v_high
        if np.any(self.filled) and self.values is not None and self.output_dim and self.output_dim > 1:
            try:
                target_label = int(np.argmax(y_arr))
                if target_label in self.class_vigilance:
                    alloc_thresh = self.class_vigilance[target_label]
            except Exception:
                pass
        allocate = (not np.any(self.filled)) or (surprise > alloc_thresh) or mismatch or low_sim
        if allocate:
            self._allocate(z, y_arr)
        else:
            k = min(self.n_winners, int(np.sum(self.filled)))
            widx = np.argsort(sims_masked)[-k:][::-1]
            ws = sims_masked[widx]
            ex = np.exp(ws - np.max(ws))
            w = ex / (ex.sum() + 1e-8)
            for wi, idx in zip(w, widx):
                # per-slot adaptive learning rates
                eta_k = self.slot_eta_k[idx] if getattr(self, 'per_slot_adaptive', False) else self.eta_k
                eta_v = self.slot_eta_v[idx] if getattr(self, 'per_slot_adaptive', False) else self.eta_v
                self.keys[idx] = (1 - eta_k * wi) * self.keys[idx] + (eta_k * wi) * z
                self.keys[idx] /= np.linalg.norm(self.keys[idx]) + 1e-8
                if self.values is not None:
                    self.values[idx] = self.values[idx] + eta_v * wi * (y_arr - self.values[idx])
                else:
                    self.scalar_values[idx] = self.scalar_values[idx] + eta_v * wi * (
                        float(y_arr) - self.scalar_values[idx]
                    )
                # track per-slot local error history
                self.slot_err_ema[idx] = 0.99 * self.slot_err_ema[idx] + 0.01 * err
                self.slot_count[idx] += 1.0
                self.utility[idx] = self.decay * self.utility[idx] + float(wi) * max(
                    0.0, 1.0 - 0.5 * surprise
                )
        self.utility *= 0.9995
        self.step += 1
        return {"err": err, "surprise": surprise, "allocate": bool(allocate), "pred": pred}

    def predict_vector(self, x):
        pred, conf, widx, z = self.predict(x)
        return np.asarray(pred, dtype=np.float32)

    def predict_class(self, x):
        pred, conf, widx, z = self.predict(x)
        pred_arr = np.asarray(pred)
        if self.values is not None and self.output_dim and self.output_dim > 1:
            return int(np.argmax(pred_arr))
        return int(np.round(float(pred_arr)))


class RLS_SelfGate:
    """Candidate B: fixed random features + online multi-output RLS-like updates
    with self-gated forgetting and a small residual correction memory.

    Incremental relative to adaptive RLS/random-feature online learners; included
    as an exploratory candidate and likely not strongly novel.
    """

    def __init__(self, input_dim, output_dim, n_features=256, seed=0, lam=1.0, forget=0.995):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_features = n_features
        self.lam = lam
        self.forget = forget
        self.Wproj = self.rng.normal(
            0, 1.0 / np.sqrt(max(1, input_dim)), size=(input_dim, n_features)
        ).astype(np.float64)
        self.bproj = self.rng.uniform(0, 2 * np.pi, size=(n_features,)).astype(np.float64)
        self.W = np.zeros((output_dim, n_features), dtype=np.float64)
        self.P = (1.0 / lam) * np.eye(n_features, dtype=np.float64)
        self.corr_phi = []
        self.corr_res = []
        self.max_corr = 64
        self.err_ema = 1.0
        self.improve_ema = 0.0
        self.last_err = None
        self.step = 0

    def phi(self, x):
        z = np.cos(np.asarray(x, dtype=np.float64).reshape(-1) @ self.Wproj + self.bproj)
        return z

    def predict(self, x):
        p = self.phi(x)
        base = self.W @ p
        corr = np.zeros_like(base)
        if self.corr_phi:
            Ph = np.stack(self.corr_phi, axis=0)
            sims = Ph @ p
            idx = int(np.argmax(sims))
            if sims[idx] > 0.5:
                corr = self.corr_res[idx] * float(sims[idx])
        return base + 0.5 * corr

    def learn(self, x, y):
        p = self.phi(x)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        pred = self.predict(x)
        err_vec = y - (self.W @ p)
        err = float(np.linalg.norm(y - pred))
        if self.last_err is not None:
            self.improve_ema = 0.995 * self.improve_ema + 0.005 * (self.last_err - err)
        self.last_err = err
        self.err_ema = 0.99 * self.err_ema + 0.01 * err
        forget = self.forget
        if self.improve_ema < 0:
            forget = min(0.999, self.forget + 0.001)
        else:
            forget = max(0.97, self.forget - 0.0005)
        Pp = self.P @ p
        denom = forget + float(p @ Pp)
        k = Pp / (denom + 1e-12)
        self.W = self.W + np.outer(err_vec, k)
        self.P = (self.P - np.outer(k, Pp)) / max(1e-6, forget)
        if err > 1.2 * self.err_ema:
            if len(self.corr_phi) >= self.max_corr:
                j = self.step % self.max_corr
                self.corr_phi[j] = p.copy()
                self.corr_res[j] = (y - pred).copy()
            else:
                self.corr_phi.append(p.copy())
                self.corr_res.append((y - pred).copy())
        self.step += 1
        return {"err": err, "forget": forget}

    def predict_class(self, x):
        return int(np.argmax(self.predict(x)))


class CSM:
    """Candidate C: simple variable-context count model (PPM/CTW-like family).

    Online next-symbol predictor using count tables for contexts 0..k with
    backoff and an endogenous heuristic that can grow/shrink context depth.
    This is not claimed novel relative to PPM/CTW/variable-order Markov models;
    included for self-improvement/stream experiments and comparison.
    """

    def __init__(self, n_symbols, max_context=6, init_context=1, seed=0):
        self.n = int(n_symbols)
        self.max_context = int(max_context)
        self.context = int(init_context)
        self.rng = np.random.default_rng(seed)
        self.tables = [dict() for _ in range(self.max_context + 1)]
        self.alpha = 0.1
        self.surprise_ema = 1.0
        self.step = 0

    def _key(self, history, k):
        if k <= 0:
            return ()
        if len(history) < k:
            return None
        return tuple(int(x) for x in history[-k:])

    def _counts_for(self, history, k):
        key = self._key(history, k)
        if key is None:
            return None
        return self.tables[k].get(key)

    def _probs(self, history, k):
        counts = np.ones(self.n, dtype=np.float64) * self.alpha
        used = False
        for kk in range(max(0, int(k)), -1, -1):
            c = self._counts_for(history, kk)
            if c is not None:
                counts = counts + c
                used = True
                break
        # if no specific context found, counts remain uniform prior
        probs = counts / (counts.sum() + 1e-12)
        return probs, used

    def predict(self, history):
        probs, _ = self._probs(history, self.context)
        return int(np.argmax(probs))

    def learn_step(self, history, symbol):
        sym = int(symbol)
        probs, _ = self._probs(history, self.context)
        p = float(max(1e-12, probs[sym]))
        surprise = float(-np.log(p))
        # update
        max_k = min(self.context, len(history))
        for k in range(0, max_k + 1):
            key = self._key(history, k)
            if key is None:
                continue
            tab = self.tables[k]
            if key not in tab:
                tab[key] = np.zeros(self.n, dtype=np.float64)
            tab[key][sym] += 1.0
        # endogenous adaptation heuristic
        self.surprise_ema = 0.995 * self.surprise_ema + 0.005 * surprise
        if self.step % 50 == 0 and self.step > 0 and self.max_context > 0:
            # simple bounded grow/shrink based on surprise trend
            if self.surprise_ema > 1.4 and self.context < self.max_context:
                self.context += 1
            elif self.surprise_ema < 0.9 and self.context > 0:
                self.context -= 1
        self.step += 1
        return {"surprise": surprise, "context": int(self.context)}
