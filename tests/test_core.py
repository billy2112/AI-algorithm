import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import set_seed, bootstrap_ci, save_json, load_json
from src.data import get_digits_split, split_digits_tasks, make_tiny_char_stream
from src.metrics import accuracy, average_accuracy, forgetting_measure
from src.candidates import HSAM, CSM


class TestUtils(unittest.TestCase):
    def test_seed_repro_numpy(self):
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_allclose(a, b)

    def test_bootstrap_ci(self):
        out = bootstrap_ci([1, 2, 3, 4, 5], n_boot=500, seed=0)
        for k in ["mean", "std", "ci_low", "ci_high", "n"]:
            self.assertIn(k, out)
        self.assertEqual(out["n"], 5)
        self.assertAlmostEqual(out["mean"], 3.0)
        self.assertLessEqual(out["ci_low"], out["mean"])
        self.assertGreaterEqual(out["ci_high"], out["mean"])

    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            obj = {"a": 1, "b": [1, 2, 3]}
            save_json(p, obj)
            obj2 = load_json(p)
            self.assertEqual(obj, obj2)


class TestDataMetrics(unittest.TestCase):
    def test_digits_split_deterministic(self):
        (Xtr1, ytr1), (Xte1, yte1) = get_digits_split(seed=0)
        (Xtr2, ytr2), (Xte2, yte2) = get_digits_split(seed=0)
        np.testing.assert_array_equal(Xtr1, Xtr2)
        np.testing.assert_array_equal(ytr1, ytr2)
        self.assertEqual(Xtr1.shape[1], 64)
        self.assertGreater(len(Xtr1), 0)
        self.assertGreater(len(Xte1), 0)

    def test_split_digits_tasks(self):
        tasks = split_digits_tasks(seed=1)
        self.assertEqual(len(tasks), 5)
        seen = set()
        for t in tasks:
            a, b = t["classes"]
            self.assertTrue(set(np.unique(t["y_train"])).issubset({a, b}))
            self.assertTrue(set(np.unique(t["y_test"])).issubset({a, b}))
            seen.update([a, b])
        self.assertEqual(seen, set(range(10)))

    def test_char_stream(self):
        seq, vocab, ch2i = make_tiny_char_stream(length=200, seed=0)
        self.assertEqual(len(seq), 200)
        self.assertGreater(len(vocab), 5)
        self.assertTrue(all(0 <= s < len(vocab) for s in seq))

    def test_metrics(self):
        self.assertAlmostEqual(accuracy([1, 2, 3], [1, 0, 3]), 2 / 3)
        mat = np.array([[1.0, 0.8, 0.7], [0.0, 1.0, 0.9], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(average_accuracy(mat), (0.7 + 0.9 + 1.0) / 3)
        # forgetting: task0 max past 1.0 final 0.7 -> 0.3, task1 max past 1.0 final 0.9 -> 0.1
        self.assertAlmostEqual(forgetting_measure(mat), (0.3 + 0.1) / 2)


class TestCandidates(unittest.TestCase):
    def test_hsam_learns_toy(self):
        set_seed(0)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 16)).astype(np.float32)
        y = rng.integers(0, 4, size=(32,))

        def oh(v):
            z = np.zeros(4, dtype=np.float32)
            z[int(v)] = 1.0
            return z

        m = HSAM(input_dim=16, output_dim=4, n_slots=64, n_winners=1, seed=0, proj_dim=16, eta_k=0.1, eta_v=0.8)
        for ep in range(40):
            idx = rng.permutation(32)
            for i in idx:
                m.learn(X[i], oh(y[i]))
        preds = [int(np.argmax(m.predict(X[i])[0])) for i in range(32)]
        acc = float((np.array(preds) == y).mean())
        self.assertGreaterEqual(acc, 0.9)
        self.assertGreaterEqual(int(m.filled.sum()), 1)

    def test_hsam_predict_shapes(self):
        m = HSAM(input_dim=8, output_dim=3, n_slots=10, seed=0)
        p, conf, widx, z = m.predict(np.ones(8, dtype=np.float32))
        self.assertEqual(np.asarray(p).shape, (3,))
        self.assertEqual(z.shape, (min(8, 64),))

    def test_csm_basic(self):
        m = CSM(n_symbols=5, max_context=3, init_context=1, seed=0)
        hist = [0, 1, 2, 0, 1]
        pred = m.predict(hist)
        self.assertTrue(0 <= pred < 5)
        info = m.learn_step(hist, 2)
        self.assertIn("surprise", info)
        self.assertIn("context", info)


if __name__ == "__main__":
    unittest.main()
