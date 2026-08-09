import unittest

import numpy as np

from activation_probing_lab.probes import bootstrap_auc, fit_regularized_probe


class ProbeTest(unittest.TestCase):
    def test_regularized_probe_recovers_a_linear_signal(self) -> None:
        rng = np.random.default_rng(3)
        samples = 160
        labels = np.tile([0, 1], samples // 2)
        groups = np.repeat(np.arange(40), 4)
        activations = rng.normal(0, 1, size=(samples, 24))
        activations[:, 0] += (labels * 2 - 1) * 2.0
        fitted = fit_regularized_probe(
            activations,
            labels,
            groups,
            c_grid=[0.01, 0.1, 1.0],
            cv_folds=4,
            seed=3,
        )
        self.assertGreater(fitted.cv_auc, 0.95)

    def test_bootstrap_interval_contains_strong_auc(self) -> None:
        labels = np.tile([0, 1], 100)
        scores = labels.astype(float) * 0.9 + 0.05
        low, high = bootstrap_auc(labels, scores, samples=50, seed=4)
        self.assertGreater(low, 0.95)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
