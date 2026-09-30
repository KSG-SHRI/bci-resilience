import unittest

import numpy as np

from experiment import mask, metrics


class ExperimentTests(unittest.TestCase):
    def test_mask_replaces_exact_channel_fraction_without_mutating_input(self):
        x = np.arange(80, dtype=float).reshape(2, 40)
        original = x.copy()
        median = np.full(40, -1.0)
        result = mask(x, median, np.random.default_rng(7), fraction=0.25)
        self.assertTrue(np.array_equal(x, original))
        self.assertEqual((result == -1).sum(axis=1).tolist(), [10, 10])
        self.assertTrue(np.array_equal(result[0] == -1, result[1] == -1))

    def test_metrics_reports_both_classes(self):
        result = metrics(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
        self.assertEqual(result["confusion"], [[1, 1], [1, 1]])
        self.assertEqual(result["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
