"""Synthetic boundaries for the matched-baseline extension; no corpus fixtures."""
import unittest

import numpy as np
import pandas as pd

from auditlib.consequences import design, split_groups
from auditlib.incremental import (block_features, check_endpoints, decomposition,
                                  fit, path_contrasts, pooled_mse)


class IncrementalBoundaries(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1703)
        names = ["length", "frequency", "position", "wrong_position", "stored", "canonical",
                 "lag_length", "lag_frequency", "lag_stored", "lag_canonical",
                 "wrong_lag_length", "wrong_lag_frequency", "wrong_lag_stored", "wrong_lag_canonical"]
        self.frame = pd.DataFrame(rng.normal(size=(80, len(names))), columns=names)

    def test_crossed_blocks_change_only_the_declared_predictors(self):
        f00 = block_features(self.frame, "stored", 0, 0)
        f10 = block_features(self.frame, "stored", 1, 0)
        f01 = block_features(self.frame, "stored", 0, 1)
        np.testing.assert_array_equal(f00[["lag_length", "lag_frequency", "lag_surprisal"]],
                                      f10[["lag_length", "lag_frequency", "lag_surprisal"]])
        np.testing.assert_array_equal(f00.position, f01.position)
        np.testing.assert_array_equal(f10.position, self.frame.position)
        np.testing.assert_array_equal(f01.lag_surprisal, self.frame.lag_stored)
        self.assertFalse(np.array_equal(f00.position, f10.position))
        self.assertFalse(np.array_equal(f00.lag_surprisal, f01.lag_surprisal))
        with self.assertRaises(ValueError):
            block_features(self.frame, "canonical", 2, 1)

    def test_nuisance_design_does_not_depend_on_score_source_or_test_values(self):
        a = block_features(self.frame, "stored", 1, 1)
        b = block_features(self.frame, "canonical", 1, 1)
        x, z = design(a.iloc[:60], a.iloc[60:])
        y, w = design(b.iloc[:60], b.iloc[60:])
        np.testing.assert_array_equal(x[:, :9], y[:, :9])
        np.testing.assert_array_equal(z[:, :9], w[:, :9])
        changed = a.iloc[60:].copy()
        changed["length"] += 10000
        changed["surprisal"] -= 10000
        after, held = design(a.iloc[:60], changed)
        np.testing.assert_array_equal(x, after)
        self.assertFalse(np.array_equal(z, held))

    def test_incremental_value_may_be_negative_with_exact_bookkeeping(self):
        n = {"00": 12., "10": 10., "01": 7., "11": 4.}
        f = {"00": 11., "10": 11., "01": 5., "11": 0.}
        result = decomposition(n, f)
        self.assertEqual(result["I"], {"00": 1., "10": -1., "01": 2., "11": 4.})
        self.assertEqual(result["joint_full_gain"], 11.)
        self.assertEqual(result["joint_nuisance_gain"], 8.)
        self.assertEqual(result["incremental_change"], 3.)
        self.assertEqual(result["identity_residual"], 0.)
        self.assertEqual(result["I_paths"]["position_at_L0"], -2.)
        self.assertEqual(result["I_paths"]["position_at_L1"], 2.)
        self.assertEqual(result["I_paths"]["nonadditivity_position_L1_minus_L0"], 4.)
        self.assertEqual(result["N_paths"]["nonadditivity_position_L1_minus_L0"], 1.)

    def test_zero_baseline_is_undefined_ratio_not_zero_increment(self):
        n = dict.fromkeys(("00", "10", "01", "11"), 0.)
        f = dict.fromkeys(n, 1.)
        result = decomposition(n, f)
        self.assertTrue(all(v is None for v in result["I_over_N"].values()))
        self.assertFalse(any(result["I_over_N_defined"].values()))
        self.assertTrue(all(v == -1 for v in result["I"].values()))
        zero = decomposition(n, n)
        self.assertTrue(all(v == 0 for v in zero["I"].values()))

    def test_incomplete_nonfinite_or_negative_losses_fail(self):
        good = dict.fromkeys(("00", "10", "01", "11"), 1.)
        for bad in ({"00": 1.}, {**good, "11": np.nan}, {**good, "11": -1.}):
            with self.assertRaises(ValueError):
                decomposition(good, bad)
        with self.assertRaises(ValueError):
            path_contrasts({**good, "00": np.inf})

    def test_pooled_loss_weights_rows_not_folds(self):
        # Two held-out folds of sizes2 and1 have losses1 and9; pooled loss=11/3.
        self.assertAlmostEqual(pooled_mse(np.zeros(3), np.array([1., 1., 3.])), 11 / 3)
        self.assertNotEqual(pooled_mse(np.zeros(3), np.array([1., 1., 3.])), (1 + 9) / 2)
        with self.assertRaises(ValueError):
            pooled_mse(np.zeros(3), np.array([1., np.nan, 3.]))

    def test_nonfinite_rank_deficient_and_shape_invalid_fits_fail(self):
        for matrix in (np.ones((60, 9)), np.full((60, 9), np.nan)):
            with self.assertRaises(ValueError):
                fit(matrix, np.ones((20, 9)), np.ones(60))
        with self.assertRaises(ValueError):
            fit(np.eye(9), np.ones((2, 8)), np.ones(9))

    def test_paired_nested_models_share_disjoint_folds_and_recover_signal(self):
        data = block_features(self.frame, "canonical", 1, 1)
        groups = np.repeat(np.arange(20), 4)
        y = 200 + 4 * data.surprisal.to_numpy()
        pred_n, pred_f = np.full(80, np.nan), np.full(80, np.nan)
        visits = np.zeros(80, int)
        first = list(split_groups(groups, 5, 61))
        again = list(split_groups(groups, 5, 61))
        for (tr, te), (tr2, te2) in zip(first, again):
            self.assertFalse(set(groups[tr]) & set(groups[te]))
            np.testing.assert_array_equal(tr, tr2)
            np.testing.assert_array_equal(te, te2)
            visits[te] += 1
            a, b = design(data.iloc[tr], data.iloc[te])
            pred_n[te], _ = fit(a[:, :9], b[:, :9], y[tr])
            pred_f[te], _ = fit(a, b, y[tr])
        np.testing.assert_array_equal(visits, np.ones(80))
        np.testing.assert_allclose(pred_f, y, atol=1e-9, rtol=0)
        self.assertGreater(pooled_mse(y, pred_n) - pooled_mse(y, pred_f), 1.)

    def test_changed_original_endpoint_cannot_pass(self):
        full = {"00": 9., "10": 7., "01": 5., "11": 4.}
        prior = {"wrong_order_rmse_ms": 3., "correct_order_rmse_ms": 2., "mse_decrease_ms_squared": 5.}
        self.assertEqual(check_endpoints(full, prior), 0.)
        with self.assertRaises(ValueError):
            check_endpoints({**full, "11": 4.01}, prior)


if __name__ == "__main__":
    unittest.main()
