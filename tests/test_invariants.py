"""Synthetic fixtures only; no corpus observations or copied source examples."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from auditlib.data import keyed_predecessor
from auditlib.consequences import split_groups, design, predict
from auditlib.scoring import substitution_ids
from auditlib.output import write_json


class Invariants(unittest.TestCase):
    def test_adjacency_survives_shuffle_and_keeps_gaps(self):
        frame = pd.DataFrame({"sent_id": [1, 1, 1, 2], "context_length": [3, 1, 2, 2], "s": [30., 10., 20., 50.]})
        expected = np.array([20., np.nan, 10., np.nan])
        np.testing.assert_allclose(keyed_predecessor(frame, frame.s), expected, equal_nan=True)
        shuffled = frame.sample(frac=1, random_state=4)
        restored = pd.Series(keyed_predecessor(shuffled, shuffled.s), index=shuffled.index).sort_index()
        np.testing.assert_allclose(restored, expected, equal_nan=True)

    def test_groups_never_leak_and_every_row_is_tested_once(self):
        groups = np.repeat(np.arange(10), 3)
        counts = np.zeros(len(groups))
        for train, test in split_groups(groups, 5, 24):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            counts[test] += 1
        np.testing.assert_array_equal(counts, np.ones(len(groups)))

    def test_test_values_cannot_change_training_scale(self):
        rng = np.random.default_rng(81)
        names = ["length", "frequency", "position", "lag_length", "lag_frequency", "surprisal", "lag_surprisal"]
        frame = pd.DataFrame(rng.normal(size=(150, 7)), columns=names)
        original, _ = design(frame.iloc[:100], frame.iloc[100:])
        changed = frame.iloc[100:].copy()
        changed["length"] *= 10000
        after, _ = design(frame.iloc[:100], changed)
        np.testing.assert_array_equal(original, after)

    def test_held_out_linear_recovery(self):
        rng = np.random.default_rng(83)
        names = ["length", "frequency", "position", "lag_length", "lag_frequency", "surprisal", "lag_surprisal"]
        frame = pd.DataFrame(rng.normal(size=(150, 7)), columns=names)
        frame["sent_id"] = np.repeat(np.arange(30), 5)
        frame["response"] = 200 + 5 * frame.surprisal - 2 * frame.length
        np.testing.assert_allclose(predict(frame, "response", 19), frame.response, atol=1e-9, rtol=0)

    def test_retokenization_changes_lookup_not_conditioning_path(self):
        class SyntheticTokenizer:
            def decode(self, ids, **kwargs):
                return {7: "suffix"}[ids[0]]

            def encode(self, text, **kwargs):
                return {" suffix": [9, 10]}[text]
        target = [4, 7]
        self.assertEqual(substitution_ids(SyntheticTokenizer(), target), [4, 9])
        self.assertEqual(target, [4, 7])

    def test_output_symlink_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            original = folder / "input.txt"
            original.write_text("synthetic input")
            destination = folder / "results.json"
            destination.symlink_to(original)
            with self.assertRaises(ValueError):
                write_json(destination, {"result": 7})
            self.assertEqual(original.read_text(), "synthetic input")


if __name__ == "__main__":
    unittest.main()
