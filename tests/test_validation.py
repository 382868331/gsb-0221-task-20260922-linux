"""Input contract and error-semantics tests."""

import math
import unittest

from himeb import analyze, fit_alpha, log_evidence, log_evidence_one, posterior_group
from himeb import maximize_on_interval
from himeb import MAX_GROUPS


class ValidationTests(unittest.TestCase):
    def test_reject_nan_inf_alpha(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    log_evidence(bad, [(1, 1)])
                with self.assertRaises(ValueError):
                    posterior_group((1, 1), bad)

    def test_reject_nonpositive_alpha(self):
        for bad in (0, -1, -0.5):
            with self.assertRaises((ValueError, TypeError)):
                log_evidence_one(bad, (1, 1))

    def test_bool_rejected_everywhere(self):
        # bool is a subclass of int; the contract rejects it explicitly.
        with self.assertRaises(TypeError):
            log_evidence(True, (1, 1))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            posterior_group((1, 1), False)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            log_evidence_one(1.0, (True, 1))  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            fit_alpha([(1, 1)], grid_points=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fit_alpha([(1, 1)], max_iter=False)  # type: ignore[arg-type]

    def test_reject_float_counts(self):
        with self.assertRaises(TypeError):
            log_evidence(1.0, [(1.0, 1)])  # type: ignore[list-item]

    def test_reject_negative_count_with_location(self):
        with self.assertRaises(ValueError) as ctx:
            fit_alpha([(1, 1), (2, -3)])
        self.assertIn("group 1", str(ctx.exception))
        self.assertIn("class 1", str(ctx.exception))

    def test_reject_bad_k(self):
        with self.assertRaises(ValueError):
            fit_alpha([(5,)])
        with self.assertRaises(ValueError):
            fit_alpha([tuple([1] * 7)])
        with self.assertRaises(ValueError):
            fit_alpha([()])

    def test_reject_inconsistent_lengths(self):
        with self.assertRaises(ValueError) as ctx:
            fit_alpha([(1, 1, 0), (1, 1)])
        self.assertIn("group 1", str(ctx.exception))

    def test_reject_empty_or_oversized_batch(self):
        with self.assertRaises(ValueError):
            fit_alpha([])
        with self.assertRaises(ValueError):
            fit_alpha([(1, 1)] * (MAX_GROUPS + 1))

    def test_reject_huge_count(self):
        with self.assertRaises(ValueError):
            log_evidence_one(1.0, (2**53 + 1, 0))

    def test_type_mismatches(self):
        with self.assertRaises(TypeError):
            fit_alpha("(1,1)")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fit_alpha([11])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            log_evidence("1.0", [(1, 1)])  # type: ignore[arg-type]

    def test_search_parameter_bounds(self):
        with self.assertRaises(ValueError):
            fit_alpha([(1, 1)], grid_points=2)
        with self.assertRaises(ValueError):
            fit_alpha([(1, 1)], max_iter=0)

    def test_objective_must_be_finite(self):
        with self.assertRaises(ValueError):
            maximize_on_interval(lambda a: float("nan"))
        with self.assertRaises(ValueError):
            maximize_on_interval(lambda a: float("-inf"))

    def test_no_partial_state_on_batch_failure(self):
        # All operations are pure: a rejected batch must raise before any
        # result exists and leave nothing to roll back. Calling again with a
        # valid batch works identically to a fresh process.
        good = [(3, 1), (0, 2)]
        with self.assertRaises(ValueError):
            analyze([(1, 1), (1, -1)])
        r1 = analyze(good)
        r2 = analyze(good)
        self.assertEqual(r1.fit.alpha, r2.fit.alpha)
        self.assertEqual(r1.fit.log_evidence, r2.fit.log_evidence)


if __name__ == "__main__":
    unittest.main()
