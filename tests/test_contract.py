"""Contract tests: validation, error semantics, search limits, determinism."""

import math
import unittest

from dirichlet_eb import core


class ValidationTests(unittest.TestCase):
    def test_reject_nan_and_infinity(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    core.log_evidence(bad, [(1, 1)], 2)
                with self.assertRaises(ValueError):
                    core.compute_posteriors(bad, [(1, 1)], 2)

    def test_reject_nonpositive_alpha(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                core.log_evidence(bad, [(1, 1)], 2)

    def test_reject_bool_counts(self):
        with self.assertRaises(TypeError):
            core.validate_counts([(True, 1)])
        with self.assertRaises(TypeError):
            core.validate_counts([(1, False)])
        # A bare bool is not accepted as a batch either.
        with self.assertRaises(TypeError):
            core.validate_counts(True)

    def test_reject_bool_integer_parameters(self):
        with self.assertRaises(TypeError):
            core.fit([(1, 1)], k=True)
        with self.assertRaises(TypeError):
            core.fit([(1, 1)], grid_points=True)
        with self.assertRaises(TypeError):
            core.fit([(1, 1)], max_iterations=False)

    def test_reject_float_counts(self):
        with self.assertRaises(TypeError):
            core.validate_counts([(1.0, 2)])

    def test_reject_negative_counts(self):
        with self.assertRaises(ValueError):
            core.validate_counts([(1, -1)])

    def test_reject_inconsistent_lengths(self):
        with self.assertRaises(core.InputValidationError):
            core.validate_counts([(1, 2), (1, 2, 3)])

    def test_reject_empty_batch_and_out_of_limits(self):
        with self.assertRaises(core.InputValidationError):
            core.validate_counts([])
        with self.assertRaises(core.InputValidationError):
            core.validate_counts([(0,)] * (core.MAX_GROUPS + 1))
        with self.assertRaises(core.InputValidationError):
            core.validate_counts([tuple(range(core.MAX_CATEGORIES + 1))])

    def test_errors_are_locatable_by_type(self):
        # All package errors are catchable as EBMError, validation errors
        # specifically as InputValidationError.
        with self.assertRaises(core.EBMError):
            core.validate_counts([(1, -1)])
        self.assertTrue(issubclass(core.InputValidationError, core.EBMError))

    def test_rejected_batch_leaves_no_partial_output(self):
        # fit must fail atomically: an invalid late group raises before any
        # FitResult exists.
        good = [(1, 0)] * 39
        bad = good + [(1, -2)]
        with self.assertRaises(ValueError):
            core.fit(bad)


class IdentifiabilityTests(unittest.TestCase):
    def test_all_empty_groups_raise_identifiability_not_search_error(self):
        with self.assertRaises(core.IdentifiabilityError):
            core.fit([(0, 0), (0, 0)])

    def test_k_equals_one_raises_identifiability(self):
        with self.assertRaises(core.IdentifiabilityError):
            core.fit([(3,), (1,)])

    def test_identifiability_is_distinct_from_budget_exhaustion(self):
        self.assertFalse(
            issubclass(core.IdentifiabilityError, core.SearchBudgetExhaustedError)
        )
        self.assertFalse(
            issubclass(core.SearchBudgetExhaustedError, core.IdentifiabilityError)
        )


class SearchLimitTests(unittest.TestCase):
    def test_budget_exhaustion_raises_specialized_error(self):
        # A one-iteration budget cannot shrink a bracket of width ~0.18 to
        # tol 1e-10.  This is a resource failure, reported as such.
        with self.assertRaises(core.SearchBudgetExhaustedError):
            core.fit(
                [(2, 0), (0, 2), (1, 1)],
                grid_points=101,
                tolerance=1e-12,
                max_iterations=1,
            )

    def test_bad_search_parameters_rejected(self):
        with self.assertRaises(core.InputValidationError):
            core.fit([(1, 1)], grid_points=4)
        with self.assertRaises(core.InputValidationError):
            core.fit([(1, 1)], tolerance=0.0)
        with self.assertRaises(core.InputValidationError):
            core.fit([(1, 1)], tolerance=1.5)
        with self.assertRaises(core.InputValidationError):
            core.fit([(1, 1)], max_iterations=0)
        with self.assertRaises(ValueError):
            core.fit([(1, 1)], tolerance=float("nan"))


class CandidateAndDeterminismTests(unittest.TestCase):
    def test_coarse_grid_can_report_multiple_basins(self):
        # A constant-ish constructed objective cannot come from real data, but
        # the grid machinery must still return every detected basin: with a
        # coarse grid and balanced data, endpoint plateau + interior argmax
        # cells can both be marked.  Here we verify candidate_count >= 1 and
        # that every candidate lies inside the declared interval.
        result = core.fit([(6, 6)] * 4, grid_points=11)
        self.assertGreaterEqual(result.candidate_count, 1)
        for cand in result.candidates:
            self.assertGreaterEqual(cand.log_alpha, math.log(core.ALPHA_MIN) - 1e-12)
            self.assertLessEqual(cand.log_alpha, math.log(core.ALPHA_MAX) + 1e-12)

    def test_results_are_deterministic(self):
        rows = [(2, 0), (0, 2), (1, 1), (0, 0)]
        r1 = core.fit(rows)
        r2 = core.fit(rows)
        self.assertEqual(r1.alpha, r2.alpha)
        self.assertEqual(r1.log_evidence, r2.log_evidence)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(
            tuple(c.log_alpha for c in r1.candidates),
            tuple(c.log_alpha for c in r2.candidates),
        )

    def test_frozen_result_structures(self):
        result = core.fit([(2, 0), (0, 2)])
        with self.assertRaises(Exception):
            result.alpha = 3.0
        with self.assertRaises(Exception):
            result.groups[0].mean = (0.9, 0.1)


if __name__ == "__main__":
    unittest.main()
