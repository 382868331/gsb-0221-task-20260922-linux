"""Tests for the empirical-Bayes fit: reference agreement, regimes, replicates."""

import math
import unittest

from dirichlet_eb import core

from tests import reference


class FitReferenceTests(unittest.TestCase):
    def _check_against_reference(self, rows):
        result = core.fit(rows)
        ref_alpha, ref_status = reference.ref_fit_alpha(rows)
        self.assertEqual(result.status, ref_status, msg=f"rows={rows}")
        if ref_status == "interior":
            # Independent ternary maximizer agreement: relative 1e-6.
            self.assertLess(
                abs(result.alpha - ref_alpha) / ref_alpha,
                1e-6,
                msg=f"rows={rows}: {result.alpha} vs {ref_alpha}",
            )
        else:
            self.assertEqual(result.alpha, ref_alpha)
        # Fitted alpha actually maximizes the reference evidence on a probe set.
        probe = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4]
        best_ref = reference.ref_log_evidence(rows, result.alpha)
        for a in probe:
            self.assertLessEqual(
                reference.ref_log_evidence(rows, a),
                best_ref + 1e-8,
                msg=f"rows={rows} alpha={a} beats fitted {result.alpha}",
            )

    def test_interior_mode_mixture(self):
        self._check_against_reference([(2, 0), (0, 2), (1, 1)])

    def test_lower_endpoint_single_category_extreme(self):
        self._check_against_reference([(5, 0), (3, 0), (0, 4), (0, 2)])

    def test_upper_endpoint_balanced(self):
        self._check_against_reference([(6, 6)] * 4)

    def test_various_small_datasets(self):
        for rows in [
            [(2, 1)],
            [(0, 2)],
            [(1, 2), (2, 1), (0, 3), (3, 0)],
            [(2, 2), (1, 1), (3, 2), (1, 3)],
            [(10, 0), (0, 10), (10, 10)],
            [(0, 0), (3, 1), (0, 0)],
        ]:
            self._check_against_reference(rows)

    def test_single_draw_groups_are_nonidentifiable(self):
        # A single observation has marginal 1/K independent of alpha.
        for rows in [[(1, 0)], [(0, 1)], [(1, 0), (0, 1)], [(0, 0), (1, 0)]]:
            with self.assertRaises(core.IdentifiabilityError, msg=f"rows={rows}"):
                core.fit(rows)

    def test_group_replicates_drive_alpha_toward_heterogeneity(self):
        # One group: alpha near the boundary behavior of a single beta-binomial.
        one = core.fit([(4, 1)])
        # Replicating an *identical* group must not change the MLE at all:
        # evidence is a fixed sum of per-group functions, and the group is the
        # same each time, so maximizing f is the same as maximizing 4*f.
        rep = core.fit([(4, 1)] * 4)
        self.assertEqual(one.status, rep.status)
        self.assertAlmostEqual(one.alpha, rep.alpha, places=8)
        self.assertAlmostEqual(one.log_evidence * 4, rep.log_evidence, places=10)

        # Replicating *diverse* single-category groups pushes alpha downward
        # (data favor theta near vertices).
        diverse = core.fit([(4, 0)] * 4 + [(0, 4)] * 4)
        self.assertEqual(diverse.status, core.OptStatus.LOWER_ENDPOINT.value)
        self.assertEqual(diverse.alpha, core.ALPHA_MIN)

    def test_search_does_not_leave_interval(self):
        result = core.fit([(2, 0), (0, 2), (1, 1)])
        self.assertGreaterEqual(result.alpha, core.ALPHA_MIN)
        self.assertLessEqual(result.alpha, core.ALPHA_MAX)
        for cand in result.candidates:
            self.assertGreaterEqual(cand.bracket[0], math.log(core.ALPHA_MIN) - 1e-15)
            self.assertLessEqual(cand.bracket[1], math.log(core.ALPHA_MAX) + 1e-15)

    def test_fitted_posteriors_use_fitted_alpha(self):
        rows = [(0, 0), (2, 0), (0, 0), (0, 2), (1, 1)]
        result = core.fit(rows)
        self.assertEqual(result.nonempty_group_count, 3)
        self.assertEqual(result.group_count, 5)
        # Empty groups report the prior and still get a predictive distribution.
        empty = result.groups[0]
        self.assertTrue(empty.is_empty)
        self.assertEqual(empty.mean, (0.5, 0.5))
        self.assertEqual(empty.predictive, (0.5, 0.5))
        self.assertEqual(empty.predicted_category, 0)
        # Non-empty group is shrunk toward the prior mean.
        post = result.groups[1]
        self.assertFalse(post.is_empty)
        self.assertGreater(post.mean[0], 2.0 / (2.0 + result.alpha))
        self.assertAlmostEqual(post.mean[0], (2 + result.alpha) / (2 + 2 * result.alpha), places=13)

    def test_roundtrip_alpha_at_fitted_value_is_consistent(self):
        # Supplementary only: the public evidence at the reported alpha agrees
        # with the fit's stored value.
        rows = [(2, 0), (0, 2), (1, 1)]
        result = core.fit(rows)
        self.assertAlmostEqual(
            core.log_evidence(result.alpha, rows, 2),
            result.log_evidence,
            places=11,
        )


if __name__ == "__main__":
    unittest.main()
