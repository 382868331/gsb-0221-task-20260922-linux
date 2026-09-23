"""Optimizer, endpoint/interior/flat statuses, replication, multimodality."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import dense_alpha_scan, evidence_expansion  # noqa: E402

from himeb import (  # noqa: E402
    ALPHA_MAX,
    ALPHA_MIN,
    analyze,
    fit_alpha,
    log_evidence,
    maximize_on_interval,
)


def _gaussian_mode_objective():
    # Two separated interior modes in t = log(alpha), of distinct heights.
    t1, t2 = math.log(0.02), math.log(60.0)

    def f(a: float) -> float:
        t = math.log(a)
        return math.exp(-(((t - t1) / 0.35) ** 2)) + 0.6 * math.exp(
            -(((t - t2) / 0.35) ** 2)
        )

    return f, t1, t2


class OptimizerTests(unittest.TestCase):
    def test_endpoints_always_candidates(self):
        f, _, _ = _gaussian_mode_objective()
        r = maximize_on_interval(f)
        kinds = {c.kind for c in r.candidates}
        self.assertIn("lower_endpoint", kinds)
        self.assertIn("upper_endpoint", kinds)

    def test_finds_all_interior_modes(self):
        f, t1, t2 = _gaussian_mode_objective()
        r = maximize_on_interval(f)
        interiors = [c for c in r.candidates if c.kind == "interior"]
        self.assertGreaterEqual(len(interiors), 2)
        ts = sorted(math.log(c.alpha) for c in interiors)
        self.assertAlmostEqual(ts[0], t1, delta=0.05)
        self.assertAlmostEqual(ts[-1], t2, delta=0.05)
        # Higher mode wins.
        self.assertAlmostEqual(r.alpha, math.exp(t1), delta=1e-4)
        self.assertEqual(r.status, "interior")
        # Each interior candidate retains its log-alpha search bracket.
        for c in interiors:
            self.assertIsNotNone(c.bracket)
            self.assertLess(c.bracket[0], math.log(c.alpha))
            self.assertLess(math.log(c.alpha), c.bracket[1])

    def test_budget_exhaustion_is_flagged_distinct_from_no_solution(self):
        f, _, _ = _gaussian_mode_objective()
        r = maximize_on_interval(f, max_iter=1)
        # A continuous function on the compact interval DOES have a maximum;
        # the only defect is that a refinement ran out of iterations.
        self.assertFalse(r.converged)
        self.assertTrue(r.budget_exhausted)
        self.assertEqual(r.status, "interior")
        self.assertTrue(any(not c.converged for c in r.candidates))
        # Result remains a usable approximate maximizer near the true mode.
        self.assertAlmostEqual(r.alpha, 0.02, delta=0.02)

    def test_proved_no_unique_solution_is_not_budget_failure(self):
        r = fit_alpha([(0, 0), (0, 0)])
        self.assertEqual(r.status, "flat")
        self.assertTrue(r.converged)
        self.assertFalse(r.budget_exhausted)
        self.assertEqual(r.max_iterations_used, 0)

        r2 = fit_alpha([(1, 0), (0, 1), (0, 0)])
        self.assertEqual(r2.status, "flat")
        self.assertTrue(r2.converged)
        self.assertFalse(r2.budget_exhausted)
        self.assertEqual(r2.n_nonempty_groups, 2)
        self.assertAlmostEqual(r2.log_evidence, -2.0 * math.log(2.0), places=14)


class FitStatusTests(unittest.TestCase):
    def test_lower_endpoint_single_class_extreme(self):
        rows = [(8, 0)] * 4
        r = fit_alpha(rows)
        self.assertEqual(r.status, "lower_endpoint")
        self.assertEqual(r.alpha, ALPHA_MIN)
        # Monotone decreasing: evidence strictly falls as alpha grows.
        self.assertLess(
            log_evidence(1e-3, rows), log_evidence(ALPHA_MIN, rows)
        )
        self.assertLess(log_evidence(1.0, rows), log_evidence(1e-3, rows))

    def test_upper_endpoint_balanced_extreme(self):
        rows = [(200, 200)] * 4
        r = fit_alpha(rows)
        self.assertEqual(r.status, "upper_endpoint")
        self.assertEqual(r.alpha, ALPHA_MAX)
        self.assertGreater(
            log_evidence(ALPHA_MAX, rows), log_evidence(5000.0, rows)
        )

    def test_interior_fit_vs_independent_dense_scan(self):
        rows = [
            (4, 1), (5, 0), (0, 5), (2, 2), (3, 2),
            (1, 4), (0, 3), (2, 1),
        ]
        r = fit_alpha(rows)
        self.assertEqual(r.status, "interior")
        ref_alpha, ref_v = dense_alpha_scan(rows, n_points=4001)
        self.assertAlmostEqual(r.alpha, ref_alpha, delta=0.02 * ref_alpha)
        self.assertAlmostEqual(
            r.log_evidence,
            math.fsum(
                evidence_expansion(c0, c0 + c1, r.alpha) for c0, c1 in rows
            ),
            places=10,
        )

    def test_replication_invariance(self):
        base = [(4, 1), (5, 0), (0, 5), (2, 2), (3, 2), (1, 4)]
        r1 = fit_alpha(base)
        r3 = fit_alpha(base * 3)
        self.assertEqual(r1.status, r3.status)
        self.assertAlmostEqual(r1.alpha, r3.alpha, delta=1e-6 * r1.alpha)
        # Tripling non-empty groups triples the log evidence.
        self.assertAlmostEqual(r3.log_evidence, 3.0 * r1.log_evidence, places=8)

    def test_empty_groups_excluded_from_fit(self):
        base = [(4, 1), (2, 3), (0, 2)]
        r1 = fit_alpha(base)
        r2 = fit_alpha([(0, 0)] + base + [(0, 0), (0, 0)])
        self.assertEqual(r1.alpha, r2.alpha)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r2.n_nonempty_groups, 3)

    def test_grid_restriction_is_visible(self):
        # A mode much narrower than one grid cell can be missed: with only 3
        # grid points the narrow mode below is invisible, whereas the default
        # grid finds it. This documents (rather than hides) the search limit.
        t_mode = math.log(3.0)

        def f(a: float) -> float:
            t = math.log(a)
            return math.exp(-(((t - t_mode) / 0.02) ** 2))

        r_coarse = maximize_on_interval(f, grid_points=3)
        self.assertNotAlmostEqual(r_coarse.alpha, 3.0, delta=0.1)
        r_fine = maximize_on_interval(f, grid_points=2001)
        self.assertAlmostEqual(r_fine.alpha, 3.0, delta=1e-4)


class AnalyzeTests(unittest.TestCase):
    def test_analyze_alignment_and_counts(self):
        rows = [(3, 1), (0, 0), (0, 2), (1, 1)]
        result = analyze(rows)
        self.assertEqual(result.k, 2)
        self.assertEqual(result.n_groups, 4)
        self.assertEqual(result.n_nonempty_groups, 3)
        self.assertEqual(len(result.posteriors), 4)
        for post, row in zip(result.posteriors, rows):
            self.assertEqual(post.counts, tuple(row))
            self.assertEqual(post.alpha, result.fit.alpha)
        self.assertTrue(result.posteriors[1].is_empty)

    def test_contract_limits_accepted(self):
        rows = [(1, 1, 0, 2, 0, 1)] * 40
        result = analyze(rows)
        self.assertEqual(result.k, 6)
        self.assertEqual(result.n_groups, 40)
        self.assertTrue(all(abs(sum(p.mean) - 1.0) < 1e-12 for p in result.posteriors))

    def test_shrinkage_direction(self):
        # Sparse count is pulled toward the uniform 1/K prior mean.
        result = analyze([(4, 0)] * 6)
        a = result.fit.alpha
        post = result.posteriors[0]
        mle0 = 1.0
        self.assertLess(post.mean[0], mle0)
        self.assertGreater(post.mean[0], 1.0 / 2.0)
        self.assertAlmostEqual(a, post.alpha)


if __name__ == "__main__":
    unittest.main()
