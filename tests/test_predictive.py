"""Posterior-predictive event tests (Dirichlet-multinomial).

Cross-checks the log-domain dynamic program in himeb.predictive against the
naive full enumeration in tests/reference.py (independent finite-product
formula), plus full-range normalization, single count vectors, the m=0
branch, empty/impossible events, tiny/endpoint alpha and input validation.
"""

from __future__ import annotations

import math
import random
import unittest

from himeb import (
    MAX_K,
    MAX_PREDICTIVE_M,
    fit_alpha,
    posterior_predictive,
)
from reference import dm_event_enumeration


class PosteriorPredictiveTests(unittest.TestCase):
    def test_full_range_is_one(self):
        for counts, alpha, m in [
            ((0, 0, 2), 1.0, 5),
            ((3, 0, 1), 0.1, 8),
            ((2, 2, 2, 2), 1e-4, 7),
            ((5, 0), 3.3, 4),
            ((1, 0, 2, 0, 1, 0), 100.0, 50),
        ]:
            with self.subTest(counts=counts, alpha=alpha, m=m):
                k = len(counts)
                res = posterior_predictive(counts, alpha, m, [(0, m)] * k)
                self.assertAlmostEqual(res.probability, 1.0, places=12)
                self.assertAlmostEqual(res.log_probability, 0.0, places=12)
                self.assertFalse(res.is_empty_event)
                params = [alpha + c for c in counts]
                total = sum(params)
                for j in range(k):
                    self.assertAlmostEqual(
                        res.conditional_mean[j],
                        m * params[j] / total,
                        places=9,
                    )
                self.assertAlmostEqual(sum(res.conditional_mean), m, places=8)

    def test_m_zero_branch(self):
        res = posterior_predictive((2, 3), 0.5, 0, [(0, 0), (0, 0)])
        self.assertEqual(res.probability, 1.0)
        self.assertEqual(res.log_probability, 0.0)
        self.assertEqual(res.conditional_mean, (0.0, 0.0))
        self.assertFalse(res.is_empty_event)
        # With m=0 every contract-valid pair is (0, 0), so the only event
        # is the all-zero count vector; asking for a positive bound is a
        # validation error rather than an empty event.
        with self.assertRaises(ValueError):
            posterior_predictive((2, 3), 0.5, 0, [(1, 1), (0, 0)])

    def test_infeasible_bounds_return_zero(self):
        # sum(lower) > m and sum(upper) < m both give an empty event.
        res = posterior_predictive((1, 1), 1.0, 5, [(4, 5), (4, 5)])
        self.assertTrue(res.is_empty_event)
        self.assertEqual((res.probability, res.log_probability), (0.0, -math.inf))
        self.assertTrue(all(math.isnan(v) for v in res.conditional_mean))
        res = posterior_predictive((1, 1), 1.0, 5, [(0, 1), (0, 1)])
        self.assertTrue(res.is_empty_event)

    def test_single_count_vector_matches_dm_pmf(self):
        res = posterior_predictive(
            (2, 1, 0), 0.3, 5, [(3, 3), (1, 1), (1, 1)]
        )
        ref_p, ref_mean = dm_event_enumeration(
            [2.3, 1.3, 0.3], 5, [(3, 3), (1, 1), (1, 1)]
        )
        self.assertAlmostEqual(res.probability, ref_p, places=12)
        for got, want in zip(res.conditional_mean, ref_mean):
            self.assertAlmostEqual(got, want, places=10)

    def test_random_events_against_enumeration(self):
        rng = random.Random(20260925)
        for _ in range(300):
            k = rng.randint(2, MAX_K)
            counts = tuple(rng.randint(0, 5) for _ in range(k))
            alpha = 10 ** rng.uniform(-4, 4)
            m = rng.randint(0, 8)
            bounds = []
            for _ in range(k):
                lo, hi = sorted((rng.randint(0, m), rng.randint(0, m)))
                bounds.append((lo, hi))
            res = posterior_predictive(counts, alpha, m, bounds)
            ref_p, ref_mean = dm_event_enumeration(
                [alpha + c for c in counts], m, bounds
            )
            if ref_p == 0.0:
                self.assertTrue(res.is_empty_event)
                self.assertEqual(res.probability, 0.0)
                self.assertTrue(
                    all(math.isnan(v) for v in res.conditional_mean)
                )
                continue
            self.assertFalse(res.is_empty_event)
            self.assertLessEqual(res.probability, 1.0 + 1e-12)
            self.assertAlmostEqual(
                res.probability, ref_p, delta=1e-9 * max(1.0, ref_p)
            )
            self.assertAlmostEqual(
                math.exp(res.log_probability), ref_p, delta=1e-9
            )
            for got, want in zip(res.conditional_mean, ref_mean):
                self.assertAlmostEqual(got, want, delta=1e-8)
            self.assertAlmostEqual(
                sum(res.conditional_mean), m, delta=1e-6
            )

    def test_tiny_alpha_and_endpoint_alpha(self):
        # Empty observed group + minimal alpha: posterior is nearly singular;
        # compare against enumeration rather than any multinomial plug-in.
        alpha = 1e-4
        bounds = [(1, 2), (0, 2), (0, 2)]
        res = posterior_predictive((0, 0, 0), alpha, 3, bounds)
        ref_p, ref_mean = dm_event_enumeration([alpha] * 3, 3, bounds)
        self.assertAlmostEqual(res.probability, ref_p, places=12)
        for got, want in zip(res.conditional_mean, ref_mean):
            self.assertAlmostEqual(got, want, places=10)
        # Upper endpoint alpha=1e4 keeps full accuracy (log-sum products).
        res = posterior_predictive(
            (4, 0, 1), 1e4, 6, [(1, 4), (0, 3), (0, 3)]
        )
        ref_p, ref_mean = dm_event_enumeration(
            [1e4 + 4, 1e4, 1e4 + 1], 6, [(1, 4), (0, 3), (0, 3)]
        )
        self.assertAlmostEqual(res.probability, ref_p, places=10)
        for got, want in zip(res.conditional_mean, ref_mean):
            self.assertAlmostEqual(got, want, places=8)

    def test_tiny_event_probability_stays_in_log_domain(self):
        # A rare but feasible event whose plain probability is ~1e-17 must
        # not underflow: log_probability is finite and reconstructible.
        res = posterior_predictive(
            (10, 0, 0), 1e-4, 100, [(0, 2), (98, 100), (0, 2)]
        )
        self.assertFalse(res.is_empty_event)
        self.assertTrue(math.isfinite(res.log_probability))
        self.assertLess(res.log_probability, -30.0)
        self.assertAlmostEqual(
            math.exp(res.log_probability), res.probability, places=15
        )

    def test_conditional_on_fitted_alpha_no_refit(self):
        groups = [(4, 0, 1), (0, 3, 0), (2, 2, 1), (5, 0, 0)]
        fit = fit_alpha(groups)
        res = posterior_predictive(
            groups[0], fit.alpha, 10, [(0, 5), (1, 6), (0, 5)]
        )
        self.assertEqual(res.alpha, fit.alpha)
        self.assertEqual(
            res.posterior_params,
            tuple(fit.alpha + c for c in groups[0]),
        )
        ref_p, ref_mean = dm_event_enumeration(
            list(res.posterior_params), 10, [(0, 5), (1, 6), (0, 5)]
        )
        self.assertAlmostEqual(res.probability, ref_p, places=10)
        for got, want in zip(res.conditional_mean, ref_mean):
            self.assertAlmostEqual(got, want, places=8)

    def test_validation(self):
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), float("nan"), 3, [(0, 3), (0, 3)])
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), 0.0, 3, [(0, 3), (0, 3)])
        with self.assertRaises(TypeError):
            posterior_predictive((1, 1), 1.0, True, [(0, 1), (0, 1)])
        with self.assertRaises(ValueError):
            posterior_predictive(
                (1, 1), 1.0, MAX_PREDICTIVE_M + 1, [(0, 101), (0, 101)]
            )
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), 1.0, -1, [(0, 0), (0, 0)])
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), 1.0, 3, [(0, 3)])
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), 1.0, 3, [(2, 1), (0, 3)])
        with self.assertRaises(ValueError):
            posterior_predictive((1, 1), 1.0, 3, [(0, 4), (0, 3)])
        with self.assertRaises(TypeError):
            posterior_predictive((1, 1), 1.0, 3, [(0, True), (0, 3)])
        with self.assertRaises((TypeError, ValueError)):
            posterior_predictive((1, 1), 1.0, 3, [(0, 3)])


if __name__ == "__main__":
    unittest.main()
