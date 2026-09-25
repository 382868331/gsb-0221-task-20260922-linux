"""Posterior-predictive box events: DP vs full K-vector enumeration.

For small m every K-dimensional count vector is enumerated with the
independent finite-product Dirichlet-multinomial formula. The reference
never imports himeb internals for its expected values.
"""

import itertools
import math
import unittest

from himeb import MAX_PREDICT_M, predictive_event


def enumerate_reference(counts, alpha, m, bounds):
    """Brute-force (event probability, conditional means).

    Each Dirichlet-multinomial factor is the independent finite product
    Gamma(a+y)/(Gamma(a) y!) = prod_{j=0}^{y-1} (a+j)/(j+1), so no lgamma
    cancellation occurs at large alpha.
    """
    ak = [alpha + c for c in counts]
    k = len(ak)
    a0 = sum(ak)

    def rising_factorial_ratio(x, y):
        prod = 1.0
        for j in range(y):
            prod *= (x + j) / (j + 1)
        return prod

    norm_denom = rising_factorial_ratio(a0, m)
    event_mass = 0.0
    total_mass = 0.0
    y_sums = [0.0] * k
    for combo in itertools.product(range(m + 1), repeat=k):
        if sum(combo) != m:
            continue
        p = 1.0
        for j in range(k):
            p *= rising_factorial_ratio(ak[j], combo[j])
        p /= norm_denom
        total_mass += p
        if all(bounds[j][0] <= combo[j] <= bounds[j][1] for j in range(k)):
            event_mass += p
            for j in range(k):
                y_sums[j] += combo[j] * p
    assert abs(total_mass - 1.0) < 1e-9, total_mass
    means = tuple(s / event_mass for s in y_sums) if event_mass > 0 else None
    return event_mass, means


class PredictiveReferenceTests(unittest.TestCase):
    def test_small_m_boxes_vs_enumeration(self):
        cases = [
            ((2, 1, 3), 0.5, 5, ((0, 2), (0, 5), (1, 4))),
            ((0, 0), 1e-4, 4, ((1, 3), (0, 4))),
            ((3, 0, 0, 2), 7.0, 8, ((0, 8), (0, 1), (0, 8), (2, 6))),
            ((1, 1), 1.0, 3, ((0, 3), (0, 3))),
            ((5, 2), 1e-4, 6, ((0, 0), (6, 6))),
            ((0, 0, 0), 1e-2, 7, ((0, 7), (0, 0), (0, 7))),
            ((2, 1, 3), 1e4, 9, ((1, 4), (0, 3), (2, 9))),
            ((1, 0, 2, 0, 1), 2.0, 6,
             ((0, 2), (0, 6), (1, 3), (0, 6), (0, 2))),
        ]
        for counts, alpha, m, bounds in cases:
            with self.subTest(counts=counts, alpha=alpha, m=m):
                got = predictive_event(counts, alpha, m, bounds)
                ref_p, ref_mean = enumerate_reference(counts, alpha, m, bounds)
                self.assertLessEqual(abs(got.probability - ref_p), 1e-12)
                self.assertAlmostEqual(
                    got.probability, math.exp(got.log_probability), delta=1e-15
                )
                self.assertIsNotNone(got.conditional_mean)
                for a, b in zip(got.conditional_mean, ref_mean):
                    self.assertLessEqual(abs(a - b), 1e-10)
                self.assertAlmostEqual(sum(got.conditional_mean), m, delta=1e-9)

    def test_single_count_vectors(self):
        cases = [
            ((2, 1, 3), 0.5, 5, (2, 1, 2)),
            ((4, 0), 0.3, 6, (0, 6)),
            ((1, 1, 1), 2.0, 4, (2, 2, 0)),
        ]
        for counts, alpha, m, point in cases:
            bounds = tuple((y, y) for y in point)
            with self.subTest(point=point):
                got = predictive_event(counts, alpha, m, bounds)
                ref_p, ref_mean = enumerate_reference(counts, alpha, m, bounds)
                self.assertLessEqual(abs(got.probability - ref_p), 1e-14)
                self.assertEqual(got.conditional_mean, tuple(point))
                self.assertEqual(ref_mean, tuple(point))

    def test_full_range_is_one_and_marginal_means(self):
        for counts, alpha, m in [
            ((2, 1, 3), 0.5, 20),
            ((0, 0, 0, 0), 1e-4, 10),
            ((3, 1), 7.0, 30),
        ]:
            with self.subTest(counts=counts, m=m):
                bounds = [(0, m)] * len(counts)
                got = predictive_event(counts, alpha, m, bounds)
                self.assertAlmostEqual(got.probability, 1.0, delta=1e-12)
                self.assertLessEqual(abs(got.log_probability), 1e-12)
                ak = [alpha + c for c in counts]
                total = sum(ak)
                for mean, a in zip(got.conditional_mean, ak):
                    self.assertAlmostEqual(mean, m * a / total, delta=1e-9)

    def test_m_zero(self):
        ok = predictive_event((1, 2), 0.5, 0, [(0, 0), (0, 0)])
        self.assertEqual(ok.probability, 1.0)
        self.assertEqual(ok.log_probability, 0.0)
        self.assertEqual(ok.conditional_mean, (0.0, 0.0))
        bad = predictive_event((1, 2), 0.5, 0, [(1, 1), (0, 0)])
        self.assertEqual(bad.probability, 0.0)
        self.assertEqual(bad.log_probability, float("-inf"))
        self.assertIsNone(bad.conditional_mean)

    def test_impossible_bounds_zero(self):
        r = predictive_event((1, 1), 2.0, 5, [(4, 5), (4, 5)])
        self.assertEqual(r.probability, 0.0)
        self.assertEqual(r.log_probability, float("-inf"))
        self.assertIsNone(r.conditional_mean)
        r = predictive_event((1, 1, 1), 2.0, 5, [(0, 1), (0, 1), (0, 1)])
        self.assertEqual(r.probability, 0.0)
        self.assertIsNone(r.conditional_mean)
        # Individually valid but jointly infeasible bounds.
        r = predictive_event((1, 1, 1), 1.0, 5, [(5, 5), (5, 5), (0, 5)])
        self.assertEqual(r.probability, 0.0)
        self.assertIsNone(r.conditional_mean)

    def test_tiny_alpha_rare_event_keeps_log_probability(self):
        r = predictive_event((5, 0), 1e-4, 40, [(0, 0), (40, 40)])
        ref_p, _ = enumerate_reference((5, 0), 1e-4, 40, [(0, 0), (40, 40)])
        self.assertGreater(ref_p, 0.0)
        self.assertLessEqual(
            abs(math.exp(r.log_probability) - ref_p) / ref_p, 1e-11
        )
        self.assertAlmostEqual(r.conditional_mean[0], 0.0, delta=1e-12)
        self.assertAlmostEqual(r.conditional_mean[1], 40.0, delta=1e-12)

    def test_uses_fitted_alpha(self):
        from himeb import fit_alpha
        fit = fit_alpha([(4, 1), (2, 3), (5, 0)])
        r = predictive_event((4, 1), fit.alpha, 12, [(4, 12), (0, 8)])
        ref_p, ref_mean = enumerate_reference(
            (4, 1), fit.alpha, 12, [(4, 12), (0, 8)]
        )
        self.assertLessEqual(abs(r.probability - ref_p), 1e-12)
        for a, b in zip(r.conditional_mean, ref_mean):
            self.assertLessEqual(abs(a - b), 1e-10)
        self.assertEqual(r.alpha, fit.alpha)

    def test_limits_and_validation(self):
        self.assertEqual(MAX_PREDICT_M, 100)
        r = predictive_event((1, 0, 0, 0, 0, 1), 3.0, 100, [(0, 100)] * 6)
        self.assertAlmostEqual(r.probability, 1.0, delta=1e-10)
        with self.assertRaises(ValueError):
            predictive_event((1, 1), 1.0, 101, [(0, 101), (0, 101)])
        with self.assertRaises(ValueError):
            predictive_event((1, 1), 1.0, -1, [(0, 1), (0, 1)])
        with self.assertRaises(TypeError):
            predictive_event((1, 1), 1.0, True, [(0, 1), (0, 1)])
        with self.assertRaises(ValueError):
            predictive_event((1, 1), 1.0, 5, [(0, 5), (4, 3)])
        with self.assertRaises(ValueError):
            predictive_event((1, 1), 1.0, 5, [(0, 5)])
        with self.assertRaises(TypeError):
            predictive_event((1, 1), 1.0, 5, [(0, 5), 5])
        with self.assertRaises(ValueError):
            predictive_event((1, 1), float("nan"), 5, [(0, 5), (0, 5)])


if __name__ == "__main__":
    unittest.main()
