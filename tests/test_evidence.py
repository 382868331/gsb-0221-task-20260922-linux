"""Evidence and posterior checks against the independent K=2 references."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import (  # noqa: E402
    beta_moments_via_integrals,
    beta_posterior_moments,
    evidence_expansion,
    evidence_quadrature,
)

from himeb import log_evidence, log_evidence_one, posterior_group  # noqa: E402


class EvidenceReferenceTests(unittest.TestCase):
    """K=2 small counts: production evidence vs integral-polynomial expansion.

    The reference never calls the package; tolerance is tied to the measured
    reference error (~2e-8 for n <= 8 over the full alpha interval), not used
    to mask boundary mistakes.
    """

    ALPHAS = (1e-4, 1e-2, 0.3, 1.0, 2.0, 7.0, 100.0, 1e4)

    def test_single_rows_vs_expansion(self):
        for n in range(0, 9):
            for x in range(0, n + 1):
                for a in self.ALPHAS:
                    with self.subTest(x=x, n=n, alpha=a):
                        got = log_evidence_one(a, (x, n - x))
                        ref = evidence_expansion(x, n, a)
                        self.assertLessEqual(
                            abs(got - ref),
                            2e-8,
                            msg=f"({x},{n-x}) alpha={a}: {got} vs {ref}",
                        )

    def test_batches_vs_expansion(self):
        rows = [(0, 2), (3, 1), (2, 2), (5, 0), (1, 4), (0, 0)]
        for a in (1e-3, 0.5, 3.0, 1e4):
            with self.subTest(alpha=a):
                got = log_evidence(a, rows)
                ref = math.fsum(
                    evidence_expansion(c0, c0 + c1, a) for c0, c1 in rows
                )
                self.assertLessEqual(abs(got - ref), 5e-8)

    def test_vs_direct_quadrature(self):
        # A second, unrelated route: direct adaptive-Simpson integration.
        for x, n in [(1, 2), (2, 2), (1, 3), (2, 4), (3, 5), (0, 4), (4, 4)]:
            for a in (1.0, 2.5, 10.0):
                with self.subTest(x=x, n=n, alpha=a):
                    got = log_evidence_one(a, (x, n - x))
                    ref = evidence_quadrature(x, n, a)
                    self.assertLessEqual(abs(got - ref), 2e-9)

    def test_empty_group_is_exactly_zero(self):
        self.assertEqual(log_evidence_one(0.37, (0, 0)), 0.0)
        self.assertEqual(log_evidence(1e4, [(0, 0), (0, 0)]), 0.0)

    def test_empty_group_does_not_change_evidence(self):
        nonempty = [(2, 1), (0, 3)]
        base = log_evidence(1.7, nonempty)
        self.assertEqual(
            log_evidence(1.7, nonempty + [(0, 0)]), base
        )
        self.assertEqual(
            log_evidence(1.7, [(0, 0)] + nonempty + [(0, 0)]), base
        )

    def test_single_draw_evidence_is_constant(self):
        # Each n=1 row contributes -log(K) = -log(2) for every alpha.
        rows = [(1, 0), (0, 1), (1, 0), (0, 0)]
        expected = -3.0 * math.log(2.0)
        for a in (1e-4, 0.5, 1.0, 1e4):
            self.assertAlmostEqual(log_evidence(a, rows), expected, places=14)


class PosteriorReferenceTests(unittest.TestCase):
    def test_moments_vs_beta_integrals(self):
        # Count pairs (c0, c1): total n = c0 + c1.
        for c0, c1 in [(0, 0), (2, 1), (3, 3), (0, 5), (5, 0), (4, 2), (1, 2)]:
            n = c0 + c1
            for a in (0.2, 1.0, 5.0, 1e3):
                with self.subTest(c0=c0, c1=c1, alpha=a):
                    post = posterior_group((c0, c1), a)
                    # Independent closed-form Beta moments for both classes.
                    m0, s0, v0 = beta_posterior_moments(c0, n, a)
                    m1, s1, v1 = beta_posterior_moments(c1, n, a)
                    self.assertAlmostEqual(post.mean[0], m0, places=12)
                    self.assertAlmostEqual(post.mean[1], m1, places=12)
                    self.assertAlmostEqual(post.variance[0], v0, places=12)
                    self.assertAlmostEqual(post.variance[1], v1, places=12)
                    self.assertAlmostEqual(m0 + m1, 1.0, places=12)
                    self.assertAlmostEqual(
                        s0, post.mean[0] ** 2 + v0, places=12
                    )
                    # Second independent route via lgamma beta-integral ratios
                    # (only accurate when beta arguments are not large).
                    if a <= 10.0:
                        i0, si0, vi0 = beta_moments_via_integrals(c0, n, a)
                        i1, si1, vi1 = beta_moments_via_integrals(c1, n, a)
                        self.assertAlmostEqual(post.mean[0], i0, places=12)
                        self.assertAlmostEqual(post.variance[0], vi0, places=12)
                        self.assertAlmostEqual(post.mean[1], i1, places=12)
                        self.assertAlmostEqual(post.variance[1], vi1, places=12)
                        self.assertAlmostEqual(s0, si0, places=12)
                        self.assertAlmostEqual(s1, si1, places=12)

    def test_predictive_equals_mean_and_normalizes(self):
        post = posterior_group((3, 1, 0, 2), 0.8)
        self.assertEqual(post.predictive, post.mean)
        self.assertAlmostEqual(sum(post.predictive), 1.0, places=15)
        self.assertEqual(post.predicted, 0)

    def test_empty_group_returns_prior(self):
        post = posterior_group((0, 0, 0, 0), 2.0)
        self.assertTrue(post.is_empty)
        self.assertTrue(all(abs(m - 0.25) < 1e-15 for m in post.mean))
        a, k = 2.0, 4
        prior_var = (a * (k * a - a)) / ((k * a) ** 2 * (k * a + 1.0))
        for v in post.variance:
            self.assertAlmostEqual(v, prior_var, places=15)

    def test_simplex_and_ordering(self):
        post = posterior_group((0, 4), 0.1)
        self.assertAlmostEqual(sum(post.mean), 1.0, places=15)
        self.assertEqual(post.predicted, 1)
        # Heavy shrinkage for a tiny alpha: the empty class still keeps a
        # strictly positive predictive mass.
        self.assertGreater(post.predictive[0], 0.0)


if __name__ == "__main__":
    unittest.main()
