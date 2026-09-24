"""Per-class (asymmetric) Dirichlet concentrations: evidence, fit, posterior.

Expected values come from hand-derived rising-factorial formulas, the
independent beta-integral/quadrature references in ``reference.py``, and
finite differences of those references -- never from the production module.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import (  # noqa: E402
    evidence_beta_binomial,
    evidence_quadrature_asym,
)

from himeb import (  # noqa: E402
    ALPHA_MAX,
    ALPHA_MIN,
    analyze_concentrations,
    fit_alpha,
    fit_concentrations,
    log_evidence,
    log_evidence_one,
    posterior_group,
)


class AsymmetricEvidenceHandTests(unittest.TestCase):
    """K=3, alphas = (0.5, 2.0, 1.5), A = 4.0, hand-derived probabilities.

    Dirichlet-multinomial: p(x) = C(n; x) * prod_k ak^[xk] / A^[n], where
    a^[m] = a(a+1)...(a+m-1) is the rising factorial.
    """

    ALPHAS = (0.5, 2.0, 1.5)

    def test_single_draws(self):
        # p(e_j) = alpha_j / A.
        expected = (0.5 / 4.0, 2.0 / 4.0, 1.5 / 4.0)
        for j in range(3):
            row = tuple(1 if i == j else 0 for i in range(3))
            with self.subTest(row=row):
                self.assertAlmostEqual(
                    log_evidence_one(self.ALPHAS, row),
                    math.log(expected[j]),
                    places=14,
                )

    def test_two_draws_same_class(self):
        # p(2 e_1) = (2.0 * 3.0) / (4.0 * 5.0) = 0.3
        self.assertAlmostEqual(
            log_evidence_one(self.ALPHAS, (0, 2, 0)),
            math.log(0.3),
            places=14,
        )
        # p(2 e_0) = (0.5 * 1.5) / (4.0 * 5.0) = 0.0375
        self.assertAlmostEqual(
            log_evidence_one(self.ALPHAS, (2, 0, 0)),
            math.log(0.0375),
            places=14,
        )

    def test_two_draws_mixed(self):
        # p(e_0 + e_2) = 2 * 0.5 * 1.5 / (4.0 * 5.0) = 0.075
        self.assertAlmostEqual(
            log_evidence_one(self.ALPHAS, (1, 0, 1)),
            math.log(0.075),
            places=14,
        )

    def test_three_draws_mixed(self):
        # p(e_0+e_1+e_2) = 6 * 0.5 * 2.0 * 1.5 / (4.0 * 5.0 * 6.0) = 0.075
        self.assertAlmostEqual(
            log_evidence_one(self.ALPHAS, (1, 1, 1)),
            math.log(0.075),
            places=14,
        )

    def test_empty_row_is_exactly_zero(self):
        self.assertEqual(log_evidence_one(self.ALPHAS, (0, 0, 0)), 0.0)

    def test_batch_sums_rows(self):
        rows = [(1, 0, 0), (0, 2, 0), (0, 0, 0), (1, 1, 1)]
        expected = (
            math.log(0.5 / 4.0) + math.log(0.3) + 0.0 + math.log(0.075)
        )
        self.assertAlmostEqual(
            log_evidence(self.ALPHAS, rows), expected, places=14
        )


class AsymmetricEvidenceReferenceTests(unittest.TestCase):
    """K=2 per-class evidence vs independent beta- to direct quadrature."""

    ROWS = [(0, 2), (3, 1), (2, 2), (5, 0), (1, 4), (0, 0), (4, 4)]

    def test_vs_beta_binomial_closed_form(self):
        # The lgamma-difference reference itself cancels at very large
        # concentrations, so it is kept to moderate values here (the same
        # reference limitation the symmetric suite documents).
        for a0, a1 in [(0.3, 2.5), (1.0, 1.0), (7.0, 0.1), (2.0, 2.0), (20.0, 3.0)]:
            for x, n in [(0, 2), (1, 2), (3, 4), (0, 5), (5, 5), (2, 3)]:
                with self.subTest(x=x, n=n, a0=a0, a1=a1):
                    got = log_evidence_one((a0, a1), (x, n - x))
                    ref = evidence_beta_binomial(x, n, a0, a1)
                    self.assertAlmostEqual(got, ref, places=12)

    def test_batch_vs_beta_binomial(self):
        for alphas in [(0.5, 3.0), (2.0, 0.25), (1.0, 1.0)]:
            with self.subTest(alphas=alphas):
                got = log_evidence(alphas, self.ROWS)
                ref = math.fsum(
                    evidence_beta_binomial(c0, c0 + c1, *alphas)
                    for c0, c1 in self.ROWS
                )
                self.assertAlmostEqual(got, ref, places=12)

    def test_vs_direct_quadrature(self):
        # Integrand vanishes at both endpoints for these parameters.
        for x, n in [(1, 2), (2, 3), (3, 5), (2, 4)]:
            for a0, a1 in [(1.0, 2.5), (3.0, 1.0), (2.0, 6.0)]:
                with self.subTest(x=x, n=n, a0=a0, a1=a1):
                    got = log_evidence_one((a0, a1), (x, n - x))
                    ref = evidence_quadrature_asym(x, n, a0, a1)
                    self.assertAlmostEqual(got, ref, places=9)


class SymmetricDegeneracyTests(unittest.TestCase):
    """A vector of equal concentrations must reproduce the scalar path."""

    ROWS2 = [(4, 1), (0, 3), (2, 2), (0, 0), (5, 0)]
    ROWS4 = [(2, 0, 1, 3), (0, 0, 0, 0), (1, 1, 1, 1), (4, 0, 0, 2)]

    def test_evidence_scalar_equals_uniform_vector_exactly(self):
        for a in (1e-4, 0.37, 1.0, 9.5, 1e4):
            with self.subTest(alpha=a):
                self.assertEqual(
                    log_evidence(a, self.ROWS2),
                    log_evidence((a, a), self.ROWS2),
                )
                self.assertEqual(
                    log_evidence(a, self.ROWS4),
                    log_evidence((a, a, a, a), self.ROWS4),
                )
                self.assertEqual(
                    log_evidence_one(a, (3, 2)),
                    log_evidence_one((a, a), (3, 2)),
                )

    def test_posterior_scalar_equals_uniform_vector_exactly(self):
        for counts in [(3, 1), (0, 0), (0, 5)]:
            for a in (0.2, 1.0, 1e3):
                with self.subTest(counts=counts, alpha=a):
                    p_scalar = posterior_group(counts, a)
                    p_vector = posterior_group(counts, (a,) * len(counts))
                    self.assertEqual(p_scalar.mean, p_vector.mean)
                    self.assertEqual(p_scalar.variance, p_vector.variance)
                    self.assertEqual(p_scalar.predicted, p_vector.predicted)

    def test_symmetric_data_recovers_symmetric_fit(self):
        # Exactly swap-symmetric, strongly overdispersed data.
        rows = [(5, 0), (0, 5), (4, 1), (1, 4), (5, 0), (0, 5)]
        vec_fit = fit_concentrations(rows)
        self.assertTrue(vec_fit.converged)
        a0, a1 = vec_fit.alphas
        # Class-symmetric data: the two fitted concentrations agree.
        self.assertAlmostEqual(a0, a1, delta=1e-6 * a0)
        # And they land on the scalar (symmetric-prior) estimate.
        scalar_fit = fit_alpha(rows)
        self.assertEqual(scalar_fit.status, "interior")
        self.assertAlmostEqual(a0, scalar_fit.alpha, delta=0.01 * scalar_fit.alpha)


class PermutationTests(unittest.TestCase):
    """Relabeling classes must permute alphas and leave evidence unchanged."""

    def test_evidence_invariant_under_joint_permutation(self):
        alphas = (0.5, 2.0, 1.5)
        rows = [(3, 0, 1), (0, 2, 2), (1, 1, 0), (0, 0, 0)]
        # Reverse the class order of both the counts and the concentrations.
        rev_alphas = alphas[::-1]
        rev_rows = [tuple(reversed(r)) for r in rows]
        self.assertAlmostEqual(
            log_evidence(alphas, rows),
            log_evidence(rev_alphas, rev_rows),
            places=13,
        )
        self.assertAlmostEqual(
            log_evidence_one(alphas, (2, 1, 0)),
            log_evidence_one(rev_alphas, (0, 1, 2)),
            places=13,
        )

    def test_fit_permutes_with_class_relabeling(self):
        rows = [(9, 1, 0), (0, 8, 1), (1, 0, 9), (8, 0, 1),
                (0, 9, 0), (1, 1, 8), (7, 0, 0), (0, 0, 9)]
        fit = fit_concentrations(rows)
        rev_fit = fit_concentrations([tuple(reversed(r)) for r in rows])
        self.assertTrue(fit.converged)
        self.assertTrue(rev_fit.converged)
        for a, b in zip(fit.alphas, reversed(rev_fit.alphas)):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(fit.log_evidence, rev_fit.log_evidence, places=12)

    def test_posterior_permutes_with_class_relabeling(self):
        alphas = (0.5, 2.0, 1.5)
        p = posterior_group((3, 0, 1), alphas)
        q = posterior_group((1, 0, 3), alphas[::-1])
        self.assertEqual(p.mean, q.mean[::-1])
        self.assertEqual(p.variance, q.variance[::-1])
        self.assertEqual(p.predicted, 2 - q.predicted)


class ConcentrationFitTests(unittest.TestCase):
    # Lopsided (class 0 dominant) and overdispersed: interior per-class MLE.
    ROWS = [(9, 0, 0), (5, 4, 0), (8, 0, 1), (4, 5, 0),
            (9, 1, 0), (3, 6, 0), (7, 1, 1), (2, 6, 1)]

    def test_converges_and_reflects_class_imbalance(self):
        fit = fit_concentrations(self.ROWS)
        self.assertTrue(fit.converged)
        self.assertEqual(fit.status, "converged")
        self.assertGreaterEqual(fit.iterations, 1)
        self.assertLessEqual(fit.iterations, fit.max_iter)
        self.assertLess(fit.final_step, fit.tolerance)
        # Class 0 dominates the counts, class 2 is rarest: the fitted
        # concentrations order the same way.
        self.assertGreater(fit.alphas[0], fit.alphas[1])
        self.assertGreater(fit.alphas[1], fit.alphas[2])
        self.assertTrue(all(a > 0.0 for a in fit.alphas))
        self.assertEqual(fit.n_nonempty_groups, len(self.ROWS))

    def test_fitted_evidence_beats_start_and_symmetric_fit(self):
        fit = fit_concentrations(self.ROWS)
        self.assertGreaterEqual(
            fit.log_evidence, log_evidence((1.0, 1.0, 1.0), self.ROWS)
        )
        # The per-class model nests the symmetric one, so its maximized
        # evidence cannot be worse (up to float noise).
        sym = fit_alpha(self.ROWS)
        self.assertGreaterEqual(fit.log_evidence, sym.log_evidence - 1e-9)
        # The reported evidence is self-consistent with the public function.
        self.assertEqual(fit.log_evidence, log_evidence(fit.alphas, self.ROWS))

    def test_stationarity_via_independent_finite_differences(self):
        # K=2 so the independent beta-binomial reference can be differenced.
        rows = [(4, 1), (5, 0), (0, 5), (2, 2), (3, 2), (1, 4), (0, 3), (2, 1)]
        fit = fit_concentrations(rows)
        self.assertTrue(fit.converged)
        self.assertEqual(fit.clamped_classes, ())

        def ref_evidence(a0, a1):
            return math.fsum(
                evidence_beta_binomial(c0, c0 + c1, a0, a1) for c0, c1 in rows
            )

        a0, a1 = fit.alphas
        for j, aj in enumerate((a0, a1)):
            h = 1e-5 * aj
            plus = ref_evidence(a0 + h * (j == 0), a1 + h * (j == 1))
            minus = ref_evidence(a0 - h * (j == 0), a1 - h * (j == 1))
            grad = (plus - minus) / (2.0 * h)
            with self.subTest(class_j=j, grad=grad):
                self.assertLess(abs(grad), 1e-4)

    def test_max_iter_exhaustion_is_reported_not_raised(self):
        fit = fit_concentrations(self.ROWS, max_iter=1)
        self.assertFalse(fit.converged)
        self.assertEqual(fit.status, "max_iter_exhausted")
        self.assertEqual(fit.iterations, 1)
        # Still a usable, strictly positive concentration vector.
        self.assertTrue(all(a > 0.0 for a in fit.alphas))
        self.assertTrue(math.isfinite(fit.log_evidence))

    def test_zero_count_class_clamped_to_lower_bound_stays_positive(self):
        # Class 1 is never observed. Its concentration is driven to the
        # lower projection boundary at the first update and stays strictly
        # positive; class 0's likelihood seeks the upper boundary, which
        # the linear iteration can only creep toward, so the finite budget
        # is honestly reported as exhausted rather than converged.
        rows = [(5, 0), (9, 0), (2, 0), (6, 0)]
        fit = fit_concentrations(rows)
        self.assertEqual(fit.status, "max_iter_exhausted")
        self.assertFalse(fit.converged)
        self.assertEqual(fit.alphas[1], ALPHA_MIN)
        self.assertIn(1, fit.clamped_classes)
        self.assertTrue(all(a > 0.0 for a in fit.alphas))
        # Consistent with the public evidence at the returned point.
        self.assertEqual(fit.log_evidence, log_evidence(fit.alphas, rows))

    def test_all_empty_groups_is_proved_flat(self):
        fit = fit_concentrations([(0, 0, 0), (0, 0, 0)])
        self.assertEqual(fit.status, "flat")
        self.assertTrue(fit.converged)
        self.assertEqual(fit.iterations, 0)
        self.assertEqual(fit.alphas, (1.0, 1.0, 1.0))
        self.assertEqual(fit.log_evidence, 0.0)
        self.assertEqual(fit.n_nonempty_groups, 0)

    def test_empty_groups_do_not_change_fit(self):
        base = [(4, 1, 0), (2, 3, 1), (0, 2, 2)]
        fit1 = fit_concentrations(base)
        fit2 = fit_concentrations([(0, 0, 0)] + base + [(0, 0, 0), (0, 0, 0)])
        self.assertEqual(fit1.alphas, fit2.alphas)
        self.assertEqual(fit1.log_evidence, fit2.log_evidence)
        self.assertEqual(fit2.n_nonempty_groups, 3)

    def test_single_draw_groups_converge_to_count_proportions(self):
        # With only n=1 groups the scale is unidentifiable; the iteration
        # preserves the starting sum (K * 1.0 = 2) and the fixed point is
        # alpha_k = (class-k count) * A / G1, i.e. (4/3, 2/3) here.
        rows = [(1, 0), (1, 0), (0, 1)]
        fit = fit_concentrations(rows)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.alphas[0], 4.0 / 3.0, places=9)
        self.assertAlmostEqual(fit.alphas[1], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(sum(fit.alphas), 2.0, places=12)

    def test_start_argument_accepted_scalar_and_vector(self):
        rows = [(4, 1), (0, 3), (2, 2)]
        f_scalar = fit_concentrations(rows, start=0.5)
        f_vector = fit_concentrations(rows, start=(0.5, 0.5))
        self.assertEqual(f_scalar.alphas, f_vector.alphas)
        f_default = fit_concentrations(rows)
        # Different starts converge to the same stationary point here.
        self.assertAlmostEqual(
            f_scalar.alphas[0], f_default.alphas[0], delta=1e-8
        )


class AsymmetricPosteriorTests(unittest.TestCase):
    def test_hand_computed_posterior(self):
        # a_post = (2+1, 6+0) = (3, 6), total = 9.
        post = posterior_group((1, 0), (2.0, 6.0))
        self.assertAlmostEqual(post.mean[0], 1.0 / 3.0, places=15)
        self.assertAlmostEqual(post.mean[1], 2.0 / 3.0, places=15)
        # var_k = a_k (A - a_k) / (A^2 (A+1)) = 3*6 / (81*10) = 1/45.
        self.assertAlmostEqual(post.variance[0], 1.0 / 45.0, places=15)
        self.assertAlmostEqual(post.variance[1], 1.0 / 45.0, places=15)
        self.assertEqual(post.predictive, post.mean)
        self.assertEqual(post.predicted, 1)
        self.assertEqual(post.alpha, (2.0, 6.0))

    def test_empty_group_returns_normalized_prior(self):
        post = posterior_group((0, 0), (2.0, 6.0))
        self.assertTrue(post.is_empty)
        self.assertAlmostEqual(post.mean[0], 0.25, places=15)
        self.assertAlmostEqual(post.mean[1], 0.75, places=15)
        # Prior variance: 2*6 / (8^2 * 9) = 1/48.
        self.assertAlmostEqual(post.variance[0], 1.0 / 48.0, places=15)
        self.assertAlmostEqual(post.variance[1], 1.0 / 48.0, places=15)

    def test_analyze_concentrations_alignment(self):
        rows = [(9, 1, 0), (0, 0, 0), (8, 2, 0), (9, 0, 1)]
        result = analyze_concentrations(rows)
        self.assertEqual(result.k, 3)
        self.assertEqual(result.n_groups, 4)
        self.assertEqual(result.n_nonempty_groups, 3)
        self.assertEqual(len(result.posteriors), 4)
        for post, row in zip(result.posteriors, rows):
            self.assertEqual(post.counts, tuple(row))
            self.assertEqual(post.alpha, result.fit.alphas)
            self.assertAlmostEqual(sum(post.mean), 1.0, places=12)
        self.assertTrue(result.posteriors[1].is_empty)
        # Empty group gets the normalized fitted prior, not the uniform one.
        empty_mean = result.posteriors[1].mean
        total = math.fsum(result.fit.alphas)
        for j in range(3):
            self.assertAlmostEqual(
                empty_mean[j], result.fit.alphas[j] / total, places=12
            )


class AsymmetricValidationTests(unittest.TestCase):
    def test_vector_length_must_match_k(self):
        with self.assertRaises(ValueError):
            log_evidence((0.5, 0.5, 0.5), [(1, 1)])
        with self.assertRaises(ValueError):
            log_evidence_one((0.5,), (1, 1))
        with self.assertRaises(ValueError):
            posterior_group((1, 1, 1), (1.0, 2.0))
        with self.assertRaises(ValueError):
            fit_concentrations([(1, 1)], start=(1.0, 2.0, 3.0))

    def test_vector_elements_must_be_positive_finite_numbers(self):
        for bad in [(0.5, -1.0), (0.5, 0.0), (0.5, float("nan")),
                    (0.5, float("inf"))]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    log_evidence(bad, [(1, 1)])
        with self.assertRaises(ValueError):
            fit_concentrations([(1, 1)], start=(1.0, -2.0))

    def test_vector_element_type_errors_carry_coordinates(self):
        with self.assertRaises(TypeError) as ctx:
            log_evidence((0.5, True), [(1, 1)])
        self.assertIn("alpha[1]", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            log_evidence(("0.5", 1.0), [(1, 1)])
        self.assertIn("alpha[0]", str(ctx.exception))

    def test_scalar_contract_unchanged(self):
        with self.assertRaises(TypeError):
            log_evidence(True, [(1, 1)])
        with self.assertRaises(TypeError):
            log_evidence("1.0", [(1, 1)])
        with self.assertRaises(ValueError):
            log_evidence(0.0, [(1, 1)])

    def test_iteration_parameter_validation(self):
        rows = [(2, 1), (1, 2)]
        with self.assertRaises(ValueError):
            fit_concentrations(rows, max_iter=0)
        with self.assertRaises(TypeError):
            fit_concentrations(rows, max_iter=True)
        with self.assertRaises(ValueError):
            fit_concentrations(rows, tol=0.0)
        with self.assertRaises(ValueError):
            fit_concentrations(rows, tol=1.5)
        with self.assertRaises(ValueError):
            fit_concentrations(rows, tol=float("nan"))
        with self.assertRaises(TypeError):
            fit_concentrations(rows, tol=True)
        with self.assertRaises(TypeError):
            fit_concentrations(rows, tol="1e-8")

    def test_no_partial_state_on_failure(self):
        good = [(3, 1), (0, 2)]
        with self.assertRaises(ValueError):
            fit_concentrations([(1, 1), (1, -1)])
        with self.assertRaises(ValueError):
            analyze_concentrations([(1, 1)], start=(1.0, 0.0))
        r1 = fit_concentrations(good)
        r2 = fit_concentrations(good)
        self.assertEqual(r1.alphas, r2.alphas)
        self.assertEqual(r1.log_evidence, r2.log_evidence)


if __name__ == "__main__":
    unittest.main()
