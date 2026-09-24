"""Per-class (non-symmetric Dirichlet) concentration: evidence, posterior, fit.

Covers the iteration that extends the symmetric prior to independent
class-specific concentrations:

* evidence/posterior correctness against hand-computed fractions and the
  independent asymmetric beta-binomial expansion (never calls himeb for
  expected values);
* symmetry degeneracy (exchangeable data -> equal concentrations; symmetric
  init stays exactly on the symmetry ray);
* zero counts (empty groups ignored; empty-only batch is the proved-flat
  case; a never-observed class rests on the lower box edge);
* class permutation equivariance;
* finite-iteration convergence reporting (converged vs max_sweeps_reached);
* validation: positivity, length, types, init box, search parameters.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import (  # noqa: E402
    beta_moments_via_integrals_ab,
    beta_posterior_moments_ab,
    dense_alphas_scan_2d,
    digamma,
    evidence_expansion_ab,
    evidence_polya,
    evidence_polya_k,
    evidence_quadrature_ab,
)

from himeb import (  # noqa: E402
    ALPHA_MAX,
    ALPHA_MIN,
    analyze_dirichlet,
    fit_alpha,
    fit_alphas,
    log_evidence,
    log_evidence_dirichlet,
    posterior_group_dirichlet,
)


class EvidenceDirichletTests(unittest.TestCase):
    def test_vs_independent_polya_urn(self):
        # Primary independent route: the sequential Polya-urn factorization
        # (products of positive fractions, no alternating cancellation) is
        # accurate across the whole [1e-4, 1e4] box, including extreme
        # asymmetric corners where the expansion cancels.
        pairs = [(1e-3, 4.0), (0.07, 0.9), (2.0, 0.3), (11.0, 0.02),
                 (1.0, 1.0), (50.0, 50.0), (1e4, 0.01), (1e4, 1e-4),
                 (3.0, 7.0), (0.2, 0.4)]
        worst = 0.0
        for n in range(0, 9):
            for x in range(0, n + 1):
                for a0, a1 in pairs:
                    with self.subTest(x=x, n=n, a0=a0, a1=a1):
                        got = log_evidence_dirichlet((a0, a1), [(x, n - x)])
                        ref = evidence_polya(x, n, a0, a1)
                        worst = max(worst, abs(got - ref))
                        self.assertLessEqual(
                            abs(got - ref), 1e-12,
                            msg=f"({x},{n-x}) ({a0},{a1}): {got} vs {ref}",
                        )
        # Production is integer-log-sum based; observed agreement is far
        # inside binary64 rounding, not merely "within a loose tolerance".
        self.assertLessEqual(worst, 2e-14)

    def test_vs_asymmetric_expansion_in_its_valid_envelope(self):
        # Second independent route: binomial expansion + beta integrals. Its
        # alternating sum loses significance in extreme asymmetric corners
        # (documented reference limitation), so only use its accurate
        # envelope: moderate a0, small counts.
        pairs = [(1e-3, 4.0), (0.07, 0.9), (2.0, 0.3), (1.0, 1.0),
                 (50.0, 50.0)]
        for n in range(0, 9):
            for x in range(0, n + 1):
                for a0, a1 in pairs:
                    with self.subTest(x=x, n=n, a0=a0, a1=a1):
                        got = log_evidence_dirichlet((a0, a1), [(x, n - x)])
                        ref = evidence_expansion_ab(x, n, a0, a1)
                        self.assertLessEqual(abs(got - ref), 2e-8)

    def test_vs_direct_quadrature(self):
        # Third independent route: direct asymmetric Simpson integration,
        # valid where the integrand is not a sub-resolution spike.
        for x, n in [(1, 2), (2, 3), (3, 5), (0, 4), (2, 4)]:
            for a0, a1 in ((1.0, 2.5), (2.5, 1.0), (5.0, 2.0), (1.0, 1.0)):
                with self.subTest(x=x, n=n, a0=a0, a1=a1):
                    got = log_evidence_dirichlet((a0, a1), [(x, n - x)])
                    ref = evidence_quadrature_ab(x, n, a0, a1)
                    self.assertLessEqual(abs(got - ref), 2e-9)

    def test_batches_vs_polya(self):
        rows = [(0, 2), (3, 1), (5, 0), (0, 0), (1, 4)]
        for a0, a1 in ((1e-3, 2.0), (0.7, 0.4), (3.0, 9.0), (1e4, 0.01)):
            with self.subTest(a0=a0, a1=a1):
                got = log_evidence_dirichlet((a0, a1), rows)
                ref = math.fsum(
                    evidence_polya(c0, c0 + c1, a0, a1)
                    for c0, c1 in rows
                )
                self.assertLessEqual(abs(got - ref), 1e-12)
            if a0 <= 50.0:
                ref = math.fsum(
                    evidence_expansion_ab(c0, c0 + c1, a0, a1)
                    for c0, c1 in rows
                )
                self.assertLessEqual(
                    abs(log_evidence_dirichlet((a0, a1), rows) - ref), 5e-8
                )

    def test_hand_computed_single_draws(self):
        # A lone draw under Dirichlet(a0, a1) has marginal mass a_j/(a0+a1),
        # an exact fraction independent of any lgamma machinery.
        a0, a1 = 0.5, 2.0
        self.assertAlmostEqual(
            log_evidence_dirichlet((a0, a1), [(1, 0)]), math.log(0.2),
            places=15,
        )
        self.assertAlmostEqual(
            log_evidence_dirichlet((a0, a1), [(0, 1)]), math.log(0.8),
            places=15,
        )
        # Two independent draws (1,0) then (0,1): 0.2 * 0.8 = 0.16.
        self.assertAlmostEqual(
            log_evidence_dirichlet((a0, a1), [(1, 0), (0, 1)]),
            math.log(0.16), places=14,
        )
        # (2,0) beta-binomial: a0(a0+1)/[(a0+a1)(a0+a1+1)] = 3/35.
        self.assertAlmostEqual(
            log_evidence_dirichlet((a0, a1), [(2, 0)]),
            math.log(3.0 / 35.0), places=14,
        )

    def test_symmetric_vector_matches_scalar_evidence(self):
        rows = [(4, 0, 1), (0, 3, 0), (2, 2, 1), (0, 0, 0), (5, 0, 2)]
        for a in (1e-4, 0.13, 1.0, 37.0, 1e4):
            with self.subTest(a=a):
                self.assertEqual(
                    log_evidence_dirichlet((a, a, a), rows),
                    log_evidence(a, rows),
                )

    def test_empty_groups_are_zero_and_ignored(self):
        self.assertEqual(log_evidence_dirichlet((0.3, 2.0), [(0, 0)]), 0.0)
        base = [(2, 1), (0, 3)]
        v = log_evidence_dirichlet((1.1, 0.4), base)
        self.assertEqual(
            log_evidence_dirichlet((1.1, 0.4), [(0, 0)] + base), v
        )
        self.assertEqual(
            log_evidence_dirichlet((1.1, 0.4), base + [(0, 0), (0, 0)]), v
        )


class PosteriorDirichletTests(unittest.TestCase):
    def test_hand_computed_moments(self):
        # (2, 0) with prior (0.5, 2) -> posterior (2.5, 2), total 4.5.
        post = posterior_group_dirichlet((2, 0), (0.5, 2.0))
        self.assertAlmostEqual(post.mean[0], 2.5 / 4.5, places=15)
        self.assertAlmostEqual(post.mean[1], 2.0 / 4.5, places=15)
        self.assertAlmostEqual(
            post.variance[0], 2.5 * 2.0 / (4.5 ** 2 * 5.5), places=15
        )
        self.assertAlmostEqual(post.variance[1], post.variance[0], places=15)
        self.assertEqual(post.predicted, 0)
        self.assertEqual(post.predictive, post.mean)
        self.assertAlmostEqual(sum(post.mean), 1.0, places=15)

    def test_vs_independent_beta_moments(self):
        for c0, c1 in [(0, 0), (2, 1), (3, 3), (0, 5), (5, 0), (4, 2)]:
            n = c0 + c1
            for a0, a1 in ((0.2, 3.0), (1.0, 1.0), (5.0, 0.4), (1e3, 0.7)):
                with self.subTest(c0=c0, c1=c1, a0=a0, a1=a1):
                    post = posterior_group_dirichlet((c0, c1), (a0, a1))
                    m0, _, v0 = beta_posterior_moments_ab(c0, n, a0, a1)
                    m1, _, v1 = beta_posterior_moments_ab(c1, n, a1, a0)
                    self.assertAlmostEqual(post.mean[0], m0, places=12)
                    self.assertAlmostEqual(post.mean[1], m1, places=12)
                    self.assertAlmostEqual(post.variance[0], v0, places=12)
                    self.assertAlmostEqual(post.variance[1], v1, places=12)
                    if max(a0, a1, n) <= 10.0:
                        i0, _, iv0 = beta_moments_via_integrals_ab(
                            c0, n, a0, a1
                        )
                        i1, _, iv1 = beta_moments_via_integrals_ab(
                            c1, n, a1, a0
                        )
                        self.assertAlmostEqual(post.mean[0], i0, places=12)
                        self.assertAlmostEqual(post.variance[0], iv0, places=12)
                        self.assertAlmostEqual(post.mean[1], i1, places=12)
                        self.assertAlmostEqual(post.variance[1], iv1, places=12)

    def test_empty_group_returns_prior(self):
        post = posterior_group_dirichlet((0, 0, 0), (0.5, 2.0, 0.25))
        self.assertTrue(post.is_empty)
        total = 2.75
        for j, a in enumerate((0.5, 2.0, 0.25)):
            self.assertAlmostEqual(post.mean[j], a / total, places=15)
            self.assertAlmostEqual(
                post.variance[j],
                a * (total - a) / (total ** 2 * (total + 1.0)),
                places=14,
            )


class FitAlphasTests(unittest.TestCase):
    def test_interior_fit_vs_independent_2d_scan(self):
        rows = [(4, 0), (3, 1), (0, 2)]
        r = fit_alphas(rows)
        self.assertEqual(r.status, "converged")
        self.assertTrue(r.converged)
        self.assertFalse(r.budget_exhausted)
        self.assertFalse(any(r.at_box))
        (g0, g1), gv = dense_alphas_scan_2d(rows, n_points=201)
        # Log-grid spacing of the reference is ~9% per cell; the coordinate
        # fit must beat it and agree to that resolution.
        self.assertGreaterEqual(r.log_evidence, gv - 1e-12)
        self.assertLess(abs(r.alphas[0] - g0) / g0, 0.12)
        self.assertLess(abs(r.alphas[1] - g1) / g1, 0.12)
        # Evidence at the reported point is self-consistent.
        self.assertAlmostEqual(
            r.log_evidence,
            math.fsum(
                evidence_expansion_ab(c0, c0 + c1, r.alphas[0], r.alphas[1])
                for c0, c1 in rows
            ),
            places=10,
        )

    def test_stationarity_via_independent_digamma(self):
        # At an interior optimum every partial derivative is zero. With
        # A = sum_k alpha_k, n_g > 0:
        #   d l / d a_j = sum_g [psi(a_j + x_gj) - psi(a_j)
        #                         + psi(A) - psi(A + n_g)]
        # evaluated with the reference digamma, not with production code.
        rows = [(4, 0, 1), (0, 3, 0), (2, 2, 1), (5, 0, 0), (1, 1, 3)]
        r = fit_alphas(rows, sweeps=300)
        self.assertEqual(r.status, "converged")
        self.assertFalse(any(r.at_box))
        a = r.alphas
        A = math.fsum(a)
        for j in range(3):
            grad = 0.0
            for row in rows:
                n = sum(row)
                if n > 0:
                    grad += (
                        digamma(a[j] + row[j]) - digamma(a[j])
                        + digamma(A) - digamma(A + n)
                    )
            self.assertLess(
                abs(grad), 1e-5,
                msg=f"class {j}: gradient {grad} not ~0 at {a}",
            )

    def test_symmetric_degeneracy_equal_data_equal_alphas(self):
        # Class-exchangeable data: the unique joint maximizer lies exactly on
        # a0 = a1; a symmetric init must stay exactly on the symmetry ray.
        rows = [(2, 2), (3, 1), (1, 3), (4, 0), (0, 4)]
        r = fit_alphas(rows, init=(1.0, 1.0), sweeps=200)
        self.assertEqual(r.status, "converged")
        self.assertEqual(r.alphas[0], r.alphas[1])
        # It agrees with the scalar symmetric fit (same engine, same box).
        rs = fit_alpha(rows)
        if rs.status == "interior":
            self.assertAlmostEqual(r.alphas[0], rs.alpha, delta=1e-10 * rs.alpha)
            self.assertAlmostEqual(r.log_evidence, rs.log_evidence, places=12)

    def test_symmetric_degeneracy_from_off_symmetric_init(self):
        # Same exchangeable data from an asymmetric start: still converges to
        # equal concentrations (unique optimum of the strictly concave
        # objective), up to the finite iteration tolerance.
        rows = [(2, 2), (3, 1), (1, 3), (4, 0), (0, 4)]
        r = fit_alphas(rows, init=(3.0, 0.2), sweeps=300)
        self.assertEqual(r.status, "converged")
        self.assertLess(
            abs(r.alphas[0] - r.alphas[1]) / r.alphas[0], 1e-5
        )

    def test_zero_counts_flat_and_exclusion(self):
        r = fit_alphas([(0, 0, 0), (0, 0, 0)])
        self.assertEqual(r.status, "flat")
        self.assertTrue(r.converged)
        self.assertFalse(r.budget_exhausted)
        self.assertEqual(r.sweeps_used, 0)
        self.assertEqual(r.alphas, (1.0, 1.0, 1.0))
        self.assertEqual(r.log_evidence, 0.0)

        base = [(4, 1), (2, 3), (0, 2)]
        r1 = fit_alphas(base)
        r2 = fit_alphas([(0, 0)] + base + [(0, 0), (0, 0)])
        for j in range(2):
            self.assertAlmostEqual(r1.alphas[j], r2.alphas[j], places=12)
        self.assertEqual(r2.n_nonempty_groups, 3)
        self.assertEqual(r2.log_evidence, r1.log_evidence)

    def test_unobserved_class_pinned_to_lower_edge(self):
        # Class 2 never receives a count: its concentration is minimized
        # (evidence increases as an unseen alpha shrinks), so its coordinate
        # maximum is the lower box edge. The two observed classes have
        # fluctuating relative frequencies, so their fitted concentrations
        # stay interior and finite.
        rows = [(2, 0, 0), (0, 2, 0), (3, 0, 0), (0, 1, 0), (1, 0, 0)]
        r = fit_alphas(rows, sweeps=300)
        self.assertEqual(r.status, "converged")
        self.assertTrue(r.at_box[2])
        self.assertAlmostEqual(r.alphas[2], ALPHA_MIN)
        self.assertFalse(r.at_box[0])
        self.assertFalse(r.at_box[1])
        for a in r.alphas[:2]:
            self.assertGreater(a, ALPHA_MIN * 10)
            self.assertLess(a, ALPHA_MAX / 10)
        # Raising the unseen class concentration must not improve evidence;
        # the comparison reuses the public per-class evidence entry.
        base = log_evidence_dirichlet(r.alphas, rows)
        for a2 in (ALPHA_MIN * 5, 1.0, 100.0):
            self.assertLess(
                log_evidence_dirichlet(
                    (r.alphas[0], r.alphas[1], a2), rows
                ),
                base,
            )

    def test_single_draws_are_not_called_flat(self):
        # Unlike the symmetric model, n=1 rows carry information under
        # per-class concentrations: p(x=e_j) = a_j / A. No 'flat' claim is
        # made. With each class observed once the exchangeable optimum is
        # a0 = a1 and the evidence is the exact constant -2 log 2 along the
        # whole symmetry ray (a flat ridge, not a unique maximizer): the
        # finite-budget search reports convergence on the ridge and the
        # evidence value is exact.
        r = fit_alphas([(1, 0), (0, 1)], sweeps=200)
        self.assertNotEqual(r.status, "flat")
        self.assertAlmostEqual(r.alphas[0], r.alphas[1], delta=1e-9)
        self.assertAlmostEqual(r.log_evidence, -2.0 * math.log(2.0), places=12)

    def test_nonconvergence_is_reported_on_boundary_mle(self):
        # When every non-empty group puts all its mass on class 0, the
        # empirical-Bayes MLE escapes to infinite concentration (a point-mass
        # prior): the maximizer is outside every finite box. The coordinate
        # ascent keeps climbing toward the upper edge and must honestly report
        # that its finite sweep budget was exhausted rather than "converged".
        rows = [(3, 0), (2, 0), (1, 1)]
        r = fit_alphas(rows, sweeps=60)
        self.assertFalse(r.converged)
        self.assertTrue(r.budget_exhausted)
        self.assertEqual(r.status, "max_sweeps_reached")
        self.assertEqual(r.sweeps_used, 60)
        # The climb is monotone: each reported point beats the box center.
        self.assertGreater(
            r.log_evidence,
            log_evidence_dirichlet((1.0, 1.0), rows),
        )

    def test_nonconvergence_under_tiny_budget(self):
        # A genuinely interior fit simply cut off after one sweep reports the
        # same non-convergence status; more budget converges and cannot lower
        # the evidence.
        rows = [(4, 0), (3, 1), (0, 2), (5, 1), (1, 3), (2, 2)]
        r1 = fit_alphas(rows, sweeps=1)
        self.assertFalse(r1.converged)
        self.assertTrue(r1.budget_exhausted)
        self.assertEqual(r1.status, "max_sweeps_reached")
        self.assertEqual(r1.sweeps_used, 1)
        r_full = fit_alphas(rows, sweeps=300)
        self.assertTrue(r_full.converged)
        self.assertEqual(r_full.status, "converged")
        self.assertGreaterEqual(r_full.log_evidence + 1e-12, r1.log_evidence)

    def test_coordinate_maximality_via_polya_reference(self):
        # At the fitted interior point, perturbing one coordinate a few
        # percent in either direction cannot raise the evidence. Expected
        # values come from the independent general-K Polya-urn fractions.
        rows = [(4, 0, 1), (0, 3, 0), (2, 2, 1), (5, 0, 0), (1, 1, 3)]
        r = fit_alphas(rows, sweeps=300)
        self.assertEqual(r.status, "converged")
        for a in r.alphas:
            self.assertGreaterEqual(a, ALPHA_MIN)
            self.assertLessEqual(a, ALPHA_MAX)
        for j in range(3):
            if r.at_box[j]:
                continue
            for factor in (0.8, 0.95, 1.05, 1.2):
                vec = list(r.alphas)
                vec[j] = min(
                    ALPHA_MAX, max(ALPHA_MIN, vec[j] * factor)
                )
                v = math.fsum(
                    evidence_polya_k(row, vec) for row in rows
                )
                self.assertLessEqual(
                    v, r.log_evidence + 1e-9,
                    msg=f"class {j} factor {factor}: {v} > {r.log_evidence}",
                )

    def test_per_class_fit_dominates_symmetric_fit(self):
        rows = [(4, 0, 1), (0, 3, 0), (2, 2, 1), (5, 0, 0)]
        rs = fit_alpha(rows)
        rv = fit_alphas(rows, sweeps=300)
        # The symmetric line a_j = alpha* is a subset of the per-class box.
        self.assertGreaterEqual(rv.log_evidence + 1e-10, rs.log_evidence)

    def test_class_permutation_equivariance(self):
        rows = [(4, 0, 1), (0, 3, 0), (2, 1, 2)]
        perm = (2, 0, 1)
        rows_p = [tuple(row[perm[j]] for j in range(3)) for row in rows]
        r = fit_alphas(rows, sweeps=300)
        rp = fit_alphas(rows_p, sweeps=300)
        # Finite-iteration coordinate ascent resolves the optimum to ~1e-6
        # relative; equivariance must hold at that resolution (both runs
        # independently find the same objective value much more tightly).
        for new, old in enumerate(perm):
            self.assertAlmostEqual(
                rp.alphas[new], r.alphas[old],
                delta=1e-5 * max(1.0, r.alphas[old]),
            )
        self.assertAlmostEqual(r.log_evidence, rp.log_evidence, places=10)
        # Posterior means permute the same way.
        post = posterior_group_dirichlet(rows[0], r.alphas)
        post_p = posterior_group_dirichlet(rows_p[0], rp.alphas)
        for new, old in enumerate(perm):
            self.assertAlmostEqual(post_p.mean[new], post.mean[old], places=7)


class AnalyzeDirichletTests(unittest.TestCase):
    def test_alignment_and_structure(self):
        rows = [(3, 1, 0), (0, 0, 0), (0, 2, 4), (1, 1, 1)]
        result = analyze_dirichlet(rows)
        self.assertEqual(result.k, 3)
        self.assertEqual(result.n_groups, 4)
        self.assertEqual(result.n_nonempty_groups, 3)
        self.assertEqual(len(result.posteriors), 4)
        for post, row in zip(result.posteriors, rows):
            self.assertEqual(post.counts, tuple(row))
            self.assertEqual(post.alphas, result.fit.alphas)
            self.assertAlmostEqual(sum(post.mean), 1.0, places=14)
        self.assertTrue(result.posteriors[1].is_empty)

    def test_contract_limits_accepted(self):
        rows = [(1, 1, 0, 2, 0, 1)] * 40
        result = analyze_dirichlet(rows)
        self.assertEqual(result.k, 6)
        self.assertEqual(result.n_groups, 40)


class AlphasValidationTests(unittest.TestCase):
    def test_reject_bad_concentrations(self):
        for bad in (
            (0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0),
            (1.0, float("inf")), (1.0, float("-inf")),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    log_evidence_dirichlet(bad, [(1, 1)])
                with self.assertRaises(ValueError):
                    posterior_group_dirichlet((1, 1), bad)

    def test_reject_nonpositive_or_mislength(self):
        with self.assertRaises(ValueError):
            log_evidence_dirichlet((1.0,), [(1, 1)])
        with self.assertRaises(ValueError):
            log_evidence_dirichlet((1.0,) * 7, [(1,) * 7])
        with self.assertRaises(ValueError):
            log_evidence_dirichlet((1.0, 1.0, 1.0), [(1, 1)])

    def test_reject_types_and_bool(self):
        with self.assertRaises(TypeError):
            log_evidence_dirichlet("(1,1)", [(1, 1)])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            log_evidence_dirichlet((True, 1.0), [(1, 1)])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            log_evidence_dirichlet((1.0, "x"), [(1, 1)])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            posterior_group_dirichlet((1, 1), (1, False))  # type: ignore[arg-type]

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], init=(0.0, 1.0))
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], init=(1.0, 1e5))
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], init=(1.0, 1.0, 1.0))
        with self.assertRaises(TypeError):
            fit_alphas([(1, 1)], init=(True, 1.0))  # type: ignore[arg-type]

    def test_search_parameter_bounds(self):
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], sweeps=0)
        with self.assertRaises(TypeError):
            fit_alphas([(1, 1)], sweeps=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], grid_points=2)
        with self.assertRaises(ValueError):
            fit_alphas([(1, 1)], max_iter=0)

    def test_batch_validation_propagates(self):
        with self.assertRaises(ValueError):
            fit_alphas([(1,)])
        with self.assertRaises(ValueError) as ctx:
            fit_alphas([(1, 1), (2, -1)])
        self.assertIn("group 1", str(ctx.exception))
        with self.assertRaises(TypeError):
            fit_alphas("nope")  # type: ignore[arg-type]

    def test_no_partial_state_on_failure(self):
        # Pure functions: a rejected batch raises before any result; a
        # follow-up valid call behaves exactly as in a fresh process.
        good = [(3, 1), (0, 2)]
        with self.assertRaises(ValueError):
            analyze_dirichlet([(1, 1), (1, -1)])
        r1 = analyze_dirichlet(good)
        r2 = analyze_dirichlet(good)
        self.assertEqual(r1.fit.alphas, r2.fit.alphas)
        self.assertEqual(r1.fit.log_evidence, r2.fit.log_evidence)


if __name__ == "__main__":
    unittest.main()
