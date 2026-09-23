"""Evidence and posterior tests against the standalone K=2 reference."""

import math
import os
import unittest

from dirichlet_eb import core

from tests import reference


# Small-count K=2 datasets used throughout: mixed, concentrated, balanced.
DATASETS = [
    [(1, 0)],
    [(0, 1)],
    [(2, 1)],
    [(2, 0), (0, 2), (1, 1)],
    [(5, 0), (3, 0), (0, 4), (0, 2)],
    [(1, 2), (2, 1), (0, 3), (3, 0)],
    [(2, 2), (1, 1), (0, 0)],
]

ALPHAS = [1e-4, 3.1623e-3, 0.1, 0.7, 1.0, 2.5, 13.0, 316.0, 1e4]

EVIDENCE_TOL = 1e-10
MOMENT_TOL = 1e-12


class EvidenceReferenceTests(unittest.TestCase):
    def test_group_evidence_matches_binomial_expansion(self):
        for rows in DATASETS:
            for a in ALPHAS:
                for n1, n0 in rows:
                    got = core.log_evidence(a, [(n1, n0)], k=2)
                    want = reference.ref_log_evidence_group(n1, n0, a)
                    self.assertLessEqual(
                        abs(got - want),
                        EVIDENCE_TOL,
                        msg=f"group={(n1, n0)} alpha={a}: {got} vs {want}",
                    )

    def test_pooled_evidence_matches_reference(self):
        for rows in DATASETS:
            for a in ALPHAS:
                got = core.log_evidence(a, rows, 2)
                want = reference.ref_log_evidence(rows, a)
                self.assertLessEqual(
                    abs(got - want),
                    EVIDENCE_TOL,
                    msg=f"rows={rows} alpha={a}: {got} vs {want}",
                )

    def test_per_group_decomposition_with_empty_groups(self):
        rows = [(0, 0), (2, 1), (0, 0), (0, 3)]
        per, validated, k = core.log_evidence_per_group(0.5, rows, 2)
        self.assertEqual(k, 2)
        self.assertEqual(validated, tuple(rows))
        self.assertEqual(per[0], 0.0)
        self.assertEqual(per[2], 0.0)
        self.assertNotEqual(per[1], 0.0)
        self.assertAlmostEqual(math.fsum(per), core.log_evidence(0.5, rows, 2), places=14)

    def test_empty_group_is_exactly_one_likelihood_at_every_alpha(self):
        for a in [1e-4, 1.0, 1e4]:
            self.assertEqual(core.log_evidence(a, [(0, 0)], 2), 0.0)

    def test_posterior_moments_match_beta_formulas(self):
        rows = [(2, 1), (0, 0), (4, 0)]
        for a in [0.05, 1.0, 77.0]:
            posts = core.compute_posteriors(a, rows, 2)
            for post, (c0, c1) in zip(posts, rows):
                # reference signature is ref_posterior(count_cat1, count_cat0)
                want_mean, want_var = reference.ref_posterior(c1, c0, a)
                for got, want in zip(post.mean, want_mean):
                    self.assertLessEqual(abs(got - want), MOMENT_TOL)
                for got, want in zip(post.variance, want_var):
                    self.assertLessEqual(abs(got - want), MOMENT_TOL)

    def test_predictive_is_posterior_mean_and_normalized(self):
        posts = core.compute_posteriors(1.3, [(3, 0, 1), (0, 0, 0)], 3)
        for post in posts:
            self.assertEqual(post.predictive, post.mean)
            self.assertAlmostEqual(sum(post.predictive), 1.0, places=14)
            self.assertAlmostEqual(sum(post.mean), 1.0, places=14)
            for v in post.variance:
                self.assertGreater(v, 0.0)

    def test_reference_module_does_not_import_core(self):
        import ast

        path = os.path.join(os.path.dirname(__file__), "reference.py")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertEqual(imported, {"decimal", "math"})


if __name__ == "__main__":
    unittest.main()
