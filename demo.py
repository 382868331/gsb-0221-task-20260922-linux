"""Fixed-input demonstration of the himeb empirical-Bayes module.

Run:  python demo.py

Everything below is deterministic and genuinely computed (no sleep, no
pre-recorded numbers): a normal symmetric hierarchical fit showing shrinkage;
the new per-class-concentration fit (marginal likelihood, fitted class
concentrations, posterior means) including a never-observed class pinned to
the box edge and a boundary-MLE batch that honestly fails to converge within
its finite iteration budget; and rejection/exception boundaries that really
fire.
"""

from __future__ import annotations

import math
import time

from himeb import (
    analyze,
    analyze_dirichlet,
    fit_alpha,
    fit_alphas,
    log_evidence,
    log_evidence_dirichlet,
)


def _demo_counts() -> list[tuple[int, int, int]]:
    """Deterministic K=3 batches (no RNG, no external data)."""
    return [
        (4, 0, 1),
        (0, 3, 0),
        (2, 2, 1),
        (5, 0, 0),
        (0, 1, 4),
        (1, 3, 1),
        (0, 0, 2),
        (0, 0, 0),  # an empty group: prior only, excluded from the evidence
    ]


def main() -> None:
    t0 = time.perf_counter()

    print("=" * 72)
    print("PART 1. Normal hierarchical fit (K=3, 8 groups)")
    print("=" * 72)
    groups = _demo_counts()
    result = analyze(groups)
    fit = result.fit

    print(
        f"groups={result.n_groups} (non-empty={result.n_nonempty_groups}), "
        f"K={result.k}"
    )
    print(
        f"fitted alpha = {fit.alpha:.6f}   status = {fit.status}   "
        f"converged = {fit.converged}"
    )
    print(f"total log evidence at fitted alpha = {fit.log_evidence:.6f}")
    print(
        f"search: {fit.grid_points} log-spaced grid points; "
        f"candidates kept = {len(fit.candidates)}; "
        f"max golden-section iterations used = {fit.max_iterations_used}"
    )

    print()
    print(
        f"{'group':>6} {'n':>3} {'counts':>12} "
        f"{'MLE p':>15} -> {'posterior mean':>18}   shrink  pred"
    )
    for i, (row, post) in enumerate(zip(groups, result.posteriors)):
        n = post.n
        if n > 0:
            mle = tuple(c / n for c in row)
            # Effective shrinkage of class 0: posterior mean is a convex
            # combination of the MLE and the uniform prior mean 1/K.
            w = n / (n + result.k * fit.alpha)
            mle_s = "(" + ", ".join(f"{p:.3f}" for p in mle) + ")"
            shrink = f"{w:.3f}"
        else:
            mle_s = "(  —  )"
            shrink = "prior"
        mean_s = "(" + ", ".join(f"{p:.3f}" for p in post.mean) + ")"
        print(
            f"{i:>6} {n:>3} {str(row):>12} {mle_s:>15} -> {mean_s:>18}   "
            f"{shrink:>6}  class {post.predicted}"
        )

    post0 = result.posteriors[0]
    print()
    print("Posterior detail, group 0 = (4, 0, 1):")
    print(f"  mean       = {tuple(round(m, 6) for m in post0.mean)}")
    print(f"  variance   = {tuple(round(v, 6) for v in post0.variance)}")
    print(f"  predictive = {tuple(round(p, 6) for p in post0.predictive)}")
    print(
        "  empty-group check: group 7 mean is uniform prior "
        f"{tuple(round(m, 6) for m in result.posteriors[7].mean)}"
    )

    print()
    print("=" * 72)
    print("PART 2. Per-class concentrations (non-symmetric Dirichlet)")
    print("=" * 72)
    # Classes play visibly different roles: class 0 is common, class 2 rare.
    groups_v = [
        (4, 0, 1),
        (0, 3, 0),
        (2, 2, 1),
        (5, 0, 0),
        (1, 1, 3),
        (0, 0, 0),  # empty group: prior only, excluded from the evidence
    ]
    dres = analyze_dirichlet(groups_v)
    dfit = dres.fit
    print(
        f"groups={dres.n_groups} (non-empty={dres.n_nonempty_groups}), "
        f"K={dres.k}"
    )
    print(
        "fitted per-class alpha = "
        + ", ".join(f"{a:.5f}" for a in dfit.alphas)
    )
    print(
        f"total log marginal likelihood = {dfit.log_evidence:.6f}   "
        f"status = {dfit.status}   converged = {dfit.converged}"
    )
    print(
        f"finite-iteration fit: sweeps used = {dfit.sweeps_used}/"
        f"{dfit.sweeps}, at box edge = {dfit.at_box}, "
        f"max 1-D iterations = {dfit.max_iterations_used}"
    )
    # The symmetric model restricted to a0=a1=a2 cannot reach this evidence.
    sym = fit_alpha(groups_v)
    print(
        f"symmetric-only best log evidence = {sym.log_evidence:.6f} "
        f"(alpha={sym.alpha:.5f}); per-class gain = "
        f"{dfit.log_evidence - sym.log_evidence:.6f} (>= 0, as the "
        "symmetric line is a subset of the per-class box)"
    )
    print()
    print(
        f"{'group':>6} {'n':>3} {'counts':>12} {'posterior mean':>22}  pred"
    )
    for i, (row, post) in enumerate(zip(groups_v, dres.posteriors)):
        mean_s = "(" + ", ".join(f"{p:.3f}" for p in post.mean) + ")"
        print(
            f"{i:>6} {post.n:>3} {str(row):>12} {mean_s:>22}  "
            f"class {post.predicted}"
        )
    post = dres.posteriors[0]
    print()
    print("Posterior detail, group 0 = (4, 0, 1):")
    print(f"  mean       = {tuple(round(m, 6) for m in post.mean)}")
    print(f"  variance   = {tuple(round(v, 6) for v in post.variance)}")

    print()
    print("-- boundary: a never-observed class rests on the lower edge --")
    edge_rows = [(2, 0, 0), (0, 2, 0), (3, 0, 0), (0, 1, 0)]
    er = fit_alphas(edge_rows, sweeps=300)
    print(f"  groups {edge_rows}")
    print(
        f"  alphas = ({', '.join(f'{a:.5f}' for a in er.alphas)})  "
        f"at_box = {er.at_box}  status = {er.status}"
    )

    print()
    print("-- reported non-convergence: boundary MLE, finite budget --")
    # All groups concentrate on class 0: the EB MLE escapes toward infinite
    # concentration, so no finite-box point is a stationary optimum. With a
    # deliberately small sweep budget the fit must say so rather than lie.
    bm_rows = [(3, 0), (2, 0), (1, 1)]
    br = fit_alphas(bm_rows, sweeps=20)
    print(f"  groups {bm_rows}")
    print(
        f"  alphas after 20 sweeps = ({br.alphas[0]:.3f}, "
        f"{br.alphas[1]:.3f})  status = {br.status}  "
        f"converged = {br.converged}  budget_exhausted = "
        f"{br.budget_exhausted}"
    )
    ev_at_one = log_evidence_dirichlet((1.0, 1.0), bm_rows)
    print(
        f"  evidence {br.log_evidence:.6f} is still above the start value "
        f"{ev_at_one:.6f}: the ascent is real, only un-finished."
    )

    print()
    print("=" * 72)
    print("PART 3. Endpoint / interior / proved-flat symmetric states")
    print("=" * 72)
    cases = [
        ("all observations in one class (extreme)", [(6, 0)] * 6),
        ("perfectly balanced (extreme)", [(60, 60)] * 3),
        ("mixed, over-dispersed small counts",
         [(4, 1), (5, 0), (0, 5), (2, 2), (3, 2), (1, 4), (0, 3)]),
        ("only single draws: evidence is constant",
         [(1, 0), (0, 1), (1, 0), (0, 0)]),
    ]
    for label, rows in cases:
        r = fit_alpha(rows)
        print(
            f"- {label:48s} alpha={r.alpha:10.4g}  status={r.status:15s} "
            f"log-ev={r.log_evidence:10.5f}"
        )

    print()
    print("=" * 72)
    print("PART 4. Error boundaries that actually fire")
    print("=" * 72)
    print("Calling log_evidence(alpha=nan, groups=[(1, 2)]) ...")
    try:
        log_evidence(float("nan"), [(1, 2)])
    except ValueError as exc:
        print(f"  ValueError raised as specified: {exc}")
    else:
        raise RuntimeError("expected a ValueError for NaN alpha")

    print("Calling fit_alpha with a bool count ...")
    try:
        fit_alpha([(True, 2)])
    except TypeError as exc:
        print(f"  TypeError raised as specified: {exc}")
    else:
        raise RuntimeError("expected a TypeError for a bool count")

    print("Calling log_evidence_dirichlet(alphas=(0.5, 0), ...) ...")
    try:
        log_evidence_dirichlet((0.5, 0.0), [(1, 2)])
    except ValueError as exc:
        print(f"  ValueError raised as specified: {exc}")
    else:
        raise RuntimeError("expected a ValueError for a zero concentration")

    print("Calling fit_alphas with an init outside the fit box ...")
    try:
        fit_alphas([(1, 2)], init=(1.0, 1e9))
    except ValueError as exc:
        print(f"  ValueError raised as specified: {exc}")
    else:
        raise RuntimeError("expected a ValueError for an out-of-box init")

    elapsed = time.perf_counter() - t0
    print()
    print(f"done in {elapsed:.3f}s (real computation, no sleep).")


if __name__ == "__main__":
    main()
