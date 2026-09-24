"""Fixed-input demonstration of the himeb empirical-Bayes module.

Run:  python demo.py

Everything below is deterministic and genuinely computed (no sleep, no
pre-recorded numbers): a normal hierarchical fit showing shrinkage, the three
optimizer statuses on extreme inputs, and one rejection/exception boundary
that really fires.
"""

from __future__ import annotations

import math
import time

from himeb import (
    ALPHA_MIN,
    analyze,
    analyze_concentrations,
    fit_alpha,
    fit_concentrations,
    log_evidence,
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
    print("PART 2. Endpoint / interior / proved-flat optimizer states")
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
    print("PART 3. An error boundary that actually fires")
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

    print()
    print("=" * 72)
    print("PART 4. Per-class concentrations Dirichlet(a1,...,aK): fit + posteriors")
    print("=" * 72)
    # Lopsided, over-dispersed deterministic batch (class 0 dominant).
    rows = [
        (9, 0, 0), (5, 4, 0), (8, 0, 1), (4, 5, 0),
        (9, 1, 0), (3, 6, 0), (7, 1, 1), (2, 6, 1),
    ]
    result = analyze_concentrations(rows)
    fit = result.fit
    print(
        f"fitted alphas = {tuple(round(a, 6) for a in fit.alphas)}   "
        f"status = {fit.status}   converged = {fit.converged}"
    )
    print(
        f"iterations = {fit.iterations}/{fit.max_iter}   "
        f"final relative step = {fit.final_step:.3g}   "
        f"log evidence = {fit.log_evidence:.6f}"
    )
    sym = fit_alpha(rows)
    print(
        f"symmetric-prior fit for comparison: alpha = {sym.alpha:.6f} "
        f"(status {sym.status}), log evidence = {sym.log_evidence:.6f}"
    )
    print(
        f"per-class evidence gain over symmetric = "
        f"{fit.log_evidence - sym.log_evidence:.6f}"
    )
    print("posterior means with the fitted per-class prior:")
    for i, (row, post) in enumerate(zip(rows, result.posteriors)):
        mean_s = "(" + ", ".join(f"{p:.3f}" for p in post.mean) + ")"
        print(f"  group {i} {str(row):>10} -> mean {mean_s}  pred class {post.predicted}")

    print()
    print("=" * 72)
    print("PART 5. Concentration boundaries: clamping and non-convergence")
    print("=" * 72)
    # Class 1 is never observed: its concentration is clamped to ALPHA_MIN
    # (never zero), while class 0's likelihood seeks the upper boundary and
    # the finite iteration budget is honestly reported as exhausted.
    zfit = fit_concentrations([(5, 0), (9, 0), (2, 0), (6, 0)])
    print(
        f"never-observed class: alphas = {zfit.alphas!r}  "
        f"status = {zfit.status}  clamped_classes = {zfit.clamped_classes}"
    )
    print(f"  ALPHA_MIN = {ALPHA_MIN} (positivity constraint held)")
    one = fit_concentrations(rows, max_iter=1)
    print(
        f"budget of 1 iteration: converged = {one.converged}  "
        f"status = {one.status}  iterations = {one.iterations} "
        "(reported, not raised)"
    )
    # All-empty batch: proved flat, no iteration at all.
    flat = fit_concentrations([(0, 0, 0), (0, 0, 0)])
    print(
        f"all-empty batch: status = {flat.status}  alphas = {flat.alphas}  "
        f"iterations = {flat.iterations}"
    )

    print()
    print("=" * 72)
    print("PART 6. A per-class error boundary that actually fires")
    print("=" * 72)
    print("Calling log_evidence with a 3-vector alpha on K=2 data ...")
    try:
        log_evidence((0.5, 0.5, 0.5), [(1, 2)])
    except ValueError as exc:
        print(f"  ValueError raised as specified: {exc}")
    else:
        raise RuntimeError("expected a ValueError for a length mismatch")

    elapsed = time.perf_counter() - t0
    print()
    print(f"done in {elapsed:.3f}s (real computation, no sleep).")


if __name__ == "__main__":
    main()
