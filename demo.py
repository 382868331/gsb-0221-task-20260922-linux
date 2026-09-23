"""Fixed-input demonstration for dirichlet_eb.

Runs in well under 8 seconds on the target machine.  All numbers are computed
for real (no sleep, no precomputed output): one normal empirical-Bayes fit
that prints posterior shrinkage, followed by rejection/error boundaries that
are actually triggered.

Run:  python demo.py
"""

import time

from dirichlet_eb import (
    ALPHA_MAX,
    ALPHA_MIN,
    IdentifiabilityError,
    InputValidationError,
    SearchBudgetExhaustedError,
    fit,
    log_evidence,
)

SEPARATOR = "-" * 72


def fmt_vec(vec, digits=4):
    return "(" + ", ".join(f"{v:.{digits}f}" for v in vec) + ")"


def main():
    # ------------------------------------------------------------------
    # Part 1: a normal result, with deterministic in-file data.
    # K = 3 categories; six groups including one deliberately empty group.
    # ------------------------------------------------------------------
    k = 3
    counts = [
        (8, 1, 1),   # strongly category 0
        (1, 8, 1),   # strongly category 1
        (1, 1, 8),   # strongly category 2
        (4, 4, 2),   # mildly balanced
        (6, 0, 0),   # single-category extreme
        (0, 0, 0),   # empty group: prior only, excluded from evidence
    ]

    print(SEPARATOR)
    print("Part 1: empirical-Bayes fit on fixed K=3 data")
    print(SEPARATOR)
    print(f"groups={len(counts)} (last one empty), categories K={k}")
    print(f"alpha search interval: [{ALPHA_MIN:g}, {ALPHA_MAX:g}] on log alpha")

    t0 = time.perf_counter()
    result = fit(counts, k=k)
    elapsed = time.perf_counter() - t0

    print(f"fit time: {elapsed*1000:.1f} ms")
    print(
        f"alpha_hat = {result.alpha:.6f}   status = {result.status}   "
        f"log evidence = {result.log_evidence:.6f}"
    )
    print(
        f"grid points = {result.grid_points}, candidate basins = "
        f"{result.candidate_count}, local-search iterations = {result.iterations}"
    )
    print()
    print("Shrinkage of each group's category probabilities:")
    print("  raw proportion n_k/N  ->  posterior mean (alpha_hat + n_k)/(K*alpha_hat + N)")
    for g, post in enumerate(result.groups):
        if post.is_empty:
            raw = tuple(float("nan") for _ in range(k))
            print(f"  group {g} {post.counts} EMPTY")
            print(f"      predictive (prior only) = {fmt_vec(post.predictive)}")
        else:
            raw = tuple(c / post.total for c in post.counts)
            print(f"  group {g} {post.counts} N={post.total}")
            print(f"      raw     = {fmt_vec(raw)}")
            print(f"      postmean= {fmt_vec(post.mean)}   variance={fmt_vec(post.variance)}")
        print(f"      predicted next category = {post.predicted_category}")

    # The pooled estimate shrinks raw proportions toward the uniform prior;
    # the empty group is exactly the prior and did not move alpha.
    nonempty = [r for r in counts if sum(r) > 0]
    print()
    print("Sanity checks:")
    print(f"  empty groups excluded from evidence: "
          f"{result.nonempty_group_count} non-empty of {result.group_count}")
    direct = log_evidence(result.alpha, nonempty, k=k)
    print(f"  log_evidence(alpha_hat, non-empty groups) = {direct:.6f} "
          f"(matches fit: {abs(direct - result.log_evidence) < 1e-11})")
    extreme = result.groups[4]
    print(
        f"  extreme group (6,0,0) raw p0=1.0000 shrunk to {extreme.mean[0]:.4f}; "
        f"unseen categories keep {extreme.mean[1]:.4f} (no hard zero)"
    )

    # ------------------------------------------------------------------
    # Part 2: rejection / exception boundaries, actually triggered.
    # ------------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("Part 2: error boundaries (each one is really raised below)")
    print(SEPARATOR)

    # (a) non-finite numeric input is rejected, not silently corrected.
    try:
        log_evidence(float("nan"), [(1, 1)], k=2)
    except ValueError as exc:
        print(f"[rejected] log_evidence(NaN, ...): {type(exc).__name__}: {exc}")

    # (b) bool is rejected where an integer parameter is required.
    try:
        fit([(1, 1)], k=True)
    except TypeError as exc:
        print(f"[rejected] fit(..., k=True): {type(exc).__name__}: {exc}")

    # (c) provable non-identifiability is distinct from a search failure:
    #     every group has at most one observation, evidence == 1/K constant.
    try:
        fit([(1, 0), (0, 1), (0, 0)], k=2)
    except IdentifiabilityError as exc:
        print(f"[non-identifiable] single-draw groups: {type(exc).__name__}")
        print(f"    {exc}")

    # (c2) budget exhaustion is a resource failure, NOT a mathematical
    #      "no solution": one golden-section iteration cannot reach tol 1e-12.
    try:
        fit([(2, 0), (0, 2), (1, 1)], k=2, tolerance=1e-12, max_iterations=1)
    except SearchBudgetExhaustedError as exc:
        print(f"[budget exhausted] max_iterations=1 at tol=1e-12: "
              f"{type(exc).__name__}")
        print(f"    {exc}")

    # (d) interface size limit.
    try:
        fit([(0, 0, 0)] * 41, k=3)
    except InputValidationError as exc:
        print(f"[rejected] 41 groups (> limit 40): {type(exc).__name__}: {exc}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
