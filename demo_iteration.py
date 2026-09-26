"""Deterministic demonstration of partially aggregated observations."""

from __future__ import annotations

import time

from himeb import (
    analyze,
    fit_alpha_partial,
    log_evidence_partial,
    partial_observation,
)


def _observations():
    return [
        partial_observation([((0, 1), 3), ((2,), 0)]),
        partial_observation([((0,), 0), ((1, 2), 3)]),
        partial_observation([((0, 2), 4), ((1,), 1)]),
        partial_observation([((0, 1), 2), ((2,), 3)]),
    ]


def main() -> None:
    started = time.perf_counter()
    observations = _observations()

    print("=" * 72)
    print("PART 1. Shared alpha from mixed complete partitions (K=3)")
    print("=" * 72)
    result = fit_alpha_partial(observations)
    print(
        f"groups={result.n_groups} (each a different partition), informative="
        f"{result.n_informative_groups}, "
        f"alpha={result.fit.alpha:.6f}, status={result.fit.status}"
    )
    print(f"total log evidence = {result.fit.log_evidence:.6f}")
    for index, (observation, contribution) in enumerate(
        zip(observations, result.contributions)
    ):
        blocks = "; ".join(
            f"classes {block.classes}: n={block.count}"
            for block in observation.blocks
        )
        print(f"  group {index}: {blocks} -> log evidence {contribution:.6f}")

    print()
    print("=" * 72)
    print("PART 2. Constant-evidence cases are proved flat, not searched")
    print("=" * 72)
    constant_cases = [
        partial_observation([((0, 1, 2), 8)]),
        partial_observation([((0, 1), 1), ((2,), 0)]),
        partial_observation([((0,), 0), ((1,), 0), ((2,), 0)]),
    ]
    flat = fit_alpha_partial(constant_cases)
    print(
        f"status={flat.fit.status}, alpha={flat.fit.alpha}, "
        f"converged={flat.fit.converged}, budget_exhausted={flat.fit.budget_exhausted}"
    )
    for alpha in (1e-4, 1.0, 1e4):
        print(
            f"  evidence at alpha={alpha:8.4g}: "
            f"{log_evidence_partial(alpha, constant_cases):.12f}"
        )

    print()
    print("=" * 72)
    print("PART 3. Original complete-count and posterior API remains intact")
    print("=" * 72)
    complete = analyze([(4, 0, 1), (0, 3, 0), (2, 2, 1)])
    print(
        f"complete-count fit alpha={complete.fit.alpha:.6f}, "
        f"status={complete.fit.status}"
    )
    print(
        "posterior group 0 mean=",
        tuple(round(value, 6) for value in complete.posteriors[0].mean),
    )

    print()
    print("=" * 72)
    print("PART 4. A partition rejection that actually fires")
    print("=" * 72)
    try:
        partial_observation([((0, 1), 2), ((1, 2), 1)])
    except ValueError as exc:
        print(f"  ValueError raised as specified: {exc}")
    else:
        raise RuntimeError("overlapping partition blocks must be rejected")

    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed:.3f}s (real computation, no sleep).")


if __name__ == "__main__":
    main()
