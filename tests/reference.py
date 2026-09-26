"""Independent reference helpers shared by the test suite.

Nothing in this module imports himeb; expected values are produced from
standalone derivations:

* beta-binomial marginal likelihood via binomial expansion + beta integrals
  (signed log-sum-exp accumulation for numerical stability);
* the same marginal via direct adaptive-Simpson quadrature on [0, 1];
* posterior Beta moments as beta-integral ratios.
"""

from __future__ import annotations

import math


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def evidence_expansion(x: int, n: int, alpha: float) -> float:
    """log p(X=x | a), K=2, expanding (1-p)**(n-x) term by term.

    p(x) = C(n,x)/B(a,a) * sum_j (-1)^j C(n-x,j) B(x+j+a, a).
    """
    if not 0 <= x <= n:
        raise ValueError("require 0 <= x <= n")
    m = n - x
    log_denom = log_beta(alpha, alpha)
    signed_log_terms: list[tuple[float, int]] = []
    for j in range(m + 1):
        log_term = (
            math.lgamma(m + 1)
            - math.lgamma(j + 1)
            - math.lgamma(m - j + 1)
            + log_beta(x + j + alpha, alpha)
            - log_denom
        )
        signed_log_terms.append((log_term, 1 if j % 2 == 0 else -1))
    scale = max(lt for lt, _ in signed_log_terms)
    total = (
        math.fsum(
            sign * math.exp(lt - scale) for lt, sign in signed_log_terms
        )
        * math.exp(scale)
    )
    return math.log(math.comb(n, x)) + math.log(total)


def _adaptive_simpson(f, a, b, fa, fb, fm, whole, tol, depth):
    m = 0.5 * (a + b)
    lm, rm = 0.5 * (a + m), 0.5 * (m + b)
    flm, frm = f(lm), f(rm)
    left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
    right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
    delta = left + right - whole
    if depth <= 0 or abs(delta) <= 15.0 * tol:
        return left + right + delta / 15.0
    return _adaptive_simpson(
        f, a, m, fa, fm, flm, left, tol * 0.5, depth - 1
    ) + _adaptive_simpson(
        f, m, b, fm, fb, frm, right, tol * 0.5, depth - 1
    )


def evidence_quadrature(x: int, n: int, alpha: float, tol: float = 1e-12) -> float:
    """Direct numerical integral of C(n,x) p^x(1-p)^(n-x) beta(p;a,a)."""
    if not 0 <= x <= n:
        raise ValueError("require 0 <= x <= n")
    log_norm = -log_beta(alpha, alpha)
    log_coef = math.log(math.comb(n, x))

    def integrand(p: float) -> float:
        if p <= 0.0:
            if x == 0 and alpha == 1.0:
                return math.exp(min(log_coef + log_norm, 700.0))
            return 0.0
        if p >= 1.0:
            if x == n and alpha == 1.0:
                return math.exp(min(log_coef + log_norm, 700.0))
            return 0.0
        log_val = (
            log_coef
            + (x + alpha - 1.0) * math.log(p)
            + (n - x + alpha - 1.0) * math.log1p(-p)
            + log_norm
        )
        if log_val < -745.0:
            return 0.0
        return math.exp(min(log_val, 700.0))

    fm = integrand(0.5)
    whole = 1.0 / 6.0 * (4.0 * fm)
    integral = _adaptive_simpson(
        integrand, 0.0, 1.0, 0.0, 0.0, fm, whole, tol, depth=30
    )
    return math.log(integral)


def beta_posterior_moments(x: int, n: int, alpha: float) -> tuple[float, float, float]:
    """Independent (mean, second moment, variance) of p | x.

    Closed-form Beta(x+a, n-x+a) moments, written independently of the
    production module. For moderate alpha also cross-checked in the tests
    against lgamma beta-integral ratios.
    """
    a = alpha + x
    b = alpha + (n - x)
    s = a + b
    mean = a / s
    second = a * (a + 1.0) / (s * (s + 1.0))
    return mean, second, second - mean * mean


def beta_moments_via_integrals(
    x: int, n: int, alpha: float
) -> tuple[float, float, float]:
    """Same moments via B(a+1,b)/B(a,b) lgamma ratios (accurate for a,b small)."""
    a = alpha + x
    b = alpha + (n - x)
    log_b0 = log_beta(a, b)
    mean = math.exp(log_beta(a + 1.0, b) - log_b0)
    second = math.exp(log_beta(a + 2.0, b) - log_b0)
    return mean, second, second - mean * mean


def dense_alpha_scan(
    count_rows: list[tuple[int, int]], n_points: int = 4001
) -> tuple[float, float]:
    """Independent maximizer: dense log-spaced scan using the expansion only.

    Input rows are count pairs (x0, x1); returns (best_alpha, best_log_ev).
    """
    lo, hi = math.log(1e-4), math.log(1e4)
    best_t, best_v = lo, -math.inf
    for i in range(n_points):
        t = lo + (hi - lo) * i / (n_points - 1)
        a = math.exp(t)
        v = math.fsum(
            evidence_expansion(c0, c0 + c1, a) for c0, c1 in count_rows
        )
        if v > best_v:
            best_t, best_v = t, v
    return math.exp(best_t), best_v


# ---------------------------------------------------------------------------
# Independent reference for category-aggregated (partial) observations.
#
# A group reports counts y_s for the blocks of a partition of the K classes.
# The merged evidence is independently computed here by ENUMERATING EVERY
# FINE ALLOCATION of the y_s draws among the classes inside block s and
# summing the ordinary Dirichlet-multinomial probabilities of the fine count
# vectors. This is a genuinely different route (a combinatorial sum) from
# the production additivity reduction to the merged Dirichlet-multinomial;
# it is intended for small totals only.
# ---------------------------------------------------------------------------


def _compositions(total: int, parts: int):
    """Yield every length-``parts`` tuple of non-negative ints summing total."""
    if parts == 1:
        yield (total,)
        return
    for head in range(total + 1):
        for tail in _compositions(total - head, parts - 1):
            yield (head,) + tail


def _fine_log_evidence(x, k: int, alpha: float) -> float:
    """Ordinary Dirichlet-multinomial log evidence of a fine count vector."""
    n = sum(x)
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in x
    )
    return (
        log_coef
        + math.lgamma(k * alpha)
        - math.lgamma(n + k * alpha)
        + math.fsum(
            math.lgamma(c + alpha) - math.lgamma(alpha) for c in x
        )
    )


def merged_evidence_by_enumeration(k, blocks, counts, alpha: float) -> float:
    """Merged log evidence via an exhaustive sum over fine allocations.

    ``blocks`` is a partition of ``range(k)`` (each block a class-index tuple)
    and ``counts`` the reported count per block. Probabilities are summed
    with a log-sum-exp accumulator (all terms are positive).
    """
    allocations_per_block = [
        list(_compositions(count, len(block)))
        for block, count in zip(blocks, counts)
    ]

    def expand(block_index: int, fine):
        if block_index == len(blocks):
            x = [0] * k
            for block, alloc in zip(blocks, fine):
                for class_index, c in zip(block, alloc):
                    x[class_index] = c
            yield tuple(x)
            return
        for alloc in allocations_per_block[block_index]:
            yield from expand(block_index + 1, fine + [alloc])

    terms = [_fine_log_evidence(x, k, alpha) for x in expand(0, [])]
    scale = max(terms)
    return scale + math.log(
        math.fsum(math.exp(term - scale) for term in terms)
    )
