"""Independent reference helpers shared by the test suite.

Nothing in this module imports himeb; expected values are produced from
standalone derivations:

* beta-binomial marginal likelihood via binomial expansion + beta integrals
  (signed log-sum-exp accumulation for numerical stability); symmetric alpha
  and general asymmetric (a, b) forms;
* the same marginal via direct adaptive-Simpson quadrature on [0, 1];
* posterior Beta moments as beta-integral ratios;
* a standalone digamma implementation for stationary-point checks.
"""

from __future__ import annotations

import math


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def evidence_expansion_ab(x: int, n: int, a: float, b: float) -> float:
    """log p(X=x | a, b), K=2, general asymmetric Beta(a, b) prior.

    p(x) = C(n,x)/B(a,b) * sum_j (-1)^j C(n-x,j) B(x+j+a, b).
    Independent of the production implementation; used for checking both the
    symmetric path (a=b) and the per-class Dirichlet evidence.
    """
    if not 0 <= x <= n:
        raise ValueError("require 0 <= x <= n")
    m = n - x
    log_denom = log_beta(a, b)
    signed_log_terms: list[tuple[float, int]] = []
    for j in range(m + 1):
        log_term = (
            math.lgamma(m + 1)
            - math.lgamma(j + 1)
            - math.lgamma(m - j + 1)
            + log_beta(x + j + a, b)
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


def evidence_expansion(x: int, n: int, alpha: float) -> float:
    """log p(X=x | a), K=2, symmetric Beta(a, a) prior."""
    return evidence_expansion_ab(x, n, alpha, alpha)


def digamma(z: float) -> float:
    """Standalone psi(z): recurrence to a large argument then asymptotic series."""
    if z <= 0.0:
        raise ValueError("digamma requires z > 0")
    result = 0.0
    while z < 10.0:
        result -= 1.0 / z
        z += 1.0
    inv = 1.0 / z
    inv2 = inv * inv
    result += math.log(z) - 0.5 * inv
    result -= inv2 * (
        1.0 / 12.0
        - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0))
    )
    return result


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


def evidence_quadrature_ab(
    x: int, n: int, a: float, b: float, tol: float = 1e-12
) -> float:
    """Direct numerical integral of C(n,x) p^x(1-p)^(n-x) beta(p;a,b).

    Asymmetric generalization of :func:`evidence_quadrature`; independent of
    the expansion and of himeb. Reliable when the integrand is not a spike
    much narrower than the quadrature's resolution (a, b >= 1 and moderate
    counts); near-integer-alpha endpoint cases are handled specially.
    """
    if not 0 <= x <= n:
        raise ValueError("require 0 <= x <= n")
    log_norm = -log_beta(a, b)
    log_coef = math.log(math.comb(n, x))

    def integrand(p: float) -> float:
        if p <= 0.0:
            if x == 0 and a == 1.0:
                return math.exp(min(log_coef + log_norm, 700.0))
            return 0.0
        if p >= 1.0:
            if x == n and b == 1.0:
                return math.exp(min(log_coef + log_norm, 700.0))
            return 0.0
        log_val = (
            log_coef
            + (x + a - 1.0) * math.log(p)
            + (n - x + b - 1.0) * math.log1p(-p)
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


def evidence_quadrature(x: int, n: int, alpha: float, tol: float = 1e-12) -> float:
    """Direct numerical integral of C(n,x) p^x(1-p)^(n-x) beta(p;a,a)."""
    return evidence_quadrature_ab(x, n, alpha, alpha, tol)


def evidence_polya(x: int, n: int, a: float, b: float) -> float:
    """Beta-binomial marginal via the sequential Pólya-urn factorization.

    Condition on an observed sequence with x ones followed by n-x zeros; the
    beta-binomial is exchangeable, so every order has this probability::

        prod_{i<x} (a + i)/(a + b + i) * prod_{j<n-x} (b + j)/(a + b + x + j)

    Each factor is a positive fraction, so there is no alternating
    cancellation: this stays accurate in the large-`a`, small-`b` corners
    where the binomial expansion reference loses significance. Multiply by
    C(n, x) for the unordered count probability.
    """
    if not 0 <= x <= n:
        raise ValueError("require 0 <= x <= n")
    log_p = math.fsum(
        -math.log1p(b / (a + i)) for i in range(x)
    ) + math.fsum(
        math.log(b + j) - math.log(a + b + x + j) for j in range(n - x)
    )
    return math.log(math.comb(n, x)) + log_p


def evidence_polya_k(
    counts: Sequence[int], alphas: Sequence[float]
) -> float:
    """General-K Dirichlet-multinomial marginal via sequential Pólya draws.

    Exact for the unordered count (multinomial coefficient times the
    probability of one fixed ordering). Every factor is a positive fraction,
    so no alternating cancellation at any alpha values. Independent of
    himeb; covers K >= 2 and class-specific alphas (including classes with
    zero counts, which simply add no factors but still enter the denominators
    through the total concentration).
    """
    n = sum(counts)
    a0 = math.fsum(alphas)
    log_p = 0.0
    drawn = 0
    for x_k, a_k in zip(counts, alphas):
        for i in range(x_k):
            # Denominator at this draw is a0 + drawn = (a_k+i) + other,
            # where other = total concentration of other classes plus the
            # items already drawn from earlier classes. Writing the fraction
            # as -log1p(other/(a_k+i)) keeps it exact even when almost all
            # mass is in this class.
            other = (a0 - a_k) + (drawn - i)
            log_p -= math.log1p(other / (a_k + i))
            drawn += 1
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in counts
    )
    return log_coef + log_p


def beta_posterior_moments_ab(
    x: int, n: int, a: float, b: float
) -> tuple[float, float, float]:
    """Independent (mean, second moment, variance) of p | x ~ Beta(x+a,n-x+b)."""
    aa = a + x
    bb = b + (n - x)
    s = aa + bb
    mean = aa / s
    second = aa * (aa + 1.0) / (s * (s + 1.0))
    return mean, second, second - mean * mean


def beta_posterior_moments(x: int, n: int, alpha: float) -> tuple[float, float, float]:
    """Independent (mean, second moment, variance) of p | x.

    Closed-form Beta(x+a, n-x+a) moments, written independently of the
    production module. For moderate alpha also cross-checked in the tests
    against lgamma beta-integral ratios.
    """
    return beta_posterior_moments_ab(x, n, alpha, alpha)


def beta_moments_via_integrals_ab(
    x: int, n: int, a: float, b: float
) -> tuple[float, float, float]:
    """Asymmetric version: B(aa+1,bb)/B(aa,bb) lgamma ratios."""
    aa = a + x
    bb = b + (n - x)
    log_b0 = log_beta(aa, bb)
    mean = math.exp(log_beta(aa + 1.0, bb) - log_b0)
    second = math.exp(log_beta(aa + 2.0, bb) - log_b0)
    return mean, second, second - mean * mean


def beta_moments_via_integrals(
    x: int, n: int, alpha: float
) -> tuple[float, float, float]:
    """Same moments via B(a+1,b)/B(a,b) lgamma ratios (accurate for a,b small)."""
    return beta_moments_via_integrals_ab(x, n, alpha, alpha)


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


def dense_alphas_scan_2d(
    count_rows: list[tuple[int, int]], n_points: int = 121
) -> tuple[tuple[float, float], float]:
    """Independent per-class maximizer for K=2: dense log-grid in both axes.

    Uses only the asymmetric beta-binomial expansion; never calls himeb.
    """
    lo, hi = math.log(1e-4), math.log(1e4)
    grid = [math.exp(lo + (hi - lo) * i / (n_points - 1))
            for i in range(n_points)]
    best_pt, best_v = (grid[0], grid[0]), -math.inf
    for a0 in grid:
        for a1 in grid:
            v = math.fsum(
                evidence_polya(c0, c0 + c1, a0, a1)
                for c0, c1 in count_rows
            )
            if v > best_v:
                best_pt, best_v = (a0, a1), v
    return best_pt, best_v
