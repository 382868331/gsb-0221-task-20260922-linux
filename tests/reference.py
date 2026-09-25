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
# Independent Dirichlet-multinomial event reference: full enumeration of
# the small-m count simplex. This never imports himeb; each count vector has
# the direct finite product probability
#   C(m; y) * B(a+y)/B(a) = m! Gamma(A)/Gamma(A+m)
#                            * prod_k Gamma(a_k+y_k)/(Gamma(a_k) y_k!)
# It is tractable only for small m (used with m <= 8 in the tests) and is
# deliberately the naive K-dimensional enumeration the production code
# must not rely on.
# ---------------------------------------------------------------------------


def dm_event_enumeration(params, m, bounds):
    """Brute-force (probability, conditional_means) of a bounded event.

    params = posterior Dirichlet parameters a_k > 0; bounds = list of
    (lower_k, upper_k). Means are None when the event probability is 0.
    """
    k = len(params)
    total_a = sum(params)
    log_norm = math.lgamma(m + 1) + math.lgamma(total_a) - math.lgamma(
        total_a + m
    )
    log_p_a = [math.lgamma(a) for a in params]
    prob = 0.0
    weighted = [0.0] * k

    def recurse(j, remaining, log_w, chosen):
        nonlocal prob
        if j == k - 1:
            y = chosen + [remaining]
            if not (bounds[j][0] <= remaining <= bounds[j][1]):
                return
            lp = log_w + math.lgamma(params[j] + remaining)
            lp -= log_p_a[j] + math.lgamma(remaining + 1)
            p = math.exp(log_norm + lp)
            prob += p
            for t in range(k):
                weighted[t] += p * y[t]
            return
        lo = max(
            bounds[j][0],
            remaining - sum(bounds[t][1] for t in range(j + 1, k)),
        )
        hi = min(
            bounds[j][1],
            remaining - sum(bounds[t][0] for t in range(j + 1, k)),
        )
        for c in range(lo, hi + 1):
            nxt = (
                log_w
                + math.lgamma(params[j] + c)
                - log_p_a[j]
                - math.lgamma(c + 1)
            )
            recurse(j + 1, remaining - c, nxt, chosen + [c])

    lo0 = max(bounds[0][0], m - sum(bounds[t][1] for t in range(1, k)))
    hi0 = min(bounds[0][1], m - sum(bounds[t][0] for t in range(1, k)))
    for c in range(lo0, hi0 + 1):
        start = math.lgamma(params[0] + c) - log_p_a[0] - math.lgamma(c + 1)
        recurse(1, m - c, start, [c])
    if prob == 0.0:
        return 0.0, None
    return prob, tuple(w / prob for w in weighted)
