"""Standalone K=2 reference used ONLY by the tests.

This module deliberately does not import the package under test.  Expected
values for the core are produced here from independent derivations:

1. The Bernoulli marginal is computed by expanding the polynomial factor
   theta**n1 * (1-theta)**n0 inside the beta integral into a finite binomial
   sum ("integral multinomial expansion")::

       L(n1, n0; a) = B(a+n1, a+n0) / B(a, a)
                    = sum_{j=0}^{n0} C(n0, j) (-1)^j B(a+n1+j, a) / B(a, a)

   Each beta RATIO is evaluated EXACTLY as a finite product

       B(a+m, a) / B(a, a)
           = Gamma(a+m) Gamma(a) Gamma(2a) /
             Gamma(2a+m) Gamma(a)^2
           = prod_{r=0}^{m-1} (a + r) / (2a + r)

   using decimal arithmetic at 50 significant digits.  This avoids both
   lgamma rounding at tiny alpha (where the float terms suffer catastrophic
   cancellation, e.g. 1 - 2 + 1 when alpha -> 0) and any shared implementation
   with the core's gamma-product formula.

2. Posterior moments use the closed Beta(a+n1, a+n0) formulas directly.

3. An independent maximizer (dense log-grid + golden-section refinement per
   detected basin) provides expected alpha values; it shares no optimizer
   code with the core.
"""

import decimal
import math

ALPHA_MIN = 1e-4
ALPHA_MAX = 1e4

_DEC = decimal.Decimal
decimal.getcontext().prec = 50


def log_beta(x, y):
    return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)


def _exact_beta_ratio(m, alpha):
    """prod_{r=0}^{m-1} (alpha+r)/(2 alpha+r) at 50-digit precision."""
    a = _DEC(repr(alpha))
    two_a = a * 2
    result = _DEC(1)
    for r in range(m):
        rr = _DEC(r)
        result *= (a + rr) / (two_a + rr)
    return result


def ref_log_evidence_group(n1, n0, alpha):
    """Log marginal for one K=2 count pair, via the binomial expansion.

    The alternating sum is carried out in high precision before taking log,
    so it stays accurate even when the float terms would cancel to ~1e-9.
    """
    if n1 == 0 and n0 == 0:
        return 0.0  # B(a, a) / B(a, a) == 1 exactly
    total = _DEC(0)
    for j in range(n0 + 1):
        coeff = _DEC(math.comb(n0, j))
        term = coeff * _exact_beta_ratio(n1 + j, alpha)
        if j % 2:
            total -= term
        else:
            total += term
    if total <= 0:  # pragma: no cover - the marginal is strictly positive
        raise ArithmeticError(f"non-positive marginal {total} for {(n1, n0)}, {alpha}")
    return float(total.ln())


def ref_log_evidence(rows, alpha):
    return math.fsum(
        ref_log_evidence_group(n1, n0, alpha) for n1, n0 in rows
    )


def ref_posterior(n1, n0, alpha):
    """Standalone Beta posterior moments for a K=2 group.

    Order matches the core's category order: (category 0, category 1),
    where category 0 has count n0 and category 1 has count n1.
    """
    a1 = alpha + n1
    a0 = alpha + n0
    s = a1 + a0
    mean0 = a0 / s
    mean1 = a1 / s
    var0 = mean0 * (1.0 - mean0) / (s + 1.0)
    var1 = mean1 * (1.0 - mean1) / (s + 1.0)
    return (mean0, mean1), (var0, var1)


def _golden_max(f, lo, hi, iterations=250):
    """Maximize f on [lo, hi] with golden-section (independent of core)."""
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    c = hi - (hi - lo) / gr
    d = lo + (hi - lo) / gr
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) / gr
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) / gr
            fd = f(d)
    x = 0.5 * (lo + hi)
    return x, f(x)


def ref_fit_alpha(rows, grid_points=801):
    """Independent global maximizer: grid over log alpha, golden per basin.

    Returns (alpha, status) with status in
    {'interior', 'lower_endpoint', 'upper_endpoint'}.
    """
    lmin, lmax = math.log(ALPHA_MIN), math.log(ALPHA_MAX)
    grid = [
        lmin + (lmax - lmin) * i / (grid_points - 1)
        for i in range(grid_points)
    ]
    vals = [ref_log_evidence(rows, math.exp(t)) for t in grid]

    basins = []
    if vals[0] >= vals[1]:
        basins.append((grid[0], grid[1]))
    if vals[-1] >= vals[-2]:
        basins.append((grid[-2], grid[-1]))
    for i in range(1, grid_points - 1):
        if vals[i] > vals[i - 1] and vals[i] >= vals[i + 1]:
            basins.append((grid[i - 1], grid[i + 1]))
    if not basins:
        i = max(range(grid_points), key=lambda h: vals[h])
        basins.append((grid[max(0, i - 1)], grid[min(grid_points - 1, i + 1)]))

    def f(t):
        return ref_log_evidence(rows, math.exp(t))

    best_t, best_v = None, -math.inf
    for lo, hi in basins:
        t, v = _golden_max(f, lo, hi)
        if v > best_v:
            best_t, best_v = t, v

    edge_tol = 1e-7
    if abs(best_t - lmin) <= edge_tol:
        return ALPHA_MIN, "lower_endpoint"
    if abs(best_t - lmax) <= edge_tol:
        return ALPHA_MAX, "upper_endpoint"
    return math.exp(best_t), "interior"
