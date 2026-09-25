"""Posterior-predictive event probabilities for future batch counts.

After fitting the shared symmetric Dirichlet concentration ``alpha`` (see
:mod:`himeb.core``), a group with observed counts x has posterior
``Dirichlet(alpha + x)``. The counts y of the next m draws therefore
follow a Dirichlet-multinomial with parameters ``a_k = alpha + x_k``;
the posterior mean is never plugged into an ordinary multinomial.

This module evaluates, in the log domain,

    P(lower_k <= y_k <= upper_k for all k | x, alpha)
    E[y_k | event, x, alpha]

with a per-class/cumulative-count dynamic program over the Dirichlet-
multinomial factorization weights. It never enumerates the K-dimensional
count simplex, so m up to ``MAX_PREDICTIVE_M`` = 100 and K up to 6 are
cheap and tiny probabilities do not underflow.
"""

from __future__ import annotations

import math
from typing import Sequence

from .core import (
    MAX_PREDICTIVE_M,
    PredictiveResult,
    _lgamma_add,
    _reject_bool_int,
    _validate_alpha,
    _validate_groups,
)


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) with -inf support."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _restricted_log_dp(
    log_w: Sequence[Sequence[float]],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
) -> tuple[float, ...]:
    """Log total unnormalized DM weight of count vectors inside bounds.

    ``log_w[k][c]`` is the log of the per-class factor
    ``Gamma(a_k + c) / (Gamma(a_k) * c!)``; the Dirichlet-multinomial
    probability of a count vector y is
    ``m! * Gamma(A) / Gamma(A + m) * prod_k w_k(y_k)`` with
    ``A = sum_k a_k``, so the shared normalization cancels in the ratio
    ``P(event) = (weight inside bounds) / (weight over all y)``.

    Dynamic programming over classes and cumulative counts in the log
    domain: ``dp[c]`` after class k is the log total weight of partial
    vectors (y_0..y_k) summing to c. This is a K-fold restricted
    convolution and never enumerates the K-dimensional count simplex.
    """
    m = len(log_w[0]) - 1
    dp = [-math.inf] * (m + 1)
    lo0, hi0 = lower[0], upper[0]
    for c in range(lo0, hi0 + 1):
        dp[c] = log_w[0][c]
    for k in range(1, len(log_w)):
        lok, hik = lower[k], upper[k]
        ndp = [-math.inf] * (m + 1)
        wk = log_w[k]
        for c in range(m + 1):
            base = dp[c]
            if base == -math.inf:
                continue
            hi_c = hik if c + hik <= m else m - c
            for y in range(lok, hi_c + 1):
                ndp[c + y] = _log_add(ndp[c + y], base + wk[y])
        dp = ndp
    return tuple(dp)


def posterior_predictive(
    counts: Sequence[int],
    alpha: object,
    m: object,
    bounds: Sequence[tuple[int, int]],
) -> PredictiveResult:
    """Posterior-predictive probability of per-class count bounds.

    Given this group observed counts and the *already fitted* common
    concentration ``alpha``, the posterior is ``Dirichlet(alpha + counts)``
    and the next ``m`` draws follow a Dirichlet-multinomial with
    parameters ``a_k = alpha + counts_k`` (the posterior mean is NOT
    plugged into an ordinary multinomial). Returns the probability that
    every class count ``y_k`` falls in the closed interval
    ``[lower_k, upper_k]``, plus the per-class expected counts
    conditional on that event.

    All results are conditional on the supplied fitted ``alpha``; this
    function performs no refitting and adds no hyperprior over alpha.

    * ``0 <= m <= MAX_PREDICTIVE_M``; ``bounds`` gives one
      ``(lower_k, upper_k)`` pair per class with
      ``0 <= lower_k <= upper_k <= m``.
    * If ``sum(lower) > m`` or ``sum(upper) < m`` the bounds cannot hold
      any count vector of total m, so the event is empty and zero is
      returned immediately (no DP run).
    * An empty event has ``probability == 0.0``,
      ``log_probability == -inf`` and all-NaN ``conditional_mean``: a
      mean conditional on a zero-probability event is undefined.
    * ``m == 0`` is handled directly: the only count vector is all
      zeros, so the probability is 1.0 iff every ``lower_k == 0``.

    The core computation is a per-class/cumulative-count dynamic
    program over the DM factorization weights in the log domain (no
    enumeration of the K-dimensional count simplex, no underflow for
    tiny probabilities).
    """
    a = _validate_alpha(alpha)
    rows, k = _validate_groups([counts])
    row = rows[0]
    m_val = _reject_bool_int("m", m, 0)
    if m_val > MAX_PREDICTIVE_M:
        raise ValueError(
            f"m must be <= MAX_PREDICTIVE_M={MAX_PREDICTIVE_M}, got {m_val}"
        )

    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
        raise TypeError(
            "bounds must be a sequence of (lower, upper) pairs, got "
            f"{type(bounds).__name__}"
        )
    if len(bounds) != k:
        raise ValueError(
            f"bounds has {len(bounds)} pairs, expected one per class (K={k})"
        )
    lower: list[int] = []
    upper: list[int] = []
    for j, pair in enumerate(bounds):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
            raise TypeError(
                f"bounds[{j}] must be a (lower, upper) pair, got "
                f"{type(pair).__name__}"
            )
        if len(pair) != 2:
            raise ValueError(
                f"bounds[{j}] must have exactly 2 elements, got {len(pair)}"
            )
        lo = _reject_bool_int(f"bounds[{j}][0]", pair[0], 0)
        hi = _reject_bool_int(f"bounds[{j}][1]", pair[1], 0)
        if lo > hi:
            raise ValueError(f"bounds[{j}]: lower {lo} exceeds upper {hi}")
        if hi > m_val:
            raise ValueError(f"bounds[{j}]: upper {hi} exceeds m={m_val}")
        lower.append(lo)
        upper.append(hi)
    lower_t = tuple(lower)
    upper_t = tuple(upper)
    params = tuple(a + c for c in row)

    def _empty_result() -> PredictiveResult:
        return PredictiveResult(
            m=m_val,
            alpha=a,
            posterior_params=params,
            lower=lower_t,
            upper=upper_t,
            log_probability=-math.inf,
            probability=0.0,
            conditional_mean=tuple(math.nan for _ in range(k)),
            is_empty_event=True,
        )

    if m_val == 0:
        # The only possible count vector is all zeros; the event holds iff
        # every lower bound is 0 (upper bounds are then 0 too since
        # upper <= m = 0).
        if all(lo == 0 for lo in lower_t):
            return PredictiveResult(
                m=0,
                alpha=a,
                posterior_params=params,
                lower=lower_t,
                upper=upper_t,
                log_probability=0.0,
                probability=1.0,
                conditional_mean=tuple(0.0 for _ in range(k)),
                is_empty_event=False,
            )
        return _empty_result()

    # Necessary and sufficient feasibility check for the bounded simplex:
    # some y with sum m exists inside the box iff sum(lower)<=m<=sum(upper).
    if sum(lower_t) > m_val or sum(upper_t) < m_val:
        return _empty_result()

    # Per-class factor weights w_k(c) = Gamma(a_k+c)/(Gamma(a_k)*c!), in
    # the log domain; lgamma(c + 1) is log(c!).
    log_w: list[list[float]] = []
    for a_k in params:
        wk = [0.0] * (m_val + 1)
        for c in range(m_val + 1):
            wk[c] = _lgamma_add(a_k, c) - math.lgamma(c + 1)
        log_w.append(wk)

    # Forward and suffix restricted convolutions; forward[j][c] covers
    # classes 0..j, suffix[j][c] covers classes j..K-1.
    forward: list[tuple[float, ...]] = []
    for j in range(k):
        forward.append(
            _restricted_log_dp(
                log_w[: j + 1], lower_t[: j + 1], upper_t[: j + 1]
            )
        )
    suffix: list[tuple[float, ...]] = [()] * k
    for j in range(k - 1, -1, -1):
        suffix[j] = _restricted_log_dp(log_w[j:], lower_t[j:], upper_t[j:])

    log_event = forward[k - 1][m_val]
    if log_event == -math.inf:
        return _empty_result()

    # log Z = log Gamma(A + m) - log Gamma(A) - log(m!), A = sum a_k:
    # the full DM distribution sums to 1, so the total weight is Z.
    total_a = math.fsum(params)
    log_z = _lgamma_add(total_a, m_val) - math.lgamma(m_val + 1)
    log_prob = log_event - log_z
    # Clamp tiny positive overshoot from float rounding; a probability
    # cannot exceed 1.
    if 0.0 < log_prob < 1e-12:
        log_prob = 0.0

    # Conditional means via leave-one-class-out recombination:
    #   E[y_k * 1_event] = sum_c c * w_k(c) * F_<k[.] * S_>k[.]
    # evaluated at total m, all in the log domain.
    means: list[float] = []
    for j in range(k):
        log_num = -math.inf
        wj = log_w[j]
        for c in range(lower_t[j], upper_t[j] + 1):
            if c == 0:
                continue  # c * w contributes zero
            rest = m_val - c
            if rest < 0:
                break
            log_cross = -math.inf
            if j == 0:
                log_cross = suffix[1][rest] if k > 1 else 0.0
            elif j == k - 1:
                log_cross = forward[k - 2][rest]
            else:
                fwd = forward[j - 1]
                sfx = suffix[j + 1]
                for s in range(rest + 1):
                    if fwd[s] == -math.inf or sfx[rest - s] == -math.inf:
                        continue
                    log_cross = _log_add(log_cross, fwd[s] + sfx[rest - s])
            if log_cross == -math.inf:
                continue
            term = math.log(c) + wj[c] + log_cross
            log_num = _log_add(log_num, term)
        if log_num == -math.inf:
            # Class j is pinned to 0 on the whole event.
            means.append(0.0)
        else:
            means.append(math.exp(log_num - log_event))

    return PredictiveResult(
        m=m_val,
        alpha=a,
        posterior_params=params,
        lower=lower_t,
        upper=upper_t,
        log_probability=log_prob,
        probability=math.exp(log_prob),
        conditional_mean=tuple(means),
        is_empty_event=False,
    )
