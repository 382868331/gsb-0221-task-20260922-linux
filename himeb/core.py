"""Hierarchical multinomial counts: deterministic empirical Bayes.

A reusable, standard-library-only module for the following small-sample
hierarchical model::

    theta_g ~ Dirichlet(alpha, alpha, ..., alpha)   # K classes, symmetric
    x_g     ~ Multinomial(n_g, theta_g)              # g = 1..G groups

A single concentration ``alpha`` is shared by every group and estimated by
maximizing the summed (Dirichlet-)multinomial marginal likelihood on the
explicit interval ``[ALPHA_MIN, ALPHA_MAX]``, optimizing over ``log(alpha)``.

Public API
----------
log_evidence(alpha, groups) -> float
fit_alpha(groups, *, grid_points=201, max_iter=100) -> FitResult
posterior_group(counts, alpha) -> GroupPosterior
analyze(groups, *, grid_points=201, max_iter=100) -> EBResult
predictive_event(counts, alpha, m, bounds) -> PredictiveEventResult

Contract limits: 2 <= K <= 6, 1 <= G <= 40, non-negative integer counts.
Numeric inputs reject NaN/Infinity; integer parameters reject ``bool``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Public constants: explicit search interval and hard contract limits.
# ---------------------------------------------------------------------------

ALPHA_MIN = 1e-4
ALPHA_MAX = 1e4
MAX_K = 6
MAX_GROUPS = 40
MAX_PREDICT_M = 100

DEFAULT_GRID_POINTS = 201
DEFAULT_MAX_ITER = 100

# Golden-section stopping width in t = log(alpha) units.
T_TOL = 1e-12

# While the added integer is at most this large, log Gamma(x+m)/Gamma(x) is
# evaluated as an exact sum of logs instead of a difference of lgamma values.
_SUM_SWITCH = 128

# Counts above this cannot be represented exactly as float64 any more; the
# evidence works in float, so such inputs are rejected rather than rounded.
_MAX_EXACT_COUNT = 2**53

# Optimizer statuses.
STATUS_INTERIOR = "interior"
STATUS_LOWER_ENDPOINT = "lower_endpoint"
STATUS_UPPER_ENDPOINT = "upper_endpoint"
STATUS_FLAT = "flat"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _reject_bool_int(name: str, value: object, minimum: int) -> int:
    """Validate a positive-bounded integer parameter; bool is rejected."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool (bool is not accepted)")
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be int, got {type(value).__name__}: {value!r}"
        )
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_alpha(alpha: object) -> float:
    if isinstance(alpha, bool):
        raise TypeError("alpha must be int or float, got bool")
    if not isinstance(alpha, (int, float)):
        raise TypeError(
            f"alpha must be int or float, got {type(alpha).__name__}"
        )
    value = float(alpha)
    if not math.isfinite(value):
        raise ValueError(
            f"alpha must be finite and not NaN/Infinity, got {alpha!r}"
        )
    if value <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    return value


def _validate_groups(
    groups: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Validate a batch of count vectors.

    All checking happens before any result is produced, so a rejected batch
    never causes a partial/stateful change (all public operations are pure).
    """
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError(
            "groups must be a sequence of count sequences, got "
            f"{type(groups).__name__}"
        )
    g_count = len(groups)
    if g_count == 0:
        raise ValueError("groups must contain at least one group")
    if g_count > MAX_GROUPS:
        raise ValueError(
            f"number of groups {g_count} exceeds limit MAX_GROUPS={MAX_GROUPS}"
        )

    k: int | None = None
    rows: list[tuple[int, ...]] = []
    for i, group in enumerate(groups):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise TypeError(
                f"group {i} must be a sequence of integer counts, got "
                f"{type(group).__name__}"
            )
        if k is None:
            k = len(group)
            if not 2 <= k <= MAX_K:
                raise ValueError(
                    f"group 0 has K={k}; require 2 <= K <= {MAX_K}"
                )
        elif len(group) != k:
            raise ValueError(
                f"group {i} has length {len(group)}, expected K={k}; "
                "all groups must share the same number of classes"
            )

        row: list[int] = []
        for j, count in enumerate(group):
            if isinstance(count, bool):
                raise TypeError(
                    f"group {i} class {j}: count must be int, got bool "
                    "(bool is not accepted as a count)"
                )
            if not isinstance(count, int):
                raise TypeError(
                    f"group {i} class {j}: count must be int, got "
                    f"{type(count).__name__}: {count!r}"
                )
            if count < 0:
                raise ValueError(
                    f"group {i} class {j}: count must be >= 0, got {count}"
                )
            if count > _MAX_EXACT_COUNT:
                raise ValueError(
                    f"group {i} class {j}: count {count} exceeds the supported "
                    f"exact-integer limit 2**53"
                )
            row.append(count)
        rows.append(tuple(row))

    assert k is not None
    return tuple(rows), k


# ---------------------------------------------------------------------------
# Log evidence
# ---------------------------------------------------------------------------


def _lgamma_add(x: float, m: int) -> float:
    """log(Gamma(x + m) / Gamma(x)) for x > 0 and non-negative integer m."""
    if m == 0:
        return 0.0
    if m <= _SUM_SWITCH:
        # Difference-free identity: keeps full accuracy when x is large and m
        # is small (matters at alpha = 1e4, where a plain lgamma difference
        # cancels two ~1e5 quantities).
        return math.fsum(math.log(x + j) for j in range(m))
    return math.lgamma(x + m) - math.lgamma(x)


def _log_evidence_row(alpha: float, k: int, row: tuple[int, ...]) -> float:
    n = sum(row)
    if n == 0:
        # Integral of the Dirichlet density: an empty draw has probability 1
        # for every alpha, so empty groups contribute exactly zero and carry
        # no information about alpha.
        return 0.0
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in row
    )
    # log multinomial coefficient
    # + lgamma(K a) - lgamma(n + K a)
    # + sum_k [lgamma(x_k + a) - lgamma(a)]
    return (
        log_coef
        - _lgamma_add(k * alpha, n)
        + math.fsum(_lgamma_add(alpha, c) for c in row)
    )


def log_evidence_one(alpha: object, counts: Sequence[int]) -> float:
    """Log marginal likelihood of one count vector.

    Accepts any finite positive ``alpha`` (the [1e-4, 1e4] interval binds the
    optimizer, not this function). An all-zero vector returns 0.0.
    """
    a = _validate_alpha(alpha)
    rows, k = _validate_groups([counts])
    return _log_evidence_row(a, k, rows[0])


def log_evidence(alpha: object, groups: Sequence[Sequence[int]]) -> float:
    """Summed log evidence over groups; empty groups contribute zero."""
    a = _validate_alpha(alpha)
    rows, k = _validate_groups(groups)
    return math.fsum(_log_evidence_row(a, k, row) for row in rows)


# ---------------------------------------------------------------------------
# Bracket-preserving 1-D maximizer on a finite t interval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Peak:
    t: float
    value: float
    kind: str  # interior | lower_endpoint | upper_endpoint
    bracket: tuple[float, float] | None
    converged: bool
    iterations: int


def _golden_max(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    max_iter: int,
) -> tuple[float, float, int, bool]:
    """Golden-section search for a maximum on [lo, hi].

    Returns (t*, f(t*), iterations_used, converged).  ``converged`` is False
    when the iteration budget is exhausted before ``T_TOL`` is reached; this
    is deliberately distinct from any "no solution" condition (a continuous
    function on this compact interval always attains a maximum).
    """
    ratio = (math.sqrt(5.0) - 1.0) / 2.0  # 1/phi ~ 0.618
    a, b = float(lo), float(hi)
    c = b - (b - a) * ratio
    d = a + (b - a) * ratio
    fc, fd = f(c), f(d)
    used = 0
    for used in range(1, max_iter + 1):
        if fc > fd:
            b = d
            d, fd = c, fc
            c = b - (b - a) * ratio
            fc = f(c)
        else:
            a = c
            c, fc = d, fd
            d = a + (b - a) * ratio
            fd = f(d)
        if b - a <= T_TOL:
            break
    if fc >= fd:
        return c, fc, used, (b - a) <= T_TOL
    return d, fd, used, (b - a) <= T_TOL


def _maximize_1d(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    grid_points: int,
    max_iter: int,
) -> tuple[_Peak, tuple[_Peak, ...]]:
    """Grid-locate every candidate peak, then refine each with golden section.

    The grid is exhaustive at its own resolution only: a mode narrower than
    one grid cell and peaking between grid nodes can be missed; callers see
    both the budget (``grid_points``, per-peak ``iterations``/``converged``)
    and every candidate bracket, so the search restriction is retained rather
    than hidden. Endpoints are always candidates.
    """
    _reject_bool_int("grid_points", grid_points, 3)
    _reject_bool_int("max_iter", max_iter, 1)
    if not math.isfinite(lo) or not math.isfinite(hi) or not hi > lo:
        raise ValueError("require finite lo < hi")

    step = (hi - lo) / (grid_points - 1)
    ts = [lo + step * i for i in range(grid_points)]
    vs = [f(t) for t in ts]

    peaks: list[_Peak] = [
        _Peak(ts[0], vs[0], STATUS_LOWER_ENDPOINT, None, True, 0),
        _Peak(ts[-1], vs[-1], STATUS_UPPER_ENDPOINT, None, True, 0),
    ]

    for i in range(1, grid_points - 1):
        if vs[i] >= vs[i - 1] and vs[i] >= vs[i + 1] and (
            vs[i] > vs[i - 1] or vs[i] > vs[i + 1]
        ):
            t_star, v_star, used, ok = _golden_max(
                f, ts[i - 1], ts[i + 1], max_iter
            )
            # Golden section on a grid cell cannot beat the grid value by a
            # meaningful amount for a flat node; guard against noise pushing
            # the refined value below the node it refines.
            if vs[i] > v_star:
                t_star, v_star = ts[i], vs[i]
            peaks.append(
                _Peak(
                    t_star,
                    v_star,
                    STATUS_INTERIOR,
                    (ts[i - 1], ts[i + 1]),
                    ok,
                    used,
                )
            )

    best = max(peaks, key=lambda p: (p.value, p.kind == STATUS_INTERIOR))
    return best, tuple(peaks)


# ---------------------------------------------------------------------------
# Public result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One local/endpoint maximum considered during the search."""

    alpha: float
    log_evidence: float
    kind: str  # interior | lower_endpoint | upper_endpoint
    bracket: tuple[float, float] | None  # in t = log(alpha); None at endpoints
    converged: bool
    iterations: int


@dataclass(frozen=True)
class FitResult:
    alpha: float
    log_evidence: float
    status: str  # interior | lower_endpoint | upper_endpoint | flat
    converged: bool
    grid_points: int
    max_iterations_used: int
    candidates: tuple[Candidate, ...]
    n_nonempty_groups: int

    @property
    def budget_exhausted(self) -> bool:
        """True iff some local refinement stopped for lack of iteration budget.

        Contrast ``status == 'flat'``, the *proved* absence of a unique
        maximizer: every group empty (evidence identically 0), or every
        non-empty group holding a single observation (evidence exactly
        constant in alpha).
        """
        return not self.converged and self.status != STATUS_FLAT


@dataclass(frozen=True)
class GroupPosterior:
    counts: tuple[int, ...]
    n: int
    is_empty: bool
    alpha: float
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    predictive: tuple[float, ...]
    predicted: int  # argmax of predictive; ties resolved to the lowest index


@dataclass(frozen=True)
class PredictiveEventResult:
    """Posterior-predictive mass of a box event and conditional means.

    The event is ``L_k <= Y_k <= U_k`` for every class k of a future batch
    of ``m`` draws, under the Dirichlet-multinomial with posterior
    parameters ``alpha + observed counts``. Results are conditional on the
    fitted alpha supplied by the caller; no posterior averaging over alpha
    is performed.

    ``conditional_mean`` is defined only when ``probability > 0``; it is
    ``None`` for a zero-probability event.
    """

    counts: tuple[int, ...]
    alpha: float
    m: int
    bounds: tuple[tuple[int, int], ...]
    probability: float
    log_probability: float
    conditional_mean: tuple[float, ...] | None


@dataclass(frozen=True)
class EBResult:
    k: int
    n_groups: int
    n_nonempty_groups: int
    fit: FitResult
    posteriors: tuple[GroupPosterior, ...]


# ---------------------------------------------------------------------------
# Fitting and posterior summaries
# ---------------------------------------------------------------------------


def maximize_on_interval(
    f: Callable[[float], float],
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> FitResult:
    """Maximize ``f(alpha)`` for alpha in ``[ALPHA_MIN, ALPHA_MAX]``.

    The search is over t = log(alpha): a uniform grid finds every grid-level
    candidate mode, each interior candidate is refined by a bracket-preserving
    golden-section search, and both endpoints are evaluated exactly. This is
    the reusable multimodal 1-D engine behind :func:`fit_alpha` and can take
    any finite-valued objective.

    ``converged=False`` / ``budget_exhausted=True`` means an interior
    refinement spent its whole iteration budget before reaching ``T_TOL``; a
    continuous function on this compact interval always attains a maximum, so
    this is a budget condition and never a proved "no solution". The search
    restriction is retained: modes narrower than one grid cell and peaking
    strictly between grid nodes can be missed, and every candidate keeps its
    log-alpha bracket for inspection.
    """
    gp = _reject_bool_int("grid_points", grid_points, 3)
    mi = _reject_bool_int("max_iter", max_iter, 1)

    def g(t: float) -> float:
        value = f(math.exp(t))
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError(
                "objective must return a finite float at every grid/interior "
                f"point, got {value!r} at alpha={math.exp(t):.8g}"
            )
        return value

    t_lo, t_hi = math.log(ALPHA_MIN), math.log(ALPHA_MAX)
    _, peaks = _maximize_1d(g, t_lo, t_hi, gp, mi)

    candidates: list[Candidate] = []
    for p in peaks:
        if p.kind == STATUS_LOWER_ENDPOINT:
            candidates.append(Candidate(ALPHA_MIN, p.value, p.kind, None, True, 0))
        elif p.kind == STATUS_UPPER_ENDPOINT:
            candidates.append(Candidate(ALPHA_MAX, p.value, p.kind, None, True, 0))
        else:
            candidates.append(
                Candidate(
                    math.exp(p.t), p.value, p.kind, p.bracket,
                    p.converged, p.iterations,
                )
            )

    winner = max(
        candidates,
        key=lambda c: (c.log_evidence, c.kind == STATUS_INTERIOR),
    )
    return FitResult(
        alpha=winner.alpha,
        log_evidence=winner.log_evidence,
        status=winner.kind,
        converged=all(c.converged for c in candidates),
        grid_points=gp,
        max_iterations_used=max(c.iterations for c in candidates),
        candidates=tuple(candidates),
        n_nonempty_groups=0,
    )


def fit_alpha(
    groups: Sequence[Sequence[int]],
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> FitResult:
    """Maximize total log evidence over alpha in [ALPHA_MIN, ALPHA_MAX].

    Uses :func:`maximize_on_interval` over t = log(alpha). Two degenerate but
    *proved* cases skip the search and return ``status='flat'`` with
    alpha = 1.0: every group empty (evidence identically 0), or every
    non-empty group holding a single observation (a lone draw is uniform over
    the K classes for every alpha, so evidence is the exact constant
    ``-G1 * log(K)``). Neither is a budget failure.
    """
    rows, k = _validate_groups(groups)
    gp = _reject_bool_int("grid_points", grid_points, 3)
    mi = _reject_bool_int("max_iter", max_iter, 1)

    nonempty = tuple(row for row in rows if sum(row) > 0)
    if not nonempty:
        # Evidence is identically zero on the whole positive half-line:
        # provably no unique maximizer. This is not a budget failure.
        flat_candidates = (
            Candidate(ALPHA_MIN, 0.0, STATUS_LOWER_ENDPOINT, None, True, 0),
            Candidate(ALPHA_MAX, 0.0, STATUS_UPPER_ENDPOINT, None, True, 0),
        )
        return FitResult(
            alpha=1.0,
            log_evidence=0.0,
            status=STATUS_FLAT,
            converged=True,
            grid_points=gp,
            max_iterations_used=0,
            candidates=flat_candidates,
            n_nonempty_groups=0,
        )

    if all(sum(row) == 1 for row in nonempty):
        # A single draw is uniformly distributed over the K classes under the
        # symmetric Dirichlet-multinomial for EVERY alpha:
        #     log p(x = e_j | alpha) = log(a) - log(K a) = -log K.
        # Hence the summed evidence is the exact constant -G1 * log K, again a
        # proved absence of a unique maximizer (not a budget failure). Empty
        # groups contribute 0 and are absent from G1.
        const_evidence = -len(nonempty) * math.log(k)
        flat_candidates = (
            Candidate(
                ALPHA_MIN, const_evidence, STATUS_LOWER_ENDPOINT, None, True, 0
            ),
            Candidate(
                ALPHA_MAX, const_evidence, STATUS_UPPER_ENDPOINT, None, True, 0
            ),
        )
        return FitResult(
            alpha=1.0,
            log_evidence=const_evidence,
            status=STATUS_FLAT,
            converged=True,
            grid_points=gp,
            max_iterations_used=0,
            candidates=flat_candidates,
            n_nonempty_groups=len(nonempty),
        )

    def evidence_alpha(a: float) -> float:
        return math.fsum(_log_evidence_row(a, k, row) for row in nonempty)

    result = maximize_on_interval(
        evidence_alpha, grid_points=gp, max_iter=mi
    )
    return FitResult(
        alpha=result.alpha,
        log_evidence=result.log_evidence,
        status=result.status,
        converged=result.converged,
        grid_points=gp,
        max_iterations_used=result.max_iterations_used,
        candidates=result.candidates,
        n_nonempty_groups=len(nonempty),
    )


def posterior_group(counts: Sequence[int], alpha: object) -> GroupPosterior:
    """Posterior summary for one count vector under the symmetric prior.

    Empty vectors return the prior itself (uniform mean, prior variance);
    they need no fitted alpha.
    """
    a = _validate_alpha(alpha)
    rows, k = _validate_groups([counts])
    row = rows[0]
    n = sum(row)
    total = k * a + n
    mean = tuple((a + c) / total for c in row)
    variance = tuple(
        (a + c) * (total - (a + c)) / (total * total * (total + 1.0))
        for c in row
    )
    # One-step-ahead multinomial predictive: equals posterior mean.
    predictive = mean
    predicted = 0
    for j in range(1, k):
        if predictive[j] > predictive[predicted]:
            predicted = j
    return GroupPosterior(
        counts=row,
        n=n,
        is_empty=(n == 0),
        alpha=a,
        mean=mean,
        variance=variance,
        predictive=predictive,
        predicted=predicted,
    )


def analyze(
    groups: Sequence[Sequence[int]],
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> EBResult:
    """Fit alpha and return aligned posterior summaries for every group."""
    rows, k = _validate_groups(groups)
    fit = fit_alpha(rows, grid_points=grid_points, max_iter=max_iter)
    posteriors = tuple(posterior_group(row, fit.alpha) for row in rows)
    return EBResult(
        k=k,
        n_groups=len(rows),
        n_nonempty_groups=fit.n_nonempty_groups,
        fit=fit,
        posteriors=posteriors,
    )


# ---------------------------------------------------------------------------
# Posterior-predictive box events (Dirichlet-multinomial)
# ---------------------------------------------------------------------------


_NEG_INF = float("-inf")


def _validate_predictive_inputs(
    counts: Sequence[int],
    alpha: object,
    m: object,
    bounds: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], int, float, tuple[tuple[int, int], ...]]:
    a = _validate_alpha(alpha)
    rows, k = _validate_groups([counts])
    row = rows[0]
    m_val = _reject_bool_int("m", m, 0)
    if m_val > MAX_PREDICT_M:
        raise ValueError(
            f"m must be <= {MAX_PREDICT_M}, got {m_val}"
        )
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
        raise TypeError(
            "bounds must be a sequence of (low, high) pairs, got "
            f"{type(bounds).__name__}"
        )
    if len(bounds) != k:
        raise ValueError(
            f"bounds has length {len(bounds)}, expected K={k} (one closed "
            "interval per class)"
        )
    checked: list[tuple[int, int]] = []
    for j, pair in enumerate(bounds):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
            raise TypeError(
                f"bounds[{j}] must be a (low, high) pair, got "
                f"{type(pair).__name__}"
            )
        if len(pair) != 2:
            raise ValueError(
                f"bounds[{j}] must have exactly 2 elements (low, high), "
                f"got length {len(pair)}"
            )
        low = _reject_bool_int(f"bounds[{j}][0]", pair[0], 0)
        high = _reject_bool_int(f"bounds[{j}][1]", pair[1], 0)
        if low > high:
            raise ValueError(
                f"bounds[{j}]: low ({low}) must be <= high ({high})"
            )
        checked.append((low, high))
    return row, k, a, tuple(checked)


def _forward_pass(
    log_weights: Sequence[Sequence[float]],
    m: int,
    ranges: Sequence[tuple[int, int]],
) -> tuple[list[list[float]], list[float]]:
    """Scaled log-domain forward DP over classes and cumulative counts.

    Layer k holds, for every reachable total t, the log partition over
    classes 0..k (unnormalized Dirichlet-multinomial factors). Each layer
    is shifted by its own finite maximum; the shifted arrays together with
    the per-layer maxima reconstruct every log quantity without underflow.
    Enumeration is over (class, cumulative count) pairs only: O(K m^2);
    no K-dimensional count vector is ever enumerated.
    """
    layers: list[list[float]] = []
    maxima: list[float] = []
    prev = [_NEG_INF] * (m + 1)
    prev[0] = 0.0
    for k, (low, high) in enumerate(ranges):
        raw = [_NEG_INF] * (m + 1)
        hi_k = min(high, m)
        for t in range(m + 1):
            lo_y = max(low, t - m)
            hi_y = min(hi_k, t)
            vals: list[float] = []
            if lo_y <= hi_y:
                for y in range(lo_y, hi_y + 1):
                    pt = prev[t - y]
                    if pt == _NEG_INF:
                        continue
                    vals.append(pt + log_weights[k][y])
            raw[t] = (
                _NEG_INF
                if not vals
                else max(vals)
                + math.log(math.fsum(math.exp(v - max(vals)) for v in vals))
            )
        finite = [v for v in raw if v != _NEG_INF]
        if not finite:
            # Whole layer unreachable: mark every remaining layer dead.
            layer_max = _NEG_INF
            shifted = [_NEG_INF] * (m + 1)
        else:
            layer_max = max(finite)
            shifted = [
                (_NEG_INF if v == _NEG_INF else v - layer_max) for v in raw
            ]
        layers.append(shifted)
        maxima.append(layer_max)
        prev = shifted
    return layers, maxima


def _backward_pass(
    log_weights: Sequence[Sequence[float]],
    m: int,
    ranges: Sequence[tuple[int, int]],
) -> tuple[list[list[float]], list[float]]:
    """Scaled log-domain backward DP (classes k..K-1 versus total t)."""
    k_count = len(ranges)
    layers: list[list[float]] = [
        [_NEG_INF] * (m + 1) for _ in range(k_count)
    ]
    maxima: list[float] = [_NEG_INF] * k_count
    nxt = [_NEG_INF] * (m + 1)
    nxt[0] = 0.0
    for k in range(k_count - 1, -1, -1):
        raw = [_NEG_INF] * (m + 1)
        low, high = ranges[k]
        hi_k = min(high, m)
        for t in range(m + 1):
            lo_y = max(low, 0)
            hi_y = min(hi_k, t)
            vals = []
            for y in range(lo_y, hi_y + 1):
                rest = nxt[t - y]
                if rest == _NEG_INF:
                    continue
                vals.append(rest + log_weights[k][y])
            raw[t] = (
                _NEG_INF
                if not vals
                else max(vals)
                + math.log(math.fsum(math.exp(v - max(vals)) for v in vals))
            )
        finite = [v for v in raw if v != _NEG_INF]
        if finite:
            layer_max = max(finite)
            layers[k] = [
                (_NEG_INF if v == _NEG_INF else v - layer_max) for v in raw
            ]
            maxima[k] = layer_max
        else:
            layers[k] = [_NEG_INF] * (m + 1)
            maxima[k] = _NEG_INF
        nxt = layers[k]
    return layers, maxima


def predictive_event(
    counts: Sequence[int],
    alpha: object,
    m: object,
    bounds: Sequence[Sequence[int]],
) -> PredictiveEventResult:
    """Posterior-predictive probability of per-class count box constraints.

    Given an observed group ``counts`` and the fitted (or any positive)
    ``alpha``, the posterior is ``Dirichlet(alpha + counts)``. For a future
    batch of ``m`` (``0 <= m <= 100``) draws with counts ``Y``, this
    computes ``P(for all k: low_k <= Y_k <= high_k)`` under the resulting
    Dirichlet-multinomial (Polya-urn) predictive, and the conditional
    expectation ``E[Y_k | event]`` for each class. The posterior mean is
    never substituted into an ordinary multinomial: over-dispersion from
    marginalizing theta is retained.

    The Dirichlet-multinomial factors as a product of per-class finite
    products ``Gamma(a_k + y_k) / (Gamma(a_k) y_k!)``; box probabilities
    and conditional means are assembled by forward/backward dynamic
    programming over classes and cumulative counts (O(K m^2), scaled log
    domain) rather than by enumerating K-dimensional count vectors.

    ``m == 0`` is handled directly. When the bounds cannot sum to m the
    probability is exactly zero and ``conditional_mean`` is ``None``
    (conditional means of a zero-probability event are undefined).
    Computation is conditional on the supplied alpha; no asymmetric prior,
    alpha uncertainty, or refitting is introduced.
    """
    row, k, a, checked = _validate_predictive_inputs(counts, alpha, m, bounds)
    m_val = m
    bounds_t = checked

    if m_val == 0:
        feasible = all(low <= 0 <= high for low, high in checked)
        if feasible:
            return PredictiveEventResult(
                counts=row,
                alpha=a,
                m=0,
                bounds=bounds_t,
                probability=1.0,
                log_probability=0.0,
                conditional_mean=tuple(0.0 for _ in range(k)),
            )
        return PredictiveEventResult(
            counts=row,
            alpha=a,
            m=0,
            bounds=bounds_t,
            probability=0.0,
            log_probability=_NEG_INF,
            conditional_mean=None,
        )

    # Cheap exact feasibility screen on the closed intervals: the class
    # counts sum to m, so disjoint total-sum ranges make the event empty.
    if sum(low for low, _ in checked) > m_val or (
        sum(high for _, high in checked) < m_val
    ):
        return PredictiveEventResult(
            counts=row,
            alpha=a,
            m=m_val,
            bounds=bounds_t,
            probability=0.0,
            log_probability=_NEG_INF,
            conditional_mean=None,
        )

    post_a = [a + c for c in row]
    log_weights = [
        [
            _lgamma_add(ak, y) - math.lgamma(y + 1)
            for y in range(m_val + 1)
        ]
        for ak in post_a
    ]

    full_ranges = [(0, m_val)] * k
    fwd_full, max_full = _forward_pass(log_weights, m_val, full_ranges)
    log_z_full = math.fsum(max_full) + fwd_full[-1][m_val]

    fwd_ev, max_ev = _forward_pass(log_weights, m_val, checked)
    shifted_final = fwd_ev[-1][m_val]
    if shifted_final == _NEG_INF:
        return PredictiveEventResult(
            counts=row,
            alpha=a,
            m=m_val,
            bounds=bounds_t,
            probability=0.0,
            log_probability=_NEG_INF,
            conditional_mean=None,
        )
    log_z_event = math.fsum(max_ev) + shifted_final
    log_prob = log_z_event - log_z_full
    prob = math.exp(log_prob)

    # Backward pass on the restricted support, then per-class conditional
    # distributions from prefix * class factor * suffix products.
    bwd_ev, bwd_max = _backward_pass(log_weights, m_val, checked)

    prefix_const = [0.0] * (k + 1)
    for idx in range(k):
        prefix_const[idx + 1] = prefix_const[idx] + max_ev[idx]
    suffix_const = [0.0] * (k + 1)
    for idx in range(k - 1, -1, -1):
        suffix_const[idx] = suffix_const[idx + 1] + bwd_max[idx]

    cond_means: list[float] = []
    for j in range(k):
        low, high = checked[j]
        terms: list[float] = []
        ys: list[int] = []
        for y in range(max(low, 0), min(high, m_val) + 1):
            # Join prefix (classes < j, total s) and suffix
            # (classes > j, total m-y-s); sum over s in log domain.
            joints: list[float] = []
            for s_total in range(0, m_val - y + 1):
                u_total = m_val - y - s_total
                if j == 0:
                    prefix = 0.0 if s_total == 0 else _NEG_INF
                    pre_const = 0.0
                else:
                    prefix = fwd_ev[j - 1][s_total]
                    pre_const = prefix_const[j]
                if prefix == _NEG_INF:
                    continue
                if j + 1 == k:
                    suffix = 0.0 if u_total == 0 else _NEG_INF
                    suf_const = 0.0
                else:
                    suffix = bwd_ev[j + 1][u_total]
                    suf_const = suffix_const[j + 1]
                if suffix == _NEG_INF:
                    continue
                joints.append(pre_const + prefix + suf_const + suffix)
            if not joints:
                continue
            top_j = max(joints)
            log_join = top_j + math.log(
                math.fsum(math.exp(v - top_j) for v in joints)
            )
            log_term = log_join + log_weights[j][y]
            terms.append(log_term)
            ys.append(y)
        if not terms:
            # Feasibility was already screened; keep this defensive branch
            # exact rather than emitting undefined conditional moments.
            cond_means.append(float("nan"))
            continue
        top = max(terms)
        mean_j = math.fsum(
            y * math.exp(v - top) for y, v in zip(ys, terms)
        ) / math.fsum(math.exp(v - top) for v in terms)
        cond_means.append(mean_j)

    return PredictiveEventResult(
        counts=row,
        alpha=a,
        m=m_val,
        bounds=bounds_t,
        probability=prob,
        log_probability=log_prob,
        conditional_mean=tuple(cond_means),
    )
