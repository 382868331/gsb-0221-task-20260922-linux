"""Hierarchical multinomial counts with a deterministic empirical-Bayes estimator.

Symmetric Dirichlet hierarchy (K <= 6 categories, G <= 40 groups)::

    theta_g ~ Dirichlet(alpha, ..., alpha)      # one concentration parameter
    n_g | theta_g ~ Multinomial(N_g, theta_g)

The marginal likelihood of one non-empty group is the Dirichlet-multinomial
probability::

    L_g(alpha) = Gamma(K alpha) / Gamma(K alpha + N_g)
                 * prod_k Gamma(alpha + n_gk) / Gamma(alpha)

Empty groups (N_g = 0) have likelihood 1: they return the prior at prediction
time and do not move the estimate of alpha.

Everything in this module is deterministic: no random numbers, no third-party
code.  Only the Python standard library is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ALPHA_MIN",
    "ALPHA_MAX",
    "MAX_CATEGORIES",
    "MAX_GROUPS",
    "OptStatus",
    "FitResult",
    "GroupPosterior",
    "CandidateMode",
    "EBMError",
    "InputValidationError",
    "IdentifiabilityError",
    "SearchBudgetExhaustedError",
    "validate_counts",
    "log_evidence",
    "log_evidence_per_group",
    "compute_posteriors",
    "fit",
]

# ---------------------------------------------------------------------------
# Public constants: explicit search interval and interface size limits.
# ---------------------------------------------------------------------------

ALPHA_MIN = 1e-4
ALPHA_MAX = 1e4
MAX_CATEGORIES = 6
MAX_GROUPS = 40

_LOG_ALPHA_MIN = math.log(ALPHA_MIN)
_LOG_ALPHA_MAX = math.log(ALPHA_MAX)

_DEFAULT_GRID_POINTS = 201
_DEFAULT_TOLERANCE = 1e-10
_DEFAULT_MAX_ITERATIONS = 200

# Endpoint classification band, IN ADDITION to the golden-section tolerance.
# Near alpha = 1e4 the lgamma arguments K*alpha + N carry argument rounding of
# order ulp(4e4) ~ 7e-12, amplified by digamma ~ O(10) into ~1e-11 log-unit
# wiggles that can create a spurious micro-optimum inside the end grid cell.
# A winner counts as an endpoint when it lies within _ENDPOINT_T_DIST (log
# alpha) of an edge and loses to the edge by at most _ENDPOINT_F_NOISE
# (log evidence units).  This is an input-quantization bound, not a loose
# optimizer tolerance, and is documented in the README error/accuracy section.
_ENDPOINT_F_NOISE = 1e-10
_ENDPOINT_T_DIST = 1e-3

# lgamma arguments stay finite well above this; guard anyway for locatable error.
_MAX_SANE_COUNT = 10**300


# ---------------------------------------------------------------------------
# Error types.  Search-budget failure (a runtime limit) is kept separate from
# a proven mathematical outcome (no unique solution exists).
# ---------------------------------------------------------------------------


class EBMError(ValueError):
    """Base class for errors raised by this package."""


class InputValidationError(EBMError):
    """An input violates the fixed public contract."""


class IdentifiabilityError(EBMError):
    """Raised when the evidence provably has no unique optimizer.

    This is a mathematical statement, not a search failure: the marginal
    likelihood is identically 1.0 in alpha, so every alpha is equivalent.
    It occurs when there are no non-empty groups (or K == 1, where the
    multinomial carries no information about theta).
    """


class SearchBudgetExhaustedError(RuntimeError):
    """Raised when a bracketed local search uses its full iteration budget
    without shrinking its bracket to the requested tolerance."""


class OptStatus(str, Enum):
    """Status of the global minimizer search result on log alpha.

    Search-budget exhaustion is not a status: it raises
    SearchBudgetExhaustedError, kept distinct from the proven
    non-identifiability case (IdentifiabilityError).
    """

    INTERIOR = "interior"
    LOWER_ENDPOINT = "lower_endpoint"
    UPPER_ENDPOINT = "upper_endpoint"


# ---------------------------------------------------------------------------
# Public result structures (frozen, so a failed batch cannot mutate them).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupPosterior:
    """Posterior summary for one count group at the fitted alpha.

    posterior_alpha: Dirichlet parameters alpha_hat + n_k.
    mean/variance:    per-category posterior moments of theta_g.
    predictive:      probability of each category for one new observation.
    predicted_category: argmax of the predictive (ties -> lowest index).
    """

    counts: tuple[int, ...]
    total: int
    is_empty: bool
    posterior_alpha: tuple[float, ...]
    concentration: float
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    predictive: tuple[float, ...]
    predicted_category: int


@dataclass(frozen=True)
class CandidateMode:
    """One local-optimization result, found from a grid candidate basin."""

    log_alpha: float
    bracket: tuple[float, float]
    log_evidence: float
    status: str


@dataclass(frozen=True)
class FitResult:
    k: int
    group_count: int
    nonempty_group_count: int
    alpha: float
    log_evidence: float
    status: str
    grid_points: int
    candidate_count: int
    function_evaluations: int
    iterations: int
    candidates: tuple[CandidateMode, ...]
    groups: tuple[GroupPosterior, ...]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_int(value, name: str) -> int:
    # bool is a subclass of int in Python; the contract rejects it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name}: must be int, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_finite_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name}: must be a real number, got {type(value).__name__} ({value!r})"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{name}: must be finite (not NaN or Infinity), got {value!r}")
    return fv


def validate_counts(counts, k=None):
    """Validate a batch of count vectors and return a tuple of int tuples.

    Validation of the whole batch completes before any result is produced, so a
    rejected batch never leaves partially processed state behind.
    """
    if isinstance(counts, (str, bytes)) or not hasattr(counts, "__iter__"):
        raise TypeError(
            "counts: must be an iterable of count vectors, "
            f"got {type(counts).__name__}"
        )

    rows: list[tuple[int, ...]] = []
    for g, row in enumerate(counts):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise TypeError(
                f"group {g}: count vector must be an iterable, "
                f"got {type(row).__name__}"
            )
        vec: list[int] = []
        for c, raw in enumerate(row):
            val = _require_int(raw, f"group {g} category {c} count")
            if val < 0:
                raise InputValidationError(
                    f"group {g} category {c} count: must be >= 0, got {val}"
                )
            if val > _MAX_SANE_COUNT:
                raise ValueError(
                    f"group {g} category {c} count: {val} exceeds supported "
                    f"limit {_MAX_SANE_COUNT} (lgamma range)"
                )
            vec.append(val)
        rows.append(tuple(vec))

    if len(rows) == 0:
        raise InputValidationError("counts: at least one group is required")
    if len(rows) > MAX_GROUPS:
        raise InputValidationError(
            f"counts: {len(rows)} groups exceed the interface limit of {MAX_GROUPS}"
        )

    if k is not None:
        _require_int(k, "k")
        if not 1 <= k <= MAX_CATEGORIES:
            raise InputValidationError(
                f"k: must be in [1, {MAX_CATEGORIES}], got {k}"
            )
    else:
        k = len(rows[0])
        if not 1 <= k <= MAX_CATEGORIES:
            raise InputValidationError(
                f"k: inferred category count must be in [1, {MAX_CATEGORIES}], "
                f"got {k}"
            )

    for g, vec in enumerate(rows):
        if len(vec) != k:
            raise InputValidationError(
                f"group {g}: has {len(vec)} categories but k={k}; "
                "all groups must have the same length"
            )

    return tuple(rows), k


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _split_nonempty(rows):
    return tuple(r for r in rows if sum(r) > 0)


def log_evidence_per_group(alpha, counts, k=None):
    """Return (per-group log likelihoods, validated rows, k).

    Empty groups contribute 0.0.  ``alpha`` may be any finite positive float
    (the evidence function itself is not restricted to the search interval;
    the search interval restricts :func:`fit`).
    """
    alpha = _require_finite_float(alpha, "alpha")
    if alpha <= 0.0:
        raise ValueError(f"alpha: must be > 0, got {alpha!r}")
    rows, kk = validate_counts(counts, k)

    lg_a = math.lgamma(alpha)
    lg_ka = math.lgamma(kk * alpha)
    out = []
    for row in rows:
        n_total = sum(row)
        if n_total == 0:
            out.append(0.0)
            continue
        val = lg_ka - math.lgamma(kk * alpha + n_total) - kk * lg_a
        for n in row:
            val += math.lgamma(alpha + n)
        out.append(val)
    return tuple(out), rows, kk


def log_evidence(alpha, counts, k=None):
    """Pooled log marginal likelihood log p(all counts | alpha).

    Sum over groups of Dirichlet-multinomial log likelihoods; empty groups
    contribute zero.
    """
    per_group, _rows, _k = log_evidence_per_group(alpha, counts, k)
    return math.fsum(per_group)


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------


def compute_posteriors(alpha, counts, k=None):
    """Dirichlet(alpha + n_k) posterior summary for every group.

    Empty groups return the prior: mean 1/K, variance of a symmetric
    Dirichlet component, and a uniform predictive distribution.
    """
    alpha = _require_finite_float(alpha, "alpha")
    if alpha <= 0.0:
        raise ValueError(f"alpha: must be > 0, got {alpha!r}")
    rows, kk = validate_counts(counts, k)

    result = []
    for row in rows:
        n_total = sum(row)
        post_a = tuple(alpha + float(n) for n in row)
        concentration = kk * alpha + float(n_total)
        mean = tuple(a / concentration for a in post_a)
        variance = tuple(m * (1.0 - m) / (concentration + 1.0) for m in mean)
        predictive = mean  # Dirichlet-multinomial one-draw predictive equals mean
        predicted = min(range(kk), key=lambda c: (-mean[c], c))
        result.append(
            GroupPosterior(
                counts=row,
                total=n_total,
                is_empty=(n_total == 0),
                posterior_alpha=post_a,
                concentration=concentration,
                mean=mean,
                variance=variance,
                predictive=predictive,
                predicted_category=predicted,
            )
        )
    return tuple(result)


# ---------------------------------------------------------------------------
# Bracketed 1-D optimization on log alpha
# ---------------------------------------------------------------------------


def _golden_minimize(f, lo, hi, tol, max_iterations):
    """Golden-section minimization on the fixed bracket [lo, hi].

    The search never evaluates outside [lo, hi].  Returns
    (x_best, bracket_lo, bracket_hi, f_best, evaluations, iterations).
    Raises SearchBudgetExhaustedError if the bracket does not shrink below
    tol within max_iterations.
    """
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.618...
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = f(c), f(d)
    evaluations = 2

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - inv_phi * (hi - lo)
            fc = f(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = f(d)
        evaluations += 1
        if hi - lo <= tol:
            break
    else:
        raise SearchBudgetExhaustedError(
            f"golden-section search exhausted {max_iterations} iterations; "
            f"final bracket width {hi - lo:.3e} > tolerance {tol:.3e}"
        )

    if fc < fd:
        hi = d
    else:
        lo = c
    x_best = 0.5 * (lo + hi)
    f_best = f(x_best)
    evaluations += 1
    return x_best, lo, hi, f_best, evaluations, iterations


def _detect_peaks(values, band):
    """Noise-aware detection of local-maximum basins on a sampled function.

    Consecutive values differing by at most ``band`` are treated as a flat
    run.  Each returned peak is ``(kind, lo_index, hi_index)`` with kind
    ``'L'`` (left edge basin), ``'R'`` (right edge basin), or ``'I'``
    (interior basin).  Genuine separate peaks need a valley deeper than
    ``band`` between them; sub-band ripples are merged.
    """
    n = len(values)
    signs = [0] * (n - 1)
    for i in range(n - 1):
        delta = values[i + 1] - values[i]
        signs[i] = 1 if delta > band else (-1 if delta < -band else 0)

    peaks: list[tuple[str, int, int]] = []

    if not signs or all(s == 0 for s in signs):
        peaks.append(("L", 0, 0))
        if n > 1:
            peaks.append(("R", n - 1, n - 1))
        return peaks

    i = 0
    while i < len(signs):
        if signs[i] != 0:
            i += 1
            continue
        a = i
        while i < len(signs) and signs[i] == 0:
            i += 1
        b = i - 1
        entered_from_left = a == 0 or signs[a - 1] == 1
        exits_right = b == len(signs) - 1 or signs[b + 1] == -1
        if entered_from_left and exits_right:
            if a == 0:
                peaks.append(("L", 0, b + 1))
            elif b == len(signs) - 1:
                peaks.append(("R", a, n - 1))
            else:
                peaks.append(("I", a, b + 1))
    for i in range(len(signs) - 1):
        if signs[i] == 1 and signs[i + 1] == -1:
            peaks.append(("I", i + 1, i + 1))
    if signs[0] == -1:
        peaks.append(("L", 0, 0))
    if signs[-1] == 1:
        peaks.append(("R", n - 1, n - 1))

    # De-duplicate while keeping grid order.
    seen: set[tuple[str, int, int]] = set()
    unique = []
    for peak in peaks:
        if peak not in seen:
            seen.add(peak)
            unique.append(peak)
    unique.sort(key=lambda p: (p[1], p[2], p[0]))
    return unique


def fit(
    counts,
    *,
    k=None,
    grid_points=_DEFAULT_GRID_POINTS,
    tolerance=_DEFAULT_TOLERANCE,
    max_iterations=_DEFAULT_MAX_ITERATIONS,
):
    """Estimate alpha by maximizing the pooled log evidence.

    Procedure (fully deterministic):
      1. evaluate log evidence on a uniform grid in log alpha over
         [log 1e-4, log 1e4];
      2. mark every grid local minimum and (where competitive) the endpoints
         as candidate basins;
      3. run a bracketed golden-section search inside each candidate's
         basin bracket (never leaving the explicit interval);
      4. return the best local mode plus endpoint/interior status and the
         complete candidate list so search restrictions remain visible.
    """
    _require_int(grid_points, "grid_points")
    if grid_points < 5:
        raise InputValidationError("grid_points: must be >= 5")
    tol = _require_finite_float(tolerance, "tolerance")
    if not 0.0 < tol < 1.0:
        raise InputValidationError("tolerance: must be in (0, 1)")
    _require_int(max_iterations, "max_iterations")
    if max_iterations < 1:
        raise InputValidationError("max_iterations: must be >= 1")

    rows, kk = validate_counts(counts, k)
    nonempty = _split_nonempty(rows)
    if len(nonempty) == 0:
        raise IdentifiabilityError(
            "all groups are empty: the evidence is identically 1.0, so alpha "
            "has no unique optimizer (proven non-identifiability, not a "
            "search failure)"
        )
    if kk == 1:
        raise IdentifiabilityError(
            "k == 1: a one-category multinomial carries no information about "
            "theta; the evidence is identically 1.0 and alpha is not identifiable"
        )
    if max(sum(r) for r in nonempty) <= 1:
        raise IdentifiabilityError(
            "every non-empty group contains at most one observation: the "
            "marginal likelihood of a single draw is exactly 1/K for every "
            "alpha (Gamma(K alpha)/Gamma(K alpha+1)*Gamma(alpha+1)/Gamma(alpha) "
            "= 1/K), so the evidence is constant and alpha has no optimizer"
        )

    # Step 1: coarse global grid scan.  The MLE maximizes log evidence, so the
    # local searches below minimize its negation on the same grid.
    grid = [
        _LOG_ALPHA_MIN
        + (_LOG_ALPHA_MAX - _LOG_ALPHA_MIN) * i / (grid_points - 1)
        for i in range(grid_points)
    ]
    values = [log_evidence(math.exp(t), nonempty, kk) for t in grid]
    evaluations = grid_points

    # Step 2: noise-aware basin detection.  Near the 1e4 edge the lgamma
    # argument quantization (see _ENDPOINT_F_NOISE) creates ripples below the
    # band on an otherwise monotone/plateau evidence.  Differences of at most
    # the band are snapped to "flat"; maximal flat runs are local-maximum
    # plateaus when entered by a rise (or the left edge) and left by a fall (or
    # the right edge).  Genuine modes must be separated by a valley deeper
    # than the band; sub-band multimodality is unresolvable in float64 anyway.
    band = _ENDPOINT_F_NOISE
    unique_peaks = _detect_peaks(values, band)

    # Step 3: bounded local optimization inside each basin bracket.  Searches
    # never evaluate outside the bracket, which itself lies inside the
    # explicit interval [log ALPHA_MIN, log ALPHA_MAX].
    candidates: list[CandidateMode] = []
    total_iterations = 0
    for kind, lo_idx, hi_idx in unique_peaks:
        if kind == "L":
            lo, hi = grid[0], grid[1]
        elif kind == "R":
            lo, hi = grid[grid_points - 2], grid[grid_points - 1]
        else:
            lo = grid[max(0, lo_idx - 1)]
            hi = grid[min(grid_points - 1, hi_idx + 1)]

        def objective(t):
            return -log_evidence(math.exp(t), nonempty, kk)

        x, blo, bhi, fx, evals, iters = _golden_minimize(
            objective, lo, hi, tol, max_iterations
        )
        evaluations += evals
        total_iterations += iters

        status = OptStatus.INTERIOR.value
        # Edge basins: accept the literal boundary point whenever it ties the
        # golden result within the quantization band.
        if kind == "L":
            f_edge = objective(_LOG_ALPHA_MIN)
            evaluations += 1
            if f_edge <= fx + band:
                x, fx, status = _LOG_ALPHA_MIN, f_edge, OptStatus.LOWER_ENDPOINT.value
        elif kind == "R":
            f_edge = objective(_LOG_ALPHA_MAX)
            evaluations += 1
            if f_edge <= fx + band:
                x, fx, status = _LOG_ALPHA_MAX, f_edge, OptStatus.UPPER_ENDPOINT.value

        candidates.append(
            CandidateMode(
                log_alpha=x,
                bracket=(blo, bhi),
                log_evidence=-fx,
                status=status,
            )
        )

    # Step 4: global pick; on exact ties prefer smaller log alpha (deterministic).
    best = min(candidates, key=lambda c: (-c.log_evidence, c.log_alpha))

    # Final edge safety check: an interior winner sitting essentially on an
    # edge with an edge value tying it within the band is reported as the edge.
    if best.status == OptStatus.INTERIOR.value:
        for edge_t, edge_alpha, edge_status in (
            (_LOG_ALPHA_MIN, ALPHA_MIN, OptStatus.LOWER_ENDPOINT),
            (_LOG_ALPHA_MAX, ALPHA_MAX, OptStatus.UPPER_ENDPOINT),
        ):
            if abs(best.log_alpha - edge_t) <= _ENDPOINT_T_DIST:
                f_edge = -log_evidence(edge_alpha, nonempty, kk)
                evaluations += 1
                if f_edge <= -best.log_evidence + band:
                    best = CandidateMode(
                        log_alpha=edge_t,
                        bracket=best.bracket,
                        log_evidence=-f_edge,
                        status=edge_status.value,
                    )
                    break

    if best.status == OptStatus.LOWER_ENDPOINT.value:
        alpha_star = ALPHA_MIN
    elif best.status == OptStatus.UPPER_ENDPOINT.value:
        alpha_star = ALPHA_MAX
    else:
        alpha_star = math.exp(best.log_alpha)
    groups = compute_posteriors(alpha_star, rows, kk)

    return FitResult(
        k=kk,
        group_count=len(rows),
        nonempty_group_count=len(nonempty),
        alpha=alpha_star,
        log_evidence=best.log_evidence,
        status=best.status,
        grid_points=grid_points,
        candidate_count=len(candidates),
        function_evaluations=evaluations,
        iterations=total_iterations,
        candidates=tuple(candidates),
        groups=groups,
    )
