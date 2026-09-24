"""Hierarchical multinomial counts: deterministic empirical Bayes.

A reusable, standard-library-only module for the following small-sample
hierarchical model::

    theta_g ~ Dirichlet(alpha_1, alpha_2, ..., alpha_K)   # K classes
    x_g     ~ Multinomial(n_g, theta_g)                   # g = 1..G groups

Two parameterizations are supported, sharing the same evidence/post-processing
code:

* **symmetric** (``alpha_1 = ... = alpha_K = alpha``): a single shared
  concentration estimated by :func:`fit_alpha` on the explicit interval
  ``[ALPHA_MIN, ALPHA_MAX]``, optimizing over ``log(alpha)``;
* **per-class** (independent ``alpha_k > 0``): :func:`fit_alphas` runs a
  finite-iteration cyclic coordinate search over the box
  ``[ALPHA_MIN, ALPHA_MAX]**K``, reusing the same bracket-preserving 1-D
  engine for each coordinate, and reports convergence explicitly.

Public API
----------
log_evidence(alpha, groups) -> float                 # symmetric scalar
log_evidence_one(alpha, counts) -> float
log_evidence_dirichlet(alphas, groups) -> float      # per-class vector
fit_alpha(groups, *, grid_points=201, max_iter=100) -> FitResult
fit_alphas(groups, *, init=None, sweeps=60,
           grid_points=25, max_iter=100) -> AlphasFitResult
posterior_group(counts, alpha) -> GroupPosterior
posterior_group_dirichlet(counts, alphas) -> DirichletGroupPosterior
analyze(groups, ...) -> EBResult
analyze_dirichlet(groups, ...) -> DirichletEBResult

Contract limits: 2 <= K <= 6, 1 <= G <= 40, non-negative integer counts.
Numeric inputs reject NaN/Infinity; integer parameters reject ``bool``; every
concentration must be strictly positive.
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

DEFAULT_GRID_POINTS = 201
DEFAULT_MAX_ITER = 100

# Per-class fit defaults: cyclic coordinate sweeps over [ALPHA_MIN, ALPHA_MAX]^K.
DEFAULT_SWEEPS = 60
DEFAULT_DIRICHLET_GRID_POINTS = 25

# Relative convergence threshold for the per-class coordinate search: a sweep
# counts as converged only when the total-log-evidence gain falls below this
# fraction of the current value magnitude (absolute floor 1e-12). Each 1-D
# step localizes its maximizer to T_TOL/2 in t = log(alpha); the sweep
# tolerance is set near that attainable floor so termination is driven by a
# genuine evidence plateau rather than a loose budget.
EVIDENCE_REL_TOL = 1e-13
_EVIDENCE_ABS_FLOOR = 1e-13

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

# Per-class coordinate-fit statuses.
STATUS_CONVERGED = "converged"
STATUS_MAX_SWEEPS_REACHED = "max_sweeps_reached"


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
# Per-class (non-symmetric) Dirichlet evidence
# ---------------------------------------------------------------------------


def _validate_alphas(
    alphas: Sequence[float], k: int | None
) -> tuple[tuple[float, ...], int]:
    """Validate a per-class concentration vector of length K.

    Every entry must be a finite, strictly positive number (bool rejected);
    when ``k`` is given, the vector length must equal it.
    """
    if isinstance(alphas, (str, bytes)) or not isinstance(alphas, Sequence):
        raise TypeError(
            "alphas must be a sequence of positive numbers, got "
            f"{type(alphas).__name__}"
        )
    kk = len(alphas)
    if k is not None and kk != k:
        raise ValueError(
            f"alphas has length {kk}, expected K={k}"
        )
    if not 2 <= kk <= MAX_K:
        raise ValueError(f"alphas has length {kk}; require 2 <= K <= {MAX_K}")
    out: list[float] = []
    for j, raw in enumerate(alphas):
        if isinstance(raw, bool):
            raise TypeError(
                f"alpha class {j}: concentration must be int or float, got bool"
            )
        if not isinstance(raw, (int, float)):
            raise TypeError(
                f"alpha class {j}: concentration must be int or float, got "
                f"{type(raw).__name__}: {raw!r}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(
                f"alpha class {j}: concentration must be finite and not "
                f"NaN/Infinity, got {raw!r}"
            )
        if value <= 0.0:
            raise ValueError(
                f"alpha class {j}: concentration must be > 0, got {raw!r}"
            )
        out.append(value)
    return tuple(out), kk


def _log_evidence_row_vec(
    alphas: tuple[float, ...], row: tuple[int, ...]
) -> float:
    """Dirichlet-multinomial log evidence with class-specific alphas."""
    n = sum(row)
    if n == 0:
        # An empty draw has probability 1 for every prior parameter value.
        return 0.0
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in row
    )
    a0 = math.fsum(alphas)
    return (
        log_coef
        - _lgamma_add(a0, n)
        + math.fsum(_lgamma_add(a, c) for a, c in zip(alphas, row))
    )


def log_evidence_dirichlet(
    alphas: Sequence[float], groups: Sequence[Sequence[int]]
) -> float:
    """Summed log marginal likelihood under Dirichlet(alpha_1,...,alpha_K).

    Every concentration must be finite and strictly positive (no box clamp
    here: the ``[ALPHA_MIN, ALPHA_MAX]`` box only constrains the optimizer).
    Empty groups contribute zero.
    """
    rows, k = _validate_groups(groups)
    avec, _ = _validate_alphas(alphas, k)
    return math.fsum(_log_evidence_row_vec(avec, row) for row in rows)


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
        elif fc < fd:
            a = c
            c, fc = d, fd
            d = a + (b - a) * ratio
            fd = f(d)
        else:
            # Exact tie: the top is flat to evaluation precision. Shrinking
            # from one side only would walk the bracket off the maximizer
            # (successive ties all take the same branch), so contract both
            # ends to [c, d]; its midpoint equals (a+b)/2, keeping the bracket
            # centered on the mode instead of drifting.
            a, b = c, d
            c = b - (b - a) * ratio
            d = a + (b - a) * ratio
            fc, fd = f(c), f(d)
        if b - a <= T_TOL:
            break
    # The maximizer is guaranteed to lie in the final bracket [a, b]. Return
    # its midpoint: geometric error is at most (b-a)/2 <= T_TOL/2 even when
    # the top is unresolvable by function value.
    m = 0.5 * (a + b)
    return m, f(m), used, (b - a) <= T_TOL


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
# Per-class concentration fit: cyclic coordinate ascent over the positive box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphasFitResult:
    """Outcome of the finite-iteration per-class concentration fit.

    ``status`` is one of ``converged`` / ``max_sweeps_reached`` / ``flat``.
    Unlike the symmetric case, a batch whose non-empty groups all hold a single
    draw is NOT flat here: a lone draw favours its observed class with a
    concentration ratio, so the evidence depends on the per-class vector and
    its maximum typically sits on the box (``at_box`` reports this).
    """

    alphas: tuple[float, ...]
    log_evidence: float
    status: str
    converged: bool
    sweeps: int
    sweeps_used: int
    grid_points: int
    max_iterations_used: int
    at_box: tuple[bool, ...]  # per class: resting on ALPHA_MIN/ALPHA_MAX
    n_nonempty_groups: int

    @property
    def budget_exhausted(self) -> bool:
        """True iff the sweep budget ran out before the gain tolerance."""
        return not self.converged and self.status != STATUS_FLAT


@dataclass(frozen=True)
class DirichletGroupPosterior:
    counts: tuple[int, ...]
    n: int
    is_empty: bool
    alphas: tuple[float, ...]
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    predictive: tuple[float, ...]
    predicted: int  # argmax of predictive; ties resolved to the lowest index


@dataclass(frozen=True)
class DirichletEBResult:
    k: int
    n_groups: int
    n_nonempty_groups: int
    fit: AlphasFitResult
    posteriors: tuple[DirichletGroupPosterior, ...]


def fit_alphas(
    groups: Sequence[Sequence[int]],
    *,
    init: Sequence[float] | None = None,
    sweeps: int = DEFAULT_SWEEPS,
    grid_points: int = DEFAULT_DIRICHLET_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> AlphasFitResult:
    """Fit independent per-class concentrations by cyclic coordinate ascent.

    Each sweep maximizes the total log evidence over one ``alpha_j`` on
    ``[ALPHA_MIN, ALPHA_MAX]`` with the other coordinates fixed, using the
    same grid-plus-bracketed-golden-section engine as :func:`fit_alpha`.
    Every 1-D slice of the Dirichlet-multinomial evidence is strictly concave
    for a non-empty batch — its second derivative is
    ``[psi'(A) - psi'(A+n_g)] + [psi'(alpha_j+x_gj) - psi'(alpha_j)] < 0``
    summed over groups (trigamma is strictly decreasing) — so each coordinate
    step has a unique boxed maximum.

    Positivity is enforced by the closed box rather than by clamping: no
    candidate outside ``[ALPHA_MIN, ALPHA_MAX]`` is ever evaluated. The
    iteration budget is finite and reported honestly — ``converged=False`` /
    ``status='max_sweeps_reached'`` when the evidence gain of a full sweep
    does not fall below ``EVIDENCE_REL_TOL`` (relative, floor 1e-12) within
    ``sweeps`` sweeps. All-empty batches are the only proved-flat case and
    return alpha = 1 for every class without searching.
    """
    rows, k = _validate_groups(groups)
    sw = _reject_bool_int("sweeps", sweeps, 1)
    gp = _reject_bool_int("grid_points", grid_points, 3)
    mi = _reject_bool_int("max_iter", max_iter, 1)
    if init is None:
        current = (1.0,) * k
    else:
        current, _ = _validate_alphas(init, k)
        for j, a in enumerate(current):
            if not ALPHA_MIN <= a <= ALPHA_MAX:
                raise ValueError(
                    f"init class {j}: concentration {a!r} must lie inside the "
                    f"fit box [{ALPHA_MIN}, {ALPHA_MAX}]"
                )

    nonempty = tuple(row for row in rows if sum(row) > 0)
    if not nonempty:
        # Evidence identically zero for every parameter vector: provably no
        # unique maximizer, not a budget condition.
        return AlphasFitResult(
            alphas=(1.0,) * k,
            log_evidence=0.0,
            status=STATUS_FLAT,
            converged=True,
            sweeps=sw,
            sweeps_used=0,
            grid_points=gp,
            max_iterations_used=0,
            at_box=(False,) * k,
            n_nonempty_groups=0,
        )

    def slice_objective(j: int, fixed: tuple[float, ...]):
        def f(a_j: float) -> float:
            vec = fixed[:j] + (a_j,) + fixed[j + 1:]
            return math.fsum(_log_evidence_row_vec(vec, r) for r in nonempty)

        return f

    value = math.fsum(_log_evidence_row_vec(current, r) for r in nonempty)
    max_iterations_used = 0
    converged = False
    sweeps_used = 0

    for sweeps_used in range(1, sw + 1):
        before = value
        for j in range(k):
            step = maximize_on_interval(
                slice_objective(j, current),
                grid_points=gp,
                max_iter=mi,
            )
            max_iterations_used = max(
                max_iterations_used, step.max_iterations_used
            )
            current = current[:j] + (step.alpha,) + current[j + 1:]
            value = step.log_evidence
        gain = value - before
        tol = max(_EVIDENCE_ABS_FLOOR, EVIDENCE_REL_TOL * abs(value))
        if 0.0 <= gain <= tol:
            converged = True
            break
        # A negative gain cannot happen when each coordinate step maximizes
        # over a closed interval containing the current point; guard anyway
        # against treating numerical noise as progress.
        if gain < -tol:
            raise RuntimeError(  # pragma: no cover - defensive invariant
                f"coordinate ascent decreased evidence by {gain}; "
                "bracketed maximizer violated containment"
            )

    at_box = tuple(
        a <= ALPHA_MIN * (1.0 + 1e-9) or a >= ALPHA_MAX * (1.0 - 1e-9)
        for a in current
    )
    return AlphasFitResult(
        alphas=current,
        log_evidence=value,
        status=(
            STATUS_CONVERGED if converged else STATUS_MAX_SWEEPS_REACHED
        ),
        converged=converged,
        sweeps=sw,
        sweeps_used=sweeps_used,
        grid_points=gp,
        max_iterations_used=max_iterations_used,
        at_box=at_box,
        n_nonempty_groups=len(nonempty),
    )


def posterior_group_dirichlet(
    counts: Sequence[int], alphas: Sequence[float]
) -> DirichletGroupPosterior:
    """Posterior summary under Dirichlet(alpha_1,...,alpha_K).

    The posterior is Dirichlet(alpha_k + x_k); empty vectors return the
    prior itself.
    """
    rows, k = _validate_groups([counts])
    avec, _ = _validate_alphas(alphas, k)
    row = rows[0]
    n = sum(row)
    post_a = tuple(a + c for a, c in zip(avec, row))
    total = math.fsum(post_a)
    mean = tuple(a / total for a in post_a)
    variance = tuple(
        a * (total - a) / (total * total * (total + 1.0)) for a in post_a
    )
    # One-step-ahead multinomial predictive: equals the posterior mean.
    predictive = mean
    predicted = 0
    for j in range(1, k):
        if predictive[j] > predictive[predicted]:
            predicted = j
    return DirichletGroupPosterior(
        counts=row,
        n=n,
        is_empty=(n == 0),
        alphas=avec,
        mean=mean,
        variance=variance,
        predictive=predictive,
        predicted=predicted,
    )


def analyze_dirichlet(
    groups: Sequence[Sequence[int]],
    *,
    init: Sequence[float] | None = None,
    sweeps: int = DEFAULT_SWEEPS,
    grid_points: int = DEFAULT_DIRICHLET_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> DirichletEBResult:
    """Fit per-class concentrations and summarize every group's posterior."""
    rows, k = _validate_groups(groups)
    fit = fit_alphas(
        rows, init=init, sweeps=sweeps,
        grid_points=grid_points, max_iter=max_iter,
    )
    posteriors = tuple(
        posterior_group_dirichlet(row, fit.alphas) for row in rows
    )
    return DirichletEBResult(
        k=k,
        n_groups=len(rows),
        n_nonempty_groups=fit.n_nonempty_groups,
        fit=fit,
        posteriors=posteriors,
    )
