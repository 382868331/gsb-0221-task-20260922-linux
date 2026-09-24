"""Hierarchical multinomial counts: deterministic empirical Bayes.

A reusable, standard-library-only module for the following small-sample
hierarchical model::

    theta_g ~ Dirichlet(alpha, alpha, ..., alpha)   # K classes, symmetric
    x_g     ~ Multinomial(n_g, theta_g)              # g = 1..G groups

A single concentration ``alpha`` is shared by every group and estimated by
maximizing the summed (Dirichlet-)multinomial marginal likelihood on the
explicit interval ``[ALPHA_MIN, ALPHA_MAX]``, optimizing over ``log(alpha)``.

The prior can also be asymmetric: ``Dirichlet(alpha_1, ..., alpha_K)`` with
independent per-class concentrations.  ``log_evidence``,
``log_evidence_one`` and ``posterior_group`` accept either a scalar alpha
(symmetric prior, original behavior) or a length-K sequence of positive
concentrations.  ``fit_concentrations`` estimates the per-class vector with
a finite-budget Minka fixed-point iteration, projecting every iterate onto
the positive interval ``[ALPHA_MIN, ALPHA_MAX]`` and reporting convergence
explicitly.

Public API
----------
log_evidence(alpha_or_alphas, groups) -> float
fit_alpha(groups, *, grid_points=201, max_iter=100) -> FitResult
fit_concentrations(groups, *, max_iter=1000, tol=1e-10, start=None)
    -> ConcentrationFit
posterior_group(counts, alpha_or_alphas) -> GroupPosterior
analyze(groups, *, grid_points=201, max_iter=100) -> EBResult
analyze_concentrations(groups, *, max_iter=1000, tol=1e-10, start=None)
    -> EBConcentrationResult

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

# Per-class concentration iteration statuses.
STATUS_CONVERGED = "converged"
STATUS_MAX_ITER_EXHAUSTED = "max_iter_exhausted"

DEFAULT_CONC_MAX_ITER = 1000
DEFAULT_CONC_TOL = 1e-10


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


def _validate_alpha_value(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int or float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be int or float, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite and not NaN/Infinity, got {value!r}"
        )
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _validate_alpha(alpha: object) -> float:
    return _validate_alpha_value(alpha, "alpha")


def _split_alpha(
    alpha: object, k: int
) -> tuple[float | None, tuple[float, ...] | None]:
    """Validate a scalar-or-vector concentration argument.

    Returns ``(scalar, None)`` for a scalar alpha (symmetric prior) or
    ``(None, vector)`` for a length-K sequence of per-class concentrations.
    Exactly one element of the pair is not None.
    """
    if isinstance(alpha, Sequence) and not isinstance(alpha, (str, bytes)):
        if len(alpha) != k:
            raise ValueError(
                f"alpha vector has length {len(alpha)}, expected K={k}: "
                "one positive concentration per class"
            )
        vector = tuple(
            _validate_alpha_value(a, f"alpha[{j}]") for j, a in enumerate(alpha)
        )
        return None, vector
    return _validate_alpha(alpha), None


def _validate_tol(tol: object) -> float:
    if isinstance(tol, bool):
        raise TypeError("tol must be a float in (0, 1), got bool")
    if not isinstance(tol, (int, float)):
        raise TypeError(
            f"tol must be int or float, got {type(tol).__name__}"
        )
    value = float(tol)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"tol must be finite and in (0, 1), got {tol!r}")
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


def _digamma(x: float) -> float:
    """Digamma function psi(x) for x > 0.

    Recurrence up to x >= 8, then the asymptotic expansion through the
    x**-10 term; absolute error is below 1e-12 there.
    """
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0
        - inv2
        * (
            1.0 / 120.0
            - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))
        )
    )
    return result


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


def _log_evidence_row_vec(alphas: tuple[float, ...], row: tuple[int, ...]) -> float:
    """Log marginal likelihood of one row under Dirichlet(alphas)."""
    n = sum(row)
    if n == 0:
        # Same degenerate fact as the symmetric case: an empty draw has
        # probability 1 under every prior and carries no information.
        return 0.0
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in row
    )
    total = math.fsum(alphas)
    return (
        log_coef
        - _lgamma_add(total, n)
        + math.fsum(_lgamma_add(a, c) for a, c in zip(alphas, row))
    )


def log_evidence_one(alpha: object, counts: Sequence[int]) -> float:
    """Log marginal likelihood of one count vector.

    ``alpha`` is either a positive scalar (symmetric prior) or a length-K
    sequence of positive per-class concentrations.  Scalars accept any
    finite positive value (the [1e-4, 1e4] interval binds the optimizers,
    not this function).  An all-zero vector returns 0.0.
    """
    rows, k = _validate_groups([counts])
    scalar, vector = _split_alpha(alpha, k)
    if vector is None:
        assert scalar is not None
        return _log_evidence_row(scalar, k, rows[0])
    return _log_evidence_row_vec(vector, rows[0])


def log_evidence(alpha: object, groups: Sequence[Sequence[int]]) -> float:
    """Summed log evidence over groups; empty groups contribute zero.

    ``alpha`` is either a positive scalar (symmetric prior) or a length-K
    sequence of positive per-class concentrations.
    """
    rows, k = _validate_groups(groups)
    scalar, vector = _split_alpha(alpha, k)
    if vector is None:
        assert scalar is not None
        return math.fsum(_log_evidence_row(scalar, k, row) for row in rows)
    return math.fsum(_log_evidence_row_vec(vector, row) for row in rows)


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
    alpha: float | tuple[float, ...]  # scalar, or per-class vector if so passed
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


@dataclass(frozen=True)
class ConcentrationFit:
    """Result of the finite-iteration per-class concentration fit.

    ``status`` is one of:

    * ``"converged"`` — the maximum relative change between the last two
      iterates dropped below ``tolerance`` within ``max_iter`` updates;
    * ``"max_iter_exhausted"`` — the iteration budget ran out first
      (``converged`` is False; the returned vector is the last iterate,
      still a valid positive concentration, not a proved maximizer);
    * ``"flat"`` — every group is empty, the evidence is identically zero
      and no unique maximizer exists; ``alphas`` is ``(1.0, ..., 1.0)`` by
      convention and ``iterations`` is 0.

    ``clamped_classes`` lists the class indices whose final concentration
    sits exactly on the [ALPHA_MIN, ALPHA_MAX] projection boundary (a class
    never observed in any group rests at ALPHA_MIN).
    """

    alphas: tuple[float, ...]
    log_evidence: float
    iterations: int
    converged: bool
    status: str  # converged | max_iter_exhausted | flat
    final_step: float  # max relative change of the last update (0.0 if flat)
    clamped_classes: tuple[int, ...]
    n_nonempty_groups: int
    max_iter: int
    tolerance: float


@dataclass(frozen=True)
class EBConcentrationResult:
    k: int
    n_groups: int
    n_nonempty_groups: int
    fit: ConcentrationFit
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
    """Posterior summary for one count vector under the Dirichlet prior.

    ``alpha`` is either a positive scalar (symmetric prior) or a length-K
    sequence of positive per-class concentrations; the returned
    ``GroupPosterior.alpha`` field echoes the validated form that was
    passed (float for a scalar, tuple of floats for a vector).

    Empty vectors return the prior itself (uniform mean for a scalar alpha,
    the normalized concentrations for a vector alpha); they need no fitted
    alpha.
    """
    rows, k = _validate_groups([counts])
    row = rows[0]
    scalar, vector = _split_alpha(alpha, k)
    n = sum(row)
    if vector is None:
        assert scalar is not None
        alphas: tuple[float, ...] = (scalar,) * k
        total = k * scalar + n
        stored_alpha: object = scalar
    else:
        alphas = vector
        total = math.fsum(vector) + n
        stored_alpha = vector
    a_post = tuple(a + c for a, c in zip(alphas, row))
    mean = tuple(ap / total for ap in a_post)
    variance = tuple(
        ap * (total - ap) / (total * total * (total + 1.0)) for ap in a_post
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
        alpha=stored_alpha,
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


def fit_concentrations(
    groups: Sequence[Sequence[int]],
    *,
    max_iter: int = DEFAULT_CONC_MAX_ITER,
    tol: float = DEFAULT_CONC_TOL,
    start: object = None,
) -> ConcentrationFit:
    """Fit per-class Dirichlet concentrations by fixed-point iteration.

    Maximizes the summed Dirichlet-multinomial log evidence over the
    per-class vector ``(alpha_1, ..., alpha_K)`` with Minka's fixed-point
    update for the Dirichlet-multinomial::

        alpha_k <- alpha_k * S_k / D
        S_k = sum_g [ psi(x_gk + alpha_k) - psi(alpha_k) ]
        D   = sum_g [ psi(n_g + A) - psi(A) ],   A = sum_k alpha_k

    whose fixed points are exactly the stationary points of the evidence
    (score_k = S_k - D = 0 for every class k).  Every iterate is projected
    onto the explicit positive interval [ALPHA_MIN, ALPHA_MAX], so a
    non-positive or overflowing update can never escape the valid region;
    classes landing on the boundary are reported in ``clamped_classes``.

    The iteration is finite by contract: at most ``max_iter`` updates.
    ``converged=True`` requires the last relative change of every
    coordinate to fall below ``tol``; otherwise the result is returned
    with ``status="max_iter_exhausted"`` and ``converged=False`` — a
    budget report, not an exception.  An all-empty batch is the proved
    ``flat`` case (evidence identically zero) and skips the iteration.

    ``start`` is None (start at the uniform vector ``(1.0, ..., 1.0)``),
    a positive scalar (broadcast), or a length-K positive vector.  Note
    the scale A is not identifiable from single-draw (n=1) groups alone;
    the update preserves the starting sum of concentrations, so such data
    converges to the point on the evidence ridge with that sum.

    Known limitation: the fixed-point update converges linearly.  When the
    likelihood seeks the boundary (a class never observed, or group
    proportions so consistent that the total concentration wants to grow
    without bound), the iterate creeps toward the projection boundary
    geometrically and the finite budget is honestly reported as
    ``max_iter_exhausted`` rather than as convergence.
    """
    rows, k = _validate_groups(groups)
    mi = _reject_bool_int("max_iter", max_iter, 1)
    tol_f = _validate_tol(tol)
    if start is None:
        current = (1.0,) * k
    else:
        s_scalar, s_vector = _split_alpha(start, k)
        if s_vector is not None:
            current = s_vector
        else:
            assert s_scalar is not None
            current = (s_scalar,) * k

    nonempty = tuple(row for row in rows if sum(row) > 0)
    if not nonempty:
        # Evidence is identically zero on the whole positive orthant:
        # provably no unique maximizer. This is not a budget failure.
        return ConcentrationFit(
            alphas=(1.0,) * k,
            log_evidence=0.0,
            iterations=0,
            converged=True,
            status=STATUS_FLAT,
            final_step=0.0,
            clamped_classes=(),
            n_nonempty_groups=0,
            max_iter=mi,
            tolerance=tol_f,
        )

    converged = False
    final_step = math.inf
    iterations = 0
    for iterations in range(1, mi + 1):
        total = math.fsum(current)
        dig_total = _digamma(total)
        # D > 0: every non-empty row has n >= 1 and psi is increasing.
        denom = math.fsum(
            _digamma(sum(row) + total) - dig_total for row in nonempty
        )
        nxt = []
        for j in range(k):
            aj = current[j]
            numer = math.fsum(
                _digamma(row[j] + aj) - _digamma(aj)
                for row in nonempty
                if row[j] > 0
            )
            updated = aj * numer / denom
            # Positivity/bounds constraint: project onto the interval.
            nxt.append(min(max(updated, ALPHA_MIN), ALPHA_MAX))
        final_step = max(
            abs(new - old) / old for new, old in zip(nxt, current)
        )
        current = tuple(nxt)
        if final_step < tol_f:
            converged = True
            break

    log_ev = math.fsum(_log_evidence_row_vec(current, row) for row in nonempty)
    clamped = tuple(
        j
        for j, a in enumerate(current)
        if a == ALPHA_MIN or a == ALPHA_MAX
    )
    return ConcentrationFit(
        alphas=current,
        log_evidence=log_ev,
        iterations=iterations,
        converged=converged,
        status=STATUS_CONVERGED if converged else STATUS_MAX_ITER_EXHAUSTED,
        final_step=final_step,
        clamped_classes=clamped,
        n_nonempty_groups=len(nonempty),
        max_iter=mi,
        tolerance=tol_f,
    )


def analyze_concentrations(
    groups: Sequence[Sequence[int]],
    *,
    max_iter: int = DEFAULT_CONC_MAX_ITER,
    tol: float = DEFAULT_CONC_TOL,
    start: object = None,
) -> EBConcentrationResult:
    """Fit per-class concentrations and return aligned group posteriors."""
    rows, k = _validate_groups(groups)
    fit = fit_concentrations(rows, max_iter=max_iter, tol=tol, start=start)
    posteriors = tuple(posterior_group(row, fit.alphas) for row in rows)
    return EBConcentrationResult(
        k=k,
        n_groups=len(rows),
        n_nonempty_groups=fit.n_nonempty_groups,
        fit=fit,
        posteriors=posteriors,
    )
