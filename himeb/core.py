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

Partial (category-aggregated) observations, where each group reports counts
for the sets of a partition of the original K classes (different groups may
use different partitions)::

    log_evidence_partial(alpha, k, observations) -> float
    fit_alpha_partial(k, observations, *, ...) -> PartialFitResult

Each observation is a sequence of ``(block, count)`` pairs; ``block`` is a
sequence of class indices and the blocks must form a complete partition of
``range(k)``. A block of m classes carries the symmetric parameter m*alpha.

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


def _validate_partial_observations(
    k: object,
    observations: Sequence[Sequence[tuple[Sequence[int], int]]],
) -> tuple[int, tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]:
    """Validate (K, observations) with per-group class partitions.

    Every observation is a sequence of ``(block, count)`` pairs. The blocks
    must be a complete partition of ``range(k)``: duplicated class indices,
    missing classes, overlapping blocks, empty blocks or out-of-range
    indices are all rejected. All checking happens before any result is
    produced; the functions consuming this are pure.
    """
    if isinstance(k, bool):
        raise TypeError("k must be int, got bool (bool is not accepted)")
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}: {k!r}")
    if not 2 <= k <= MAX_K:
        raise ValueError(f"k={k}; require 2 <= k <= {MAX_K}")

    if isinstance(
        observations, (str, bytes)
    ) or not isinstance(observations, Sequence):
        raise TypeError(
            "observations must be a sequence of partitions, got "
            f"{type(observations).__name__}"
        )
    g_count = len(observations)
    if g_count == 0:
        raise ValueError("observations must contain at least one group")
    if g_count > MAX_GROUPS:
        raise ValueError(
            f"number of groups {g_count} exceeds limit MAX_GROUPS="
            f"{MAX_GROUPS}"
        )

    validated: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for gi, observation in enumerate(observations):
        if isinstance(
            observation, (str, bytes)
        ) or not isinstance(observation, Sequence):
            raise TypeError(
                f"group {gi}: observation must be a sequence of "
                f"(block, count) pairs, got {type(observation).__name__}"
            )
        if len(observation) == 0:
            raise ValueError(
                f"group {gi}: partition must contain at least one block"
            )

        seen = [False] * k
        blocks: list[tuple[int, ...]] = []
        counts: list[int] = []
        for bi, item in enumerate(observation):
            where = f"group {gi} block {bi}"
            if (
                isinstance(item, (str, bytes))
                or not isinstance(item, Sequence)
                or len(item) != 2
            ):
                raise TypeError(
                    f"{where}: each partition entry must be a "
                    "(block, count) pair"
                )
            raw_block, count = item
            if isinstance(
                raw_block, (str, bytes)
            ) or not isinstance(raw_block, Sequence):
                raise TypeError(
                    f"{where}: block must be a sequence of class indices, "
                    f"got {type(raw_block).__name__}"
                )
            if len(raw_block) == 0:
                raise ValueError(f"{where}: block must be non-empty")

            block: list[int] = []
            for class_index in raw_block:
                if isinstance(class_index, bool):
                    raise TypeError(
                        f"{where}: class index must be int, got bool "
                        "(bool is not accepted as a class index)"
                    )
                if not isinstance(class_index, int):
                    raise TypeError(
                        f"{where}: class index must be int, got "
                        f"{type(class_index).__name__}: {class_index!r}"
                    )
                if not 0 <= class_index < k:
                    raise ValueError(
                        f"{where}: class index {class_index} out of range "
                        f"for K={k}"
                    )
                if seen[class_index]:
                    raise ValueError(
                        f"{where}: class {class_index} appears more than "
                        "once; blocks must not overlap or repeat classes"
                    )
                seen[class_index] = True
                block.append(class_index)

            if isinstance(count, bool):
                raise TypeError(
                    f"{where}: count must be int, got bool "
                    "(bool is not accepted as a count)"
                )
            if not isinstance(count, int):
                raise TypeError(
                    f"{where}: count must be int, got "
                    f"{type(count).__name__}: {count!r}"
                )
            if count < 0:
                raise ValueError(
                    f"{where}: count must be >= 0, got {count}"
                )
            if count > _MAX_EXACT_COUNT:
                raise ValueError(
                    f"{where}: count {count} exceeds the supported "
                    "exact-integer limit 2**53"
                )
            blocks.append(tuple(block))
            counts.append(count)

        missing = [j for j, found in enumerate(seen) if not found]
        if missing:
            raise ValueError(
                f"group {gi}: partition omits class(es) {missing}; it "
                "must cover every class exactly once"
            )
        validated.append((tuple(blocks), tuple(counts)))

    return k, tuple(validated)


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


def _log_evidence_merged_row(
    alpha: float,
    k: int,
    blocks: tuple[tuple[int, ...], ...],
    counts: tuple[int, ...],
) -> float:
    """Log evidence of one category-aggregated observation.

    Aggregating a symmetric Dirichlet over a block of m classes gives a
    symmetric Dirichlet with parameter m*alpha; the block-level counts are
    therefore again Dirichlet-multinomial (with L categories, parameters
    m_s*alpha). This is exact: summing a fine-grained marginal likelihood
    over every fine allocation consistent with the block counts gives the
    same merged marginal likelihood by additivity of the alpha parameters.
    """
    n = sum(counts)
    if n == 0:
        # An empty draw has probability 1 for every alpha.
        return 0.0
    params = tuple(alpha * len(block) for block in blocks)
    log_coef = math.lgamma(n + 1) - math.fsum(
        math.lgamma(c + 1) for c in counts
    )
    total_param = alpha * k  # sum_s m_s * alpha
    return (
        log_coef
        - _lgamma_add(total_param, n)
        + math.fsum(
            _lgamma_add(param, c) for param, c in zip(params, counts)
        )
    )


def _row_is_alpha_constant(
    blocks: tuple[tuple[int, ...], ...], counts: tuple[int, ...]
) -> bool:
    """True iff the merged row carries no information about alpha.

    Two proved cases: an empty row (n=0, evidence 0) and a single reported
    block (the fully merged partition; its draw is certain) OR n=1 (a lone
    draw lands in block s with prior probability m_s/K, independent of
    alpha). A row with n >= 2 and at least two blocks is informative.
    """
    n = sum(counts)
    if n <= 1:
        return True
    return len(blocks) <= 1


def log_evidence_partial(
    alpha: object,
    k: int,
    observations: Sequence[Sequence[tuple[Sequence[int], int]]],
) -> float:
    """Summed log evidence for category-aggregated observations.

    Each observation is a sequence of ``(block, count)`` pairs, where
    ``block`` is a sequence of original class indices and the blocks form a
    complete partition of ``range(k)`` (different groups may use different
    partitions). The prior stays the single symmetric Dirichlet(alpha,...):
    a block of m classes uses the aggregated parameter m*alpha -- no extra
    asymmetric parameter set.
    """
    a = _validate_alpha(alpha)
    kk, validated = _validate_partial_observations(k, observations)
    return math.fsum(
        _log_evidence_merged_row(a, kk, blocks, counts)
        for blocks, counts in validated
    )


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
class EBResult:
    k: int
    n_groups: int
    n_nonempty_groups: int
    fit: FitResult
    posteriors: tuple[GroupPosterior, ...]


@dataclass(frozen=True)
class PartialFitResult:
    """Shared-alpha fit from category-aggregated observations.

    ``group_log_evidence`` holds the per-group evidence contribution at the
    fitted (or conventional flat-case) alpha; the search itself only ever
    sees the informative groups (n >= 2 with at least two blocks), because
    every other contribution is an analytic constant in alpha.
    """

    k: int
    n_groups: int
    n_informative_groups: int
    fit: FitResult
    group_log_evidence: tuple[float, ...]


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


def fit_alpha_partial(
    k: int,
    observations: Sequence[Sequence[tuple[Sequence[int], int]]],
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_iter: int = DEFAULT_MAX_ITER,
) -> PartialFitResult:
    """Fit the shared symmetric alpha from category-aggregated observations.

    Reuses the same bracket search as :func:`fit_alpha` over the summed
    evidence, which is the sum of the (exact) merged Dirichlet-multinomial
    contributions. Rows that are provably constant in alpha -- empty rows,
    single draws (a lone draw hits block s with prior mass m_s/K), and fully
    merged rows (the single-block partition makes the draw certain) -- are
    excluded from the objective but kept in the reported contributions.

    When every observation is alpha-constant the total evidence is itself a
    proved constant; the function returns ``status='flat'`` with the
    conventional alpha = 1.0 WITHOUT running the search, rather than
    interpreting a stopped searcher as an identified parameter.
    """
    kk, validated = _validate_partial_observations(k, observations)
    gp = _reject_bool_int("grid_points", grid_points, 3)
    mi = _reject_bool_int("max_iter", max_iter, 1)

    informative = tuple(
        (blocks, counts)
        for blocks, counts in validated
        if not _row_is_alpha_constant(blocks, counts)
    )
    constant_contribution = math.fsum(
        _log_evidence_merged_row(1.0, kk, blocks, counts)
        for blocks, counts in validated
        if _row_is_alpha_constant(blocks, counts)
    )

    def evidence_at(alpha: float) -> float:
        return math.fsum(
            _log_evidence_merged_row(alpha, kk, blocks, counts)
            for blocks, counts in informative
        )

    if not informative:
        # Every contribution is an analytic constant: no information about
        # alpha exists in the batch. This is a proved flat case, not a
        # stopped/budget-exhausted search.
        fit = FitResult(
            alpha=1.0,
            log_evidence=constant_contribution,
            status=STATUS_FLAT,
            converged=True,
            grid_points=gp,
            max_iterations_used=0,
            candidates=(
                Candidate(
                    ALPHA_MIN, constant_contribution,
                    STATUS_LOWER_ENDPOINT, None, True, 0,
                ),
                Candidate(
                    ALPHA_MAX, constant_contribution,
                    STATUS_UPPER_ENDPOINT, None, True, 0,
                ),
            ),
            n_nonempty_groups=0,
        )
    else:
        searched = maximize_on_interval(
            evidence_at, grid_points=gp, max_iter=mi
        )
        fit = FitResult(
            alpha=searched.alpha,
            log_evidence=searched.log_evidence + constant_contribution,
            status=searched.status,
            converged=searched.converged,
            grid_points=gp,
            max_iterations_used=searched.max_iterations_used,
            candidates=tuple(
                Candidate(
                    c.alpha,
                    c.log_evidence + constant_contribution,
                    c.kind,
                    c.bracket,
                    c.converged,
                    c.iterations,
                )
                for c in searched.candidates
            ),
            n_nonempty_groups=len(informative),
        )

    contributions = tuple(
        _log_evidence_merged_row(fit.alpha, kk, blocks, counts)
        for blocks, counts in validated
    )
    return PartialFitResult(
        k=kk,
        n_groups=len(validated),
        n_informative_groups=len(informative),
        fit=fit,
        group_log_evidence=contributions,
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
