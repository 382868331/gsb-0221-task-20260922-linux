"""dirichlet_eb: deterministic empirical Bayes for hierarchical multinomial counts.

Public API:
    validate_counts(...)
    log_evidence(alpha, counts, k=None)
    log_evidence_per_group(alpha, counts, k=None)
    compute_posteriors(alpha, counts, k=None)
    fit(counts, *, k=None, grid_points=201, tolerance=1e-10, max_iterations=200)

Result/error types:
    FitResult, GroupPosterior, CandidateMode, OptStatus
    EBMError, InputValidationError, IdentifiabilityError,
    SearchBudgetExhaustedError

Constants:
    ALPHA_MIN = 1e-4, ALPHA_MAX = 1e4
    MAX_CATEGORIES = 6, MAX_GROUPS = 40
"""

from .core import (
    ALPHA_MAX,
    ALPHA_MIN,
    MAX_CATEGORIES,
    MAX_GROUPS,
    CandidateMode,
    EBMError,
    FitResult,
    GroupPosterior,
    IdentifiabilityError,
    InputValidationError,
    OptStatus,
    SearchBudgetExhaustedError,
    compute_posteriors,
    fit,
    log_evidence,
    log_evidence_per_group,
    validate_counts,
)

__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "MAX_CATEGORIES",
    "MAX_GROUPS",
    "CandidateMode",
    "EBMError",
    "FitResult",
    "GroupPosterior",
    "IdentifiabilityError",
    "InputValidationError",
    "OptStatus",
    "SearchBudgetExhaustedError",
    "compute_posteriors",
    "fit",
    "log_evidence",
    "log_evidence_per_group",
    "validate_counts",
]
