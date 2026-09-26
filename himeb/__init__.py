"""himeb: hierarchical multinomial empirical Bayes (standard library only).

See ``himeb.core`` for the full contract.
"""

from .core import (
    ALPHA_MAX,
    ALPHA_MIN,
    MAX_GROUPS,
    MAX_K,
    Candidate,
    EBResult,
    FitResult,
    GroupPosterior,
    PartialEvidenceBreakdown,
    PartialFitResult,
    PartialObservation,
    PartitionBlock,
    analyze,
    fit_alpha,
    fit_alpha_partial,
    log_evidence,
    log_evidence_partial,
    log_evidence_one,
    maximize_on_interval,
    partial_evidence_breakdown,
    partial_observation,
    posterior_group,
)

__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "MAX_GROUPS",
    "MAX_K",
    "Candidate",
    "EBResult",
    "FitResult",
    "GroupPosterior",
    "PartialEvidenceBreakdown",
    "PartialFitResult",
    "PartialObservation",
    "PartitionBlock",
    "analyze",
    "fit_alpha",
    "fit_alpha_partial",
    "log_evidence",
    "log_evidence_partial",
    "log_evidence_one",
    "maximize_on_interval",
    "partial_evidence_breakdown",
    "partial_observation",
    "posterior_group",
]
