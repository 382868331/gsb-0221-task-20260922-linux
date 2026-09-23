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
    analyze,
    fit_alpha,
    log_evidence,
    log_evidence_one,
    maximize_on_interval,
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
    "analyze",
    "fit_alpha",
    "log_evidence",
    "log_evidence_one",
    "maximize_on_interval",
    "posterior_group",
]
