"""himeb: hierarchical multinomial empirical Bayes (standard library only).

See ``himeb.core`` for the full contract.
"""

from .core import (
    ALPHA_MAX,
    ALPHA_MIN,
    MAX_PREDICTIVE_M,
    MAX_GROUPS,
    MAX_K,
    Candidate,
    EBResult,
    FitResult,
    GroupPosterior,
    PredictiveResult,
    analyze,
    fit_alpha,
    log_evidence,
    log_evidence_one,
    maximize_on_interval,
    posterior_group,
)

from .predictive import posterior_predictive

__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "MAX_PREDICTIVE_M",
    "MAX_GROUPS",
    "MAX_K",
    "Candidate",
    "EBResult",
    "FitResult",
    "GroupPosterior",
    "PredictiveResult",
    "analyze",
    "fit_alpha",
    "log_evidence",
    "log_evidence_one",
    "maximize_on_interval",
    "posterior_predictive",
    "posterior_group",
]
