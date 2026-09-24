"""himeb: hierarchical multinomial empirical Bayes (standard library only).

See ``himeb.core`` for the full contract.
"""

from .core import (
    ALPHA_MAX,
    ALPHA_MIN,
    MAX_GROUPS,
    MAX_K,
    Candidate,
    ConcentrationFit,
    EBConcentrationResult,
    EBResult,
    FitResult,
    GroupPosterior,
    analyze,
    analyze_concentrations,
    fit_alpha,
    fit_concentrations,
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
    "ConcentrationFit",
    "EBConcentrationResult",
    "EBResult",
    "FitResult",
    "GroupPosterior",
    "analyze",
    "analyze_concentrations",
    "fit_alpha",
    "fit_concentrations",
    "log_evidence",
    "log_evidence_one",
    "maximize_on_interval",
    "posterior_group",
]
