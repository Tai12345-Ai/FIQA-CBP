from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from .certificate import CertificateResult, certify_one_sided


@dataclass
class CandidateCertificate:
    name: str
    result: CertificateResult
    adjusted_delta: float


def certify_candidate_family(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    gates: Mapping[str, np.ndarray],
    threshold: float,
    target_fmr: float,
    coverage_target_low: float,
    coverage_target_high: float,
    family_delta: float = 0.05,
) -> list[CandidateCertificate]:
    """Simultaneously certify a predeclared finite candidate family.

    Uses a Bonferroni allocation. It is intentionally simple and valid; sharper
    candidate-selection methods may replace it only with an explicit proof.
    """
    if not gates:
        raise ValueError("gates cannot be empty")
    adjusted = family_delta / len(gates)
    out = []
    for name, gate in gates.items():
        result = certify_one_sided(
            embeddings=embeddings,
            identity_ids=identity_ids,
            gate=gate,
            threshold=threshold,
            target_fmr=target_fmr,
            coverage_target_low=coverage_target_low,
            coverage_target_high=coverage_target_high,
            delta=adjusted,
        )
        out.append(CandidateCertificate(name=name, result=result, adjusted_delta=adjusted))
    return out
