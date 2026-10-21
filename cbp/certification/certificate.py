from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .bounds import hoeffding_mean_radius, hoeffding_u2_radius


@dataclass
class CertificateResult:
    status: str
    n_identities: int
    coverage_hat: float
    coverage_low: float
    coverage_high: float
    genuine_yield_hat: float
    genuine_yield_low: float
    impostor_numerator_hat: float
    impostor_numerator_up: float
    fmr_upper: float
    target_fmr: float
    delta: float

    def to_dict(self):
        return asdict(self)


def _identity_groups(identity_ids: np.ndarray):
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids)
    return unique, [np.flatnonzero(ids == x) for x in unique]


def certify_one_sided(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    gate: np.ndarray,
    threshold: float,
    target_fmr: float,
    coverage_target_low: float,
    coverage_target_high: float,
    delta: float = 0.05,
) -> CertificateResult:
    """Independent identity-cluster certificate for one-sided probe filtering.

    The pair kernel averages all cross-capture comparisons between two identity
    clusters and symmetrizes which identity acts as probe. Therefore the
    independent units are identities, not image pairs.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    ids = np.asarray(identity_ids).astype(str)
    gate = np.clip(np.asarray(gate, dtype=float), 0.0, 1.0)
    unique, groups = _identity_groups(ids)
    n = len(unique)
    if n < 4:
        raise ValueError("certification needs at least 4 independent identities")

    coverage_by_id = np.array([float(np.mean(gate[idx])) for idx in groups])

    genuine_by_id = []
    for idx in groups:
        if len(idx) < 2:
            genuine_by_id.append(0.0)
            continue
        sim = z[idx] @ z[idx].T
        accept = sim >= threshold
        np.fill_diagonal(accept, False)
        per_probe = accept.sum(axis=1) / (len(idx) - 1)
        genuine_by_id.append(float(np.mean(gate[idx] * per_probe)))
    genuine_by_id = np.asarray(genuine_by_id)

    pair_values = []
    for a in range(n):
        ia = groups[a]
        for b in range(a + 1, n):
            ib = groups[b]
            match = (z[ia] @ z[ib].T) >= threshold
            a_probe = np.mean(gate[ia, None] * match)
            b_probe = np.mean(gate[ib, None] * match.T)
            pair_values.append(0.5 * (a_probe + b_probe))
    pair_values = np.asarray(pair_values, dtype=float)

    # Split failure budget across coverage lower/upper, utility lower, risk upper.
    d = delta / 4.0
    eps_mean = hoeffding_mean_radius(n, d)
    eps_u2 = hoeffding_u2_radius(n, d)

    c_hat = float(np.mean(coverage_by_id))
    c_low = max(0.0, c_hat - eps_mean)
    c_up = min(1.0, c_hat + eps_mean)
    u_hat = float(np.mean(genuine_by_id))
    u_low = max(0.0, u_hat - eps_mean)
    j_hat = float(np.mean(pair_values))
    j_up = min(1.0, j_hat + eps_u2)
    fmr_up = j_up / c_low if c_low > 0 else float("inf")

    passed = (
        fmr_up <= target_fmr
        and c_low >= coverage_target_low
        and c_up <= coverage_target_high
    )
    return CertificateResult(
        status="PASS" if passed else "INSUFFICIENT_EVIDENCE",
        n_identities=n,
        coverage_hat=c_hat,
        coverage_low=c_low,
        coverage_high=c_up,
        genuine_yield_hat=u_hat,
        genuine_yield_low=u_low,
        impostor_numerator_hat=j_hat,
        impostor_numerator_up=j_up,
        fmr_upper=fmr_up,
        target_fmr=float(target_fmr),
        delta=float(delta),
    )
