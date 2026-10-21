from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from cbp.fr.cosine import l2_normalize
from cbp.router.train import load_router, predict_router


@dataclass
class InferenceResult:
    cell: int
    keep_probability: float
    keep: bool
    mu_g: float | None
    mu_I: float | None
    evidence_response: float | None
    quality_score: float | None
    context: str


class CBPInferenceEngine:
    """Load a locked router and context policy tables for image-level gating."""

    def __init__(self, artifact_dir: str | Path, device: str = "cpu"):
        artifact_dir = Path(artifact_dir)
        self.router, self.router_metadata = load_router(artifact_dir / "router.pt", map_location=device)
        with open(artifact_dir / "policy.json", "r", encoding="utf-8") as f:
            self.policy = json.load(f)
        self.device = device

    def predict_cells(self, embeddings: np.ndarray) -> np.ndarray:
        z = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        return predict_router(self.router, z, device=self.device)

    def gate_probabilities(self, embeddings: np.ndarray, context: str) -> np.ndarray:
        if context not in self.policy["contexts"]:
            raise KeyError(f"context is not certified/supported: {context}")
        cells = self.predict_cells(embeddings)
        beta = np.asarray(self.policy["contexts"][context]["beta"], dtype=float)
        return beta[cells]

    def predict_embedding(
        self,
        embedding: np.ndarray,
        context: str,
        rng: np.random.Generator | None = None,
        deterministic_threshold: float | None = None,
    ) -> InferenceResult:
        if context not in self.policy["contexts"]:
            raise KeyError(f"context is not certified/supported: {context}")
        z = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        cell = int(predict_router(self.router, z, device=self.device)[0])
        table = self.policy["contexts"][context]
        beta = float(table["beta"][cell])
        if deterministic_threshold is None:
            rng = rng or np.random.default_rng()
            keep = bool(rng.random() <= beta)
        else:
            keep = bool(beta >= deterministic_threshold)
        return InferenceResult(
            cell=cell,
            keep_probability=beta,
            keep=keep,
            mu_g=_safe_index(table.get("mu_g"), cell),
            mu_I=_safe_index(table.get("mu_I"), cell),
            evidence_response=_safe_index(table.get("evidence_response"), cell),
            quality_score=_safe_index(table.get("quality_score"), cell),
            context=context,
        )


def _safe_index(values, index: int) -> float | None:
    if values is None:
        return None
    value = values[index]
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None
