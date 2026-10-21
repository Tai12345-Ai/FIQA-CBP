from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ContextConfig:
    name: str
    target_fmr: float
    coverage_low: float = 0.5
    coverage_high: float = 1.0
    rho: float = 1.0
    reserve: float = 0.0
    identity_budget: int | None = None
    max_impostor_pairs_for_threshold: int = 1_000_000


@dataclass
class SplitConfig:
    repr: float = 0.30
    fit: float = 0.25
    dev: float = 0.15
    cal: float = 0.15
    test: float = 0.15

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    seed: int = 20261021
    embedding_npz: str | None = None
    output_dir: str = "outputs/run"
    device: str | None = None

    # Identity protocol.
    splits: SplitConfig = field(default_factory=SplitConfig)
    min_captures_per_identity: int = 2
    max_identities_per_split: int | None = None

    # Finite observable state construction.
    n_states: int = 128
    k_candidates: list[int] = field(default_factory=lambda: [4, 8, 16])
    spectral_dim: int = 8
    max_fit_identity_pairs: int | None = None

    # Operating contexts.
    contexts: list[ContextConfig] = field(
        default_factory=lambda: [
            ContextConfig(name="fmr_1e-2", target_fmr=1e-2, coverage_low=0.5),
            ContextConfig(name="fmr_1e-3", target_fmr=1e-3, coverage_low=0.25),
        ]
    )

    # Router induction.
    router_epochs: int = 80
    router_batch_size: int = 256
    router_lr: float = 1e-3
    router_hidden: list[int] = field(default_factory=lambda: [256, 128])
    router_dropout: float = 0.1
    router_patience: int = 15

    # Diagnostics and baselines.
    phenomenon_permutations: int = 1000
    phenomenon_match_quantile: float = 0.25
    phenomenon_min_gap_ratio: float = 0.25
    phenomenon_max_mean_match_distance: float = 0.75
    phenomenon_min_pairs: int = 10
    bootstrap_repetitions: int = 1000
    run_continuous_baseline: bool = True
    continuous_bottleneck_dim: int = 8
    continuous_epochs: int = 1000
    run_image_neural_baseline: bool = True
    image_policy_epochs: int = 500
    image_policy_lr: float = 1e-3
    image_policy_hidden: list[int] = field(default_factory=lambda: [256, 128])
    image_policy_dropout: float = 0.1
    image_policy_max_identity_pairs: int | None = None

    # Independent certificate.
    certification_delta: float = 0.05
    certify_candidate_family: bool = False
    gate_numerical_tolerance: float = 1e-8
    gate_decision_min_yield_gain: float = 0.0

    @staticmethod
    def from_yaml(path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        split_raw = raw.pop("splits", {}) or {}
        ctx_raw = raw.pop("contexts", None)
        cfg = ExperimentConfig(**raw, splits=SplitConfig(**split_raw))
        if ctx_raw is not None:
            cfg.contexts = [ContextConfig(**x) for x in ctx_raw]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.embedding_npz is None:
            raise ValueError("embedding_npz is required")
        fractions = self.splits.to_dict()
        if any(v <= 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-8:
            raise ValueError("split fractions must be positive and sum to one")
        if self.n_states < 4:
            raise ValueError("n_states must be >=4")
        if not self.k_candidates or any(k < 2 for k in self.k_candidates):
            raise ValueError("k_candidates must contain integers >=2")
        if any(not (0 < c.target_fmr < 1) for c in self.contexts):
            raise ValueError("all target_fmr values must lie in (0,1)")
        if any(not (0 <= c.coverage_low <= c.coverage_high <= 1) for c in self.contexts):
            raise ValueError("invalid coverage band")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
