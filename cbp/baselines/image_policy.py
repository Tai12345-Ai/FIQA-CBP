from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import torch
from torch import nn

from cbp.fr.cosine import l2_normalize


class MultiContextImagePolicy(nn.Module):
    """Shared full-embedding neural gate with one head per context."""

    def __init__(
        self,
        input_dim: int,
        context_names: list[str],
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_names = list(context_names)
        self.hidden = tuple(int(x) for x in hidden)
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        d = self.input_dim
        for width in self.hidden:
            layers.extend([nn.Linear(d, width), nn.ReLU(), nn.Dropout(self.dropout)])
            d = width
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleDict({name: nn.Linear(d, 1) for name in self.context_names})

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "context_names": self.context_names,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
        }

    def forward(self, x: torch.Tensor, context: str) -> torch.Tensor:
        return torch.sigmoid(self.heads[context](self.trunk(x))).squeeze(-1)


@dataclass
class ImageContextStatistics:
    coverage_weights: torch.Tensor
    utility_weights: torch.Tensor
    pair_coefficients: torch.Tensor
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    degrees: torch.Tensor
    identity_budget: int
    n_identities: int
    n_pairs: int


@dataclass
class ImagePolicyTrainResult:
    model: MultiContextImagePolicy
    actions: dict[str, np.ndarray]
    objectives: dict[str, float]
    coverages: dict[str, float]
    risk_values: dict[str, float]
    feasible: dict[str, bool]
    loss_history: list[float]


def _identity_groups(identity_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.asarray(identity_ids).astype(str)
    unique = np.unique(ids)
    return unique, [np.flatnonzero(ids == identity) for identity in unique]


def build_image_context_statistics(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    threshold: float,
    alpha: float,
    identity_budget: int,
    max_identity_pairs: int | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> ImageContextStatistics:
    """Precompute differentiable linear identity-cluster statistics.

    The sparse pair-coefficient matrix maps image gate probabilities to the
    symmetrized directional excess-risk kernel for every sampled identity pair.
    """
    z = l2_normalize(embeddings)
    ids = np.asarray(identity_ids).astype(str)
    unique, groups = _identity_groups(ids)
    m = len(groups)
    if m < 4:
        raise ValueError("at least four identities are required")
    n_samples = len(z)

    coverage_weights = np.zeros(n_samples, dtype=np.float64)
    utility_weights = np.zeros(n_samples, dtype=np.float64)
    for idx in groups:
        coverage_weights[idx] = 1.0 / (m * len(idx))
        if len(idx) >= 2:
            match = (z[idx] @ z[idx].T) >= threshold
            np.fill_diagonal(match, False)
            per_probe = match.sum(axis=1) / (len(idx) - 1)
            utility_weights[idx] = per_probe / (m * len(idx))

    all_pairs = np.asarray(list(combinations(range(m), 2)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if max_identity_pairs is not None and len(all_pairs) > max_identity_pairs:
        selected = rng.choice(len(all_pairs), size=int(max_identity_pairs), replace=False)
        pairs = all_pairs[selected]
    else:
        pairs = all_pairs

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    degrees = np.zeros(m, dtype=np.float64)
    for row, (a, b) in enumerate(pairs):
        ia, ib = groups[a], groups[b]
        h = ((z[ia] @ z[ib].T) >= threshold).astype(np.float64) - float(alpha)
        coeff_a = 0.5 * h.mean(axis=1) / len(ia)
        coeff_b = 0.5 * h.mean(axis=0) / len(ib)
        rows.extend([row] * (len(ia) + len(ib)))
        cols.extend(ia.tolist() + ib.tolist())
        values.extend(coeff_a.tolist() + coeff_b.tolist())
        degrees[a] += 1
        degrees[b] += 1
    if np.any(degrees == 0):
        raise RuntimeError("pair subsampling left identities without partners")

    indices = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    vals = torch.as_tensor(values, dtype=torch.float64, device=device)
    with torch.sparse.check_sparse_tensor_invariants():
        pair_coefficients = torch.sparse_coo_tensor(
            indices,
            vals,
            size=(len(pairs), n_samples),
            dtype=torch.float64,
            device=device,
            check_invariants=True,
        ).coalesce()
    return ImageContextStatistics(
        coverage_weights=torch.as_tensor(coverage_weights, dtype=torch.float64, device=device),
        utility_weights=torch.as_tensor(utility_weights, dtype=torch.float64, device=device),
        pair_coefficients=pair_coefficients,
        pair_i=torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device),
        pair_j=torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device),
        degrees=torch.as_tensor(degrees, dtype=torch.float64, device=device),
        identity_budget=int(identity_budget),
        n_identities=m,
        n_pairs=len(pairs),
    )


def _objective_terms(
    action: torch.Tensor,
    statistics: ImageContextStatistics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action64 = action.to(torch.float64)
    coverage = torch.dot(statistics.coverage_weights, action64)
    utility = torch.dot(statistics.utility_weights, action64)
    pair_values = torch.sparse.mm(
        statistics.pair_coefficients,
        action64[:, None],
    ).squeeze(1)
    theta = pair_values.mean()

    sums = torch.zeros(
        statistics.n_identities,
        dtype=torch.float64,
        device=action.device,
    )
    sums = sums.index_add(0, statistics.pair_i, pair_values)
    sums = sums.index_add(0, statistics.pair_j, pair_values)
    influence = sums / statistics.degrees - theta
    influence = influence - torch.mean(influence)
    zeta1 = torch.mean(influence * influence)
    residual = (
        pair_values
        - theta
        - influence[statistics.pair_i]
        - influence[statistics.pair_j]
    )
    zeta2 = torch.mean(residual * residual)
    n = statistics.identity_budget
    variance = (4.0 / n) * zeta1 + (2.0 / (n * (n - 1))) * zeta2
    return utility, coverage, theta, variance.clamp_min(0.0)


def train_multi_context_image_policy(
    embeddings: np.ndarray,
    identity_ids: np.ndarray,
    thresholds: dict[str, float],
    alphas: dict[str, float],
    identity_budgets: dict[str, int],
    context_params: dict[str, dict[str, float]],
    objective: str = "evidence",
    hidden: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    epochs: int = 500,
    lr: float = 1e-3,
    penalty: float = 200.0,
    max_identity_pairs: int | None = None,
    seed: int = 0,
    device: str | None = None,
) -> ImagePolicyTrainResult:
    """Train the full-image matched neural comparator.

    ``objective='mean_risk'`` sets the evidence coefficient to zero while
    preserving architecture, data and optimization budget. ``objective='evidence'``
    uses the context-specific `rho` value.
    """
    if objective not in {"mean_risk", "evidence"}:
        raise ValueError(objective)
    names = list(context_params)
    if set(names) != set(thresholds) or set(names) != set(alphas) or set(names) != set(identity_budgets):
        raise ValueError("all context dictionaries must share keys")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    z = l2_normalize(embeddings).astype(np.float32)
    x = torch.as_tensor(z, dtype=torch.float32, device=device)
    model = MultiContextImagePolicy(
        input_dim=z.shape[1],
        context_names=names,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    statistics = {
        name: build_image_context_statistics(
            z,
            identity_ids,
            threshold=thresholds[name],
            alpha=alphas[name],
            identity_budget=identity_budgets[name],
            max_identity_pairs=max_identity_pairs,
            seed=seed + offset,
            device=device,
        )
        for offset, name in enumerate(names)
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=torch.float64, device=device)
        for name in names:
            action = model(x, name)
            utility, coverage, theta, variance = _objective_terms(action, statistics[name])
            params = context_params[name]
            rho = float(params["rho"]) if objective == "evidence" else 0.0
            risk = theta + rho * torch.sqrt(variance + 1e-15) + float(params.get("reserve", 0.0))
            violation = (
                torch.relu(risk) ** 2
                + torch.relu(torch.as_tensor(params["coverage_low"], dtype=torch.float64, device=device) - coverage) ** 2
                + torch.relu(coverage - torch.as_tensor(params["coverage_high"], dtype=torch.float64, device=device)) ** 2
            )
            loss = loss - utility + penalty * violation
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    actions: dict[str, np.ndarray] = {}
    objectives: dict[str, float] = {}
    coverages: dict[str, float] = {}
    risks: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    model.eval()
    with torch.no_grad():
        for name in names:
            action_t = model(x, name)
            utility, coverage, theta, variance = _objective_terms(action_t, statistics[name])
            params = context_params[name]
            rho = float(params["rho"]) if objective == "evidence" else 0.0
            risk = theta + rho * torch.sqrt(variance + 1e-15) + float(params.get("reserve", 0.0))
            action = action_t.detach().cpu().numpy().astype(np.float64)
            c = float(coverage.detach().cpu())
            r = float(risk.detach().cpu())
            actions[name] = action
            objectives[name] = float(utility.detach().cpu())
            coverages[name] = c
            risks[name] = r
            feasible[name] = bool(
                r <= 1e-6
                and c >= float(params["coverage_low"]) - 1e-6
                and c <= float(params["coverage_high"]) + 1e-6
            )
    return ImagePolicyTrainResult(
        model=model,
        actions=actions,
        objectives=objectives,
        coverages=coverages,
        risk_values=risks,
        feasible=feasible,
        loss_history=history,
    )


@torch.no_grad()
def predict_image_policy(
    model: MultiContextImagePolicy,
    embeddings: np.ndarray,
    context: str,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    z = l2_normalize(embeddings).astype(np.float32)
    x = torch.as_tensor(z)
    out = []
    model.eval()
    for start in range(0, len(x), batch_size):
        out.append(model(x[start:start + batch_size].to(device), context).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)
