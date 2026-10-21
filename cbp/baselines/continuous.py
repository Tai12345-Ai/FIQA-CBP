from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from cbp.statistics.moments import CellMoments


class ContinuousBottleneckPolicy(nn.Module):
    """Continuous low-rank comparator with a scalar acceptance head."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(x))).squeeze(-1)


@dataclass
class ContinuousPolicyResult:
    action: np.ndarray
    objective: float
    coverage: float
    risk_value: float
    feasible: bool
    model: ContinuousBottleneckPolicy
    loss_history: list[float]


def train_continuous_policy(
    features: np.ndarray,
    moments: CellMoments,
    rho: float,
    coverage_low: float,
    coverage_high: float,
    reserve: float = 0.0,
    bottleneck_dim: int = 8,
    epochs: int = 1000,
    lr: float = 1e-3,
    penalty: float = 200.0,
    seed: int = 0,
    device: str | None = None,
) -> ContinuousPolicyResult:
    """Train the same-objective continuous-bottleneck challenge baseline.

    ``features`` has one row per dense observable state represented by the cell
    moments. This uses a penalty surrogate rather than exact constrained
    optimization. Feasibility is recomputed with the true fitted objective and
    never inferred from training loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x_np = np.asarray(features, dtype=np.float32)
    if len(x_np) != len(moments.p):
        raise ValueError("one feature row is required per moment cell")
    p_np = np.asarray(moments.p, dtype=np.float64)
    u_np = np.asarray(moments.u, dtype=np.float64)
    ell_np = np.asarray(moments.ell, dtype=np.float64)
    B_np = np.asarray(moments.B, dtype=np.float64)

    x = torch.as_tensor(x_np, device=device)
    p_t = torch.as_tensor(p_np, dtype=torch.float64, device=device)
    u_t = torch.as_tensor(u_np, dtype=torch.float64, device=device)
    ell_t = torch.as_tensor(ell_np, dtype=torch.float64, device=device)
    B_t = torch.as_tensor(B_np, dtype=torch.float64, device=device)

    model = ContinuousBottleneckPolicy(x.shape[1], bottleneck_dim=bottleneck_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[float] = []

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        a = model(x).to(torch.float64)
        utility = torch.dot(u_t, a)
        coverage = torch.dot(p_t, a)
        variance = torch.dot(a, B_t @ a).clamp_min(0.0)
        risk = torch.dot(ell_t, a) + rho * torch.sqrt(variance + 1e-15) + reserve
        violation = (
            torch.relu(risk) ** 2
            + torch.relu(torch.as_tensor(coverage_low, dtype=torch.float64, device=device) - coverage) ** 2
            + torch.relu(coverage - torch.as_tensor(coverage_high, dtype=torch.float64, device=device)) ** 2
        )
        loss = -utility + penalty * violation
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        action = model(x).detach().cpu().numpy().astype(np.float64)
    coverage = float(p_np @ action)
    utility = float(u_np @ action)
    variance = float(max(action @ B_np @ action, 0.0))
    risk = float(ell_np @ action + rho * np.sqrt(variance) + reserve)
    feasible = (
        risk <= 1e-6
        and coverage >= coverage_low - 1e-6
        and coverage <= coverage_high + 1e-6
    )
    return ContinuousPolicyResult(
        action=action,
        objective=utility,
        coverage=coverage,
        risk_value=risk,
        feasible=feasible,
        model=model,
        loss_history=history,
    )


class MultiContextContinuousPolicy(nn.Module):
    """Shared continuous bottleneck with one acceptance head per context."""

    def __init__(
        self,
        input_dim: int,
        context_names: list[str],
        bottleneck_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.context_names = list(context_names)
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.bottleneck_dim, 1)
            for name in self.context_names
        })

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "context_names": self.context_names,
            "bottleneck_dim": self.bottleneck_dim,
            "hidden_dim": self.hidden_dim,
        }

    def forward(self, x: torch.Tensor, context: str) -> torch.Tensor:
        latent = self.encoder(x)
        return torch.sigmoid(self.heads[context](latent)).squeeze(-1)


@dataclass
class MultiContextContinuousResult:
    actions: dict[str, np.ndarray]
    objectives: dict[str, float]
    coverages: dict[str, float]
    risk_values: dict[str, float]
    feasible: dict[str, bool]
    model: MultiContextContinuousPolicy
    loss_history: list[float]


def train_multi_context_continuous_policy(
    features: np.ndarray,
    moments_by_context: dict[str, CellMoments],
    context_params: dict[str, dict[str, float]],
    bottleneck_dim: int = 8,
    epochs: int = 1000,
    lr: float = 1e-3,
    penalty: float = 200.0,
    seed: int = 0,
    device: str | None = None,
) -> MultiContextContinuousResult:
    """Train the decisive shared-continuous same-objective baseline."""
    if set(moments_by_context) != set(context_params):
        raise ValueError("moments_by_context and context_params must share keys")
    names = list(moments_by_context)
    x_np = np.asarray(features, dtype=np.float32)
    if any(len(m.p) != len(x_np) for m in moments_by_context.values()):
        raise ValueError("one feature row is required per dense state")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(x_np, device=device)
    model = MultiContextContinuousPolicy(
        input_dim=x.shape[1],
        context_names=names,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tensors = {}
    for name, moments in moments_by_context.items():
        tensors[name] = {
            "p": torch.as_tensor(moments.p, dtype=torch.float64, device=device),
            "u": torch.as_tensor(moments.u, dtype=torch.float64, device=device),
            "ell": torch.as_tensor(moments.ell, dtype=torch.float64, device=device),
            "B": torch.as_tensor(moments.B, dtype=torch.float64, device=device),
        }
    history = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        for name in names:
            t = tensors[name]
            params = context_params[name]
            action = model(x, name).to(torch.float64)
            utility = torch.dot(t["u"], action)
            coverage = torch.dot(t["p"], action)
            variance = torch.dot(action, t["B"] @ action).clamp_min(0.0)
            risk = (
                torch.dot(t["ell"], action)
                + float(params["rho"]) * torch.sqrt(variance + 1e-15)
                + float(params.get("reserve", 0.0))
            )
            violation = (
                torch.relu(risk) ** 2
                + torch.relu(torch.as_tensor(params["coverage_low"], dtype=torch.float64, device=device) - coverage) ** 2
                + torch.relu(coverage - torch.as_tensor(params["coverage_high"], dtype=torch.float64, device=device)) ** 2
            )
            total_loss = total_loss - utility + penalty * violation
        total_loss.backward()
        optimizer.step()
        history.append(float(total_loss.detach().cpu()))

    actions: dict[str, np.ndarray] = {}
    objectives: dict[str, float] = {}
    coverages: dict[str, float] = {}
    risks: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    with torch.no_grad():
        for name in names:
            action = model(x, name).detach().cpu().numpy().astype(np.float64)
            moments = moments_by_context[name]
            params = context_params[name]
            coverage = float(moments.p @ action)
            objective = float(moments.u @ action)
            variance = float(max(action @ moments.B @ action, 0.0))
            risk = float(
                moments.ell @ action
                + float(params["rho"]) * np.sqrt(variance)
                + float(params.get("reserve", 0.0))
            )
            ok = (
                risk <= 1e-6
                and coverage >= float(params["coverage_low"]) - 1e-6
                and coverage <= float(params["coverage_high"]) + 1e-6
            )
            actions[name] = action
            objectives[name] = objective
            coverages[name] = coverage
            risks[name] = risk
            feasible[name] = ok
    return MultiContextContinuousResult(
        actions=actions,
        objectives=objectives,
        coverages=coverages,
        risk_values=risks,
        feasible=feasible,
        model=model,
        loss_history=history,
    )
