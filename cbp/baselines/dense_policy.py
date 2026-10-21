from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


class DenseGate(nn.Module):
    def __init__(self, input_dim: int, hidden=(128, 64)):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


@dataclass
class DensePolicyResult:
    model: DenseGate
    beta: np.ndarray
    history: list[float]


def train_dense_state_policy(
    features: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    q_matrix: np.ndarray,
    rho: float,
    reserve: float,
    coverage_low: float,
    coverage_high: float,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    evidence_sensitive: bool = True,
    penalty: float = 100.0,
    device: str | None = None,
) -> DensePolicyResult:
    """Full-feature differentiable baseline over the finite construction states.

    It receives the same state features as CBP but is not restricted to hard
    cells. The evidence-sensitive arm optimizes the same Q_n seminorm; the
    mean-risk arm sets rho=0 while keeping all other inputs/training budget.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    p_t = torch.as_tensor(np.asarray(p), dtype=torch.float32, device=device)
    g_t = torch.as_tensor(np.asarray(g), dtype=torch.float32, device=device)
    r_t = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device)
    q_t = torch.as_tensor(np.asarray(q_matrix), dtype=torch.float32, device=device)
    model = DenseGate(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    hist = []
    effective_rho = float(rho) if evidence_sensitive else 0.0

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        beta = model(x)
        utility = torch.sum(p_t * beta * g_t)
        excess = torch.sum(p_t * beta * r_t)
        qb = q_t @ beta
        variance = torch.sum(p_t * beta * qb).clamp_min(0.0)
        risk = excess + effective_rho * torch.sqrt(variance + 1e-12) + reserve
        coverage = torch.sum(p_t * beta)
        violation = (
            torch.relu(risk) ** 2
            + torch.relu(coverage_low - coverage) ** 2
            + torch.relu(coverage - coverage_high) ** 2
        )
        loss = -utility + penalty * violation
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu()))

    with torch.no_grad():
        beta = model(x).cpu().numpy()
    return DensePolicyResult(model=model, beta=beta, history=hist)


@torch.no_grad()
def predict_dense_gate(model: DenseGate, features: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    model.eval()
    return model(x).detach().cpu().numpy()
