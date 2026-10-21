from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import RouterMLP


@dataclass
class RouterTrainResult:
    model: RouterMLP
    train_losses: list[float]
    validation_losses: list[float]
    train_accuracy: float
    validation_accuracy: float | None
    best_epoch: int


def _weighted_ce(logits, labels, weights):
    loss_vec = nn.functional.cross_entropy(logits, labels, reduction="none")
    return (loss_vec * weights).sum() / weights.sum().clamp_min(1e-12)


def train_router(
    features: np.ndarray,
    labels: np.ndarray,
    n_cells: int,
    validation_features: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    sample_weights: np.ndarray | None = None,
    validation_weights: np.ndarray | None = None,
    hidden: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
    patience: int = 10,
) -> RouterTrainResult:
    """Fit a router to frozen teacher labels.

    Validation data should contain identities disjoint from the training data.
    The returned accuracy is only a diagnostic; downstream policy disagreement
    and deployed-map refitting are the scientifically relevant checks.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_np = np.asarray(features, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    if len(x_np) != len(y_np):
        raise ValueError("features and labels must align")
    if np.any(y_np < 0) or np.any(y_np >= n_cells):
        raise ValueError("router labels outside [0, n_cells)")
    w_np = np.ones(len(x_np), dtype=np.float32) if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)

    x = torch.as_tensor(x_np)
    y = torch.as_tensor(y_np)
    w = torch.as_tensor(w_np)
    ds = TensorDataset(x, y, w)
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterMLP(x.shape[1], n_cells, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    val_x = val_y = val_w = None
    if validation_features is not None:
        if validation_labels is None:
            raise ValueError("validation_labels required with validation_features")
        val_x = torch.as_tensor(np.asarray(validation_features, dtype=np.float32), device=device)
        val_y = torch.as_tensor(np.asarray(validation_labels, dtype=np.int64), device=device)
        if validation_weights is None:
            validation_weights = np.ones(len(val_x), dtype=np.float32)
        val_w = torch.as_tensor(np.asarray(validation_weights, dtype=np.float32), device=device)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_value = float("inf")
    bad_epochs = 0

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        mass = 0.0
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _weighted_ce(model(xb), yb, wb)
            loss.backward()
            opt.step()
            batch_mass = float(wb.sum().detach().cpu())
            total += float(loss.detach().cpu()) * batch_mass
            mass += batch_mass
        train_loss = total / max(mass, 1e-12)
        train_losses.append(train_loss)

        if val_x is not None:
            model.eval()
            with torch.no_grad():
                value = float(_weighted_ce(model(val_x), val_y, val_w).detach().cpu())
            val_losses.append(value)
        else:
            value = train_loss
            val_losses.append(float("nan"))

        if value < best_value - 1e-8:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    train_pred = predict_router(model, x_np, device=device)
    val_accuracy = None
    if validation_features is not None:
        val_pred = predict_router(model, validation_features, device=device)
        val_accuracy = float(np.mean(val_pred == np.asarray(validation_labels)))
    return RouterTrainResult(
        model=model,
        train_losses=train_losses,
        validation_losses=val_losses,
        train_accuracy=float(np.mean(train_pred == y_np)),
        validation_accuracy=val_accuracy,
        best_epoch=best_epoch,
    )


@torch.no_grad()
def predict_router(
    model: RouterMLP,
    features: np.ndarray,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    out = []
    model.eval()
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size].to(device))
        out.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


@torch.no_grad()
def predict_router_proba(
    model: RouterMLP,
    features: np.ndarray,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = torch.as_tensor(np.asarray(features, dtype=np.float32))
    out = []
    model.eval()
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.n_cells), dtype=np.float32)


def save_router(model: RouterMLP, path: str | Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config(),
            "metadata": metadata or {},
        },
        path,
    )


def load_router(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[RouterMLP, dict]:
    checkpoint = torch.load(path, map_location=map_location)
    model = RouterMLP(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint.get("metadata", {})
