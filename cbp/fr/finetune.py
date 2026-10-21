from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import torch
from torch import nn


@dataclass
class FineTuneConfig:
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    identity_weight: float = 1.0
    embedding_preservation_weight: float = 1.0
    router_weight: float = 0.25
    utility_surrogate_weight: float = 0.0
    amp: bool = True
    grad_clip_norm: float | None = 5.0


@dataclass
class FineTuneResult:
    losses: list[dict[str, float]]
    steps: int


def fine_tune_backbone_with_preservation(
    backbone: nn.Module,
    identity_head: nn.Module,
    reference_backbone: nn.Module,
    train_loader: Iterable,
    config: FineTuneConfig,
    router_head: nn.Module | None = None,
    utility_surrogate: Callable[[torch.Tensor, dict], torch.Tensor] | None = None,
    device: str | torch.device | None = None,
) -> FineTuneResult:
    """Optional Stage-2 matched FR fine-tuning scaffold.

    Expected batch forms:

    - ``(images, identity_labels)``;
    - ``(images, identity_labels, teacher_cells)``;
    - dictionary with keys ``images``, ``identity_labels`` and optional
      ``teacher_cells`` plus any fields consumed by ``utility_surrogate``.

    The function combines identity classification, frozen-reference embedding
    preservation, optional router supervision and an optional differentiable
    utility/risk surrogate. It deliberately does not carry Stage-1 moments or
    certificates forward. After fine-tuning, hard matcher thresholds, states,
    moments, router and certificate must be rebuilt from scratch.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone.to(device).train()
    identity_head.to(device).train()
    if router_head is not None:
        router_head.to(device).train()
    reference_backbone.to(device).eval()
    for parameter in reference_backbone.parameters():
        parameter.requires_grad_(False)

    parameters = list(backbone.parameters()) + list(identity_head.parameters())
    if router_head is not None:
        parameters += list(router_head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in parameters if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(int(config.epochs)):
        totals = {
            "total": 0.0,
            "identity": 0.0,
            "preservation": 0.0,
            "router": 0.0,
            "utility": 0.0,
            "batches": 0.0,
        }
        for raw_batch in train_loader:
            batch = _normalize_batch(raw_batch)
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["identity_labels"].to(device, non_blocking=True).long()
            teacher_cells = batch.get("teacher_cells")
            if teacher_cells is not None:
                teacher_cells = teacher_cells.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                reference = nn.functional.normalize(reference_backbone(images), dim=-1)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                embedding = nn.functional.normalize(backbone(images), dim=-1)
                logits = identity_head(embedding)
                identity_loss = nn.functional.cross_entropy(logits, labels)
                preservation_loss = (1.0 - torch.sum(embedding * reference, dim=-1)).mean()

                router_loss = embedding.new_zeros(())
                if router_head is not None and teacher_cells is not None:
                    router_loss = nn.functional.cross_entropy(
                        router_head(embedding),
                        teacher_cells,
                    )

                utility_loss = embedding.new_zeros(())
                if utility_surrogate is not None:
                    utility_loss = utility_surrogate(embedding, batch)
                    if utility_loss.ndim != 0:
                        utility_loss = utility_loss.mean()

                total = (
                    config.identity_weight * identity_loss
                    + config.embedding_preservation_weight * preservation_loss
                    + config.router_weight * router_loss
                    + config.utility_surrogate_weight * utility_loss
                )

            scaler.scale(total).backward()
            if config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            totals["total"] += float(total.detach().cpu())
            totals["identity"] += float(identity_loss.detach().cpu())
            totals["preservation"] += float(preservation_loss.detach().cpu())
            totals["router"] += float(router_loss.detach().cpu())
            totals["utility"] += float(utility_loss.detach().cpu())
            totals["batches"] += 1.0
            step += 1
        denominator = max(totals.pop("batches"), 1.0)
        history.append({
            "epoch": float(epoch),
            **{name: value / denominator for name, value in totals.items()},
        })
    return FineTuneResult(losses=history, steps=step)


def _normalize_batch(raw_batch) -> dict:
    if isinstance(raw_batch, dict):
        if "images" not in raw_batch or "identity_labels" not in raw_batch:
            raise ValueError("batch dict requires images and identity_labels")
        return raw_batch
    if isinstance(raw_batch, (tuple, list)):
        if len(raw_batch) == 2:
            return {"images": raw_batch[0], "identity_labels": raw_batch[1]}
        if len(raw_batch) == 3:
            return {
                "images": raw_batch[0],
                "identity_labels": raw_batch[1],
                "teacher_cells": raw_batch[2],
            }
    raise TypeError("unsupported train-loader batch format")
