"""Shared output types used across models and trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None  # load-balancing loss; None for Acts 1-2


@dataclass
class ShakespeareData:
    train: torch.Tensor
    val: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]
    vocab_size: int
