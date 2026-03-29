"""Training loop — act-agnostic, works with any model returning ModelOutput."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.config import BaseConfig
from moe.data import get_batch
from moe.types import ModelOutput, ShakespeareData


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    dataset: ShakespeareData,
    cfg: BaseConfig,
) -> dict[str, float]:
    model.train(False)
    results = {}
    for split, data in [("train", dataset.train), ("val", dataset.val)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg)
            out: ModelOutput = model(xb, yb)
            losses[k] = out.loss.item()  # type: ignore[union-attr]
        results[split] = losses.mean().item()
    model.train(True)
    return results


def train(cfg: BaseConfig, model: nn.Module, dataset: ShakespeareData) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0:
            losses = compute_loss(model, dataset, cfg)
            print(
                f"step {step:4d}"
                f"  |  train loss {losses['train']:.4f}"
                f"  |  val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(dataset.train, cfg)
        out: ModelOutput = model(xb, yb)

        total_loss = out.loss
        if out.aux_loss is not None:
            total_loss = total_loss + cfg.lb_coeff * out.aux_loss  # type: ignore[operator]

        optimizer.zero_grad()
        total_loss.backward()  # type: ignore[union-attr]
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    losses = compute_loss(model, dataset, cfg)
    print(
        f"step {cfg.max_iters:4d}"
        f"  |  train loss {losses['train']:.4f}"
        f"  |  val loss {losses['val']:.4f}"
    )
    print("\n--- sample ---")
    print(sample(model, dataset.itos, cfg))


def sample(
    model: nn.Module,
    itos: dict[int, str],
    cfg: BaseConfig,
    n_tokens: int = 300,
) -> str:
    model.train(False)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        for _ in range(n_tokens):
            out: ModelOutput = model(ctx[:, -cfg.block_size :])
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat([ctx, next_tok], dim=1)
    return "".join(itos[i] for i in ctx[0].tolist())
