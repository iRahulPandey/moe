"""Data pipeline — shared across all acts."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

from moe.config import BaseConfig
from moe.types import ShakespeareData

_CACHE = Path(__file__).parent.parent.parent / ".cache" / "shakespeare.txt"
_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _fetch_text() -> str:
    if _CACHE.exists():
        return _CACHE.read_text(encoding="utf-8")
    print("downloading tiny_shakespeare (~1MB)...")
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_URL, _CACHE)
    return _CACHE.read_text(encoding="utf-8")


def load_shakespeare() -> ShakespeareData:
    text = _fetch_text()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))

    print(
        f"vocab size: {vocab_size}"
        f"  |  train tokens: {len(data[:split]):,}"
        f"  |  val tokens: {len(data[split:]):,}"
    )
    return ShakespeareData(
        train=data[:split],
        val=data[split:],
        stoi=stoi,
        itos=itos,
        vocab_size=vocab_size,
    )


def get_batch(data: torch.Tensor, cfg: BaseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)
