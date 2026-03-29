"""Hardware detection utilities."""

from __future__ import annotations

import torch


def detect_device() -> str:
    """Detect the best available device and print a summary.

    Priority: CUDA > MPS > CPU.

    Returns the device string to pass to model.to() and tensor operations.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"device: CUDA  —  {name}  ({vram:.0f} GB VRAM)")
        return "cuda"

    if torch.backends.mps.is_available():
        print("device: MPS  —  Apple Silicon")
        return "mps"

    print("device: CPU  —  no GPU detected, training will be slow")
    return "cpu"


# Recommended batch sizes per device.
# These are starting points — scale up if you have more VRAM/RAM.
BATCH_SIZE_BY_DEVICE: dict[str, int] = {
    "cuda": 256,  # typical mid-range GPU (8+ GB VRAM)
    "mps": 128,  # Apple Silicon (unified memory, no VRAM cap)
    "cpu": 32,  # CPU only — keep small to stay responsive
}
