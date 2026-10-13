"""Small per-sample summaries of activations, used as slab textures and neuron values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

FACE_MAX = 32
IMAGE_MAX = 64
VECTOR_MAX = 16384


@dataclass
class Summary:
    """Downsampled views of one activation tensor, batch first, on the CPU.

    Every tensor is mapped to canonical per-sample dims ``(C, H, W)``: vectors ``(D,)``
    become ``(1, D, 1)``, tokens ``(N, D)`` become ``(D, N, 1)``, and extra leading
    dims are merged into ``C``.

    Attributes:
        dims: Canonical ``(C, H, W)`` of one sample.
        front: Mean over ``C``, shape ``(B, h, w)``.
        side: Mean over ``W``, shape ``(B, h, c)``.
        top: Mean over ``H``, shape ``(B, w, c)``.
        vector: The full flattened sample ``(B, numel)``, if ``numel <= VECTOR_MAX``.
        image: ``(B, C, h, w)`` with ``C`` in {1, 3}, for image-like inputs only.
    """

    dims: tuple[int, int, int]
    front: torch.Tensor
    side: torch.Tensor
    top: torch.Tensor
    vector: torch.Tensor | None = None
    image: torch.Tensor | None = None


def canonical(t: torch.Tensor) -> torch.Tensor:
    """Reshape a batch-first tensor to ``(B, C, H, W)`` following the rules in ``Summary``."""
    b, rest = t.shape[0], tuple(t.shape[1:])
    if len(rest) == 0:
        return t.reshape(b, 1, 1, 1)
    if len(rest) == 1:
        return t.reshape(b, 1, rest[0], 1)
    if len(rest) == 2:
        return t.transpose(1, 2).reshape(b, rest[1], rest[0], 1)
    return t.reshape(b, math.prod(rest[:-2]), rest[-2], rest[-1])


def _pool(x: torch.Tensor, limit: int) -> torch.Tensor:
    """Average-pool a ``(B, h, w)`` tensor so neither spatial side exceeds ``limit``."""
    h, w = x.shape[-2:]
    if h <= limit and w <= limit:
        return x
    return F.adaptive_avg_pool2d(x[:, None], (min(h, limit), min(w, limit)))[:, 0]


def summarize(t: torch.Tensor, image: bool = False) -> Summary:
    """Summarize a batch-first activation tensor.

    Args:
        t: A tensor whose first dim is the batch.
        image: Also keep a downsampled copy if the sample looks like an image
            (``(C, H, W)`` with ``C`` in {1, 3}, or ``(H, W)``).

    Returns:
        A ``Summary`` on the CPU.
    """
    t = t.detach()
    if t.ndim == 0:
        t = t.reshape(1)
    t = t.float() if not t.is_complex() else t.abs().float()
    x = canonical(t)
    _, c, h, w = x.shape
    summary = Summary(
        dims=(c, h, w),
        front=_pool(x.mean(dim=1), FACE_MAX).cpu(),
        side=_pool(x.mean(dim=3).transpose(1, 2), FACE_MAX).cpu(),
        top=_pool(x.mean(dim=2).transpose(1, 2), FACE_MAX).cpu(),
    )
    if t[0].numel() <= VECTOR_MAX:
        summary.vector = t.reshape(t.shape[0], -1).cpu()
    if image:
        img = t[:, None] if t.ndim == 3 else t
        if img.ndim == 4 and img.shape[1] in (1, 3) and min(img.shape[-2:]) >= 4:
            hh, ww = img.shape[-2:]
            scale = min(1.0, IMAGE_MAX / max(hh, ww))
            size = (max(1, round(hh * scale)), max(1, round(ww * scale)))
            if size != (hh, ww):
                img = F.interpolate(img, size=size, mode="area")
            summary.image = img.cpu()
    return summary
