"""Geometry-conditioned GroupNorm for FBPUNet.

This repo's gather tensors are ``(B, C, traces, time)`` — traces are height, time
is width. GeoNorm modulates **along traces** and broadcasts down time. The
geonorm.md snippet uses ``(B, C, time, traces)``; applying that layout here would
silently condition on the time axis.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_GEOM_EMB: Optional[torch.Tensor] = None


def group_count(num_channels: int, preferred: int = 8) -> int:
    groups = min(int(preferred), int(num_channels))
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


@contextmanager
def geom_embedding_context(emb: torch.Tensor) -> Iterator[None]:
    global _GEOM_EMB
    prev = _GEOM_EMB
    _GEOM_EMB = emb
    try:
        yield
    finally:
        _GEOM_EMB = prev


def current_geom_embedding() -> Optional[torch.Tensor]:
    return _GEOM_EMB


class GeomEncoder(nn.Module):
    """Shared MLP: ``(B, L, 2) → (B, L, h)``, computed once per forward pass."""

    def __init__(self, in_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        return self.net(geom)


class GeoNorm(nn.Module):
    """GroupNorm without affine + per-trace scale/shift from the geometry embedding."""

    def __init__(self, num_channels: int, geom_dim: int, groups: int = 8):
        super().__init__()
        self.num_channels = int(num_channels)
        self.geom_dim = int(geom_dim)
        self.norm = nn.GroupNorm(group_count(num_channels, groups), num_channels, affine=False)
        self.to_mod = nn.Linear(geom_dim, 2 * num_channels)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = current_geom_embedding()
        if emb is None:
            raise RuntimeError("GeoNorm.forward requires geom_embedding_context(...)")
        if emb.dim() != 3 or emb.shape[0] != x.shape[0]:
            raise ValueError(
                f"geom embedding shape {tuple(emb.shape)} incompatible with feature map {tuple(x.shape)}"
            )
        scale, shift = self.to_mod(emb).chunk(2, dim=-1)
        scale = scale.transpose(1, 2)
        shift = shift.transpose(1, 2)
        trace_count = int(x.shape[2])
        if scale.shape[-1] != trace_count:
            scale = F.adaptive_avg_pool1d(scale, trace_count)
            shift = F.adaptive_avg_pool1d(shift, trace_count)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return (1.0 + scale) * self.norm(x) + shift


def replace_norms_with_geonorm(module: nn.Module, geom_dim: int, groups: int = 8) -> int:
    """Swap 2D Batch/Instance/GroupNorm (affine) for GeoNorm. Returns replacement count."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, GeoNorm):
            continue
        if isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            setattr(module, name, GeoNorm(child.num_features, geom_dim, groups=groups))
            replaced += 1
            continue
        if isinstance(child, nn.GroupNorm) and child.affine:
            setattr(module, name, GeoNorm(child.num_channels, geom_dim, groups=groups))
            replaced += 1
            continue
        replaced += replace_norms_with_geonorm(child, geom_dim, groups=groups)
    return replaced
