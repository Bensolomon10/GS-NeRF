"""
2D frequency maps for the depth/opacity oracle.

Follows Frequency-Aware Gaussian Splatting Decomposition (Lavi et al.,
arXiv:2503.21226): image-space Laplacian bands via repeated bilinear
downsample+upsample at full resolution (paper Eq. 6), then per-pixel
band residual energy as a scalar frequency map F(u,v).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_nchw(img: torch.Tensor) -> torch.Tensor:
    """[H,W] or [H,W,C] -> [1,C,H,W]."""
    if img.ndim == 2:
        return img[None, None]
    if img.ndim == 3:
        return img.permute(2, 0, 1).unsqueeze(0)
    raise ValueError(f"Expected [H,W] or [H,W,C], got {tuple(img.shape)}")


def lowpass_keep_resolution(img: torch.Tensor, times: int) -> torch.Tensor:
    """Paper Eq. (6): ((I) ↓2)^times then (↑2)^times back to HxW (bilinear)."""
    if times <= 0:
        return img
    x = _to_nchw(img.float())
    _, _, h, w = x.shape
    for _ in range(times):
        x = F.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )
    for _ in range(times):
        x = F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )
    if img.ndim == 2:
        return x[0, 0]
    return x[0].permute(1, 2, 0)


def laplacian_frequency_map(
    rgb: torch.Tensor,
    num_levels: int = 4,
) -> torch.Tensor:
    """
    Per-pixel frequency energy F(u,v) in [0, 1].

    Builds full-resolution low-pass pyramid levels I_k (k=1..L) as in
    FAGS §3.4–3.5, forms Laplacian residuals |I_{k+1}-I_k|, and sums them
    with weights increasing for higher bands. Normalized by the image max.
    """
    if num_levels < 2:
        raise ValueError("num_levels must be >= 2")
    gray = rgb.float().mean(dim=-1)  # [H, W]
    levels = [
        lowpass_keep_resolution(gray, num_levels - k)
        for k in range(1, num_levels + 1)
    ]
    # levels[0] = coarsest (I_1), levels[-1] ≈ original (I_L)
    energy = torch.zeros_like(gray)
    for k in range(num_levels - 1):
        band = (levels[k + 1] - levels[k]).abs()
        energy = energy + float(k + 1) * band
    denom = energy.max().clamp(min=1e-8)
    return (energy / denom).clamp(0.0, 1.0)


def depth_margin_from_frequency(
    depth_margin: float,
    frequency: torch.Tensor,
    freq_margin_scale: float = 1.0,
) -> torch.Tensor:
    """
    Per-ray margin: Δ = depth_margin * (1 + freq_margin_scale * F).

    High-frequency pixels get a wider [D−Δ, D+Δ] band.
    """
    return depth_margin * (1.0 + freq_margin_scale * frequency)


def render_step_size_from_frequency(
    render_step_size: float,
    frequency: torch.Tensor,
    freq_step_scale: float = 1.0,
    min_step_factor: float = 0.25,
) -> torch.Tensor:
    """
    Per-ray march step (LookCloser / FA-NeRF Adaptive Ray Marching).

    High-frequency content needs a smaller sampling interval so samples stay
    near the surface (Nyquist-style: denser samples where detail frequency is
    higher). With F in [0, 1]:

        δ(F) = render_step_size / (1 + freq_step_scale * F)
        δ(F) ≥ render_step_size * min_step_factor
    """
    step = render_step_size / (1.0 + freq_step_scale * frequency)
    return torch.clamp(step, min=render_step_size * min_step_factor)


def summarize_active_freq_sampling(
    frequency: torch.Tensor,
    depth_margins: torch.Tensor,
    step_sizes: torch.Tensor,
    active: torch.Tensor,
) -> dict:
    """
    Per-batch stats of F / Δ / δ on active rays (for training logs).

    Keys use short names for compact log lines:
      F_mean, F_max, d_mean/d_min/d_max (Δ), s_mean/s_min/s_max (δ),
      band_mean (= 2 * Δ mean, length of [D−Δ, D+Δ]).
    """
    empty = {
        "F_mean": 0.0,
        "F_max": 0.0,
        "d_mean": 0.0,
        "d_min": 0.0,
        "d_max": 0.0,
        "s_mean": 0.0,
        "s_min": 0.0,
        "s_max": 0.0,
        "band_mean": 0.0,
    }
    if active.dtype != torch.bool:
        active = active.bool()
    if not bool(active.any().item()):
        return empty
    f = frequency[active].float()
    d = depth_margins[active].float()
    s = step_sizes[active].float()
    return {
        "F_mean": float(f.mean().item()),
        "F_max": float(f.max().item()),
        "d_mean": float(d.mean().item()),
        "d_min": float(d.min().item()),
        "d_max": float(d.max().item()),
        "s_mean": float(s.mean().item()),
        "s_min": float(s.min().item()),
        "s_max": float(s.max().item()),
        "band_mean": float((2.0 * d).mean().item()),
    }


def format_freq_sampling_stats(stats: dict) -> str:
    """Compact string for the elapsed_time training log line."""
    return (
        f"F_mean={stats['F_mean']:.3f} F_max={stats['F_max']:.3f} | "
        f"Δ_mean={stats['d_mean']:.4f} Δ_min={stats['d_min']:.4f} "
        f"Δ_max={stats['d_max']:.4f} band_mean={stats['band_mean']:.4f} | "
        f"δ_mean={stats['s_mean']:.5f} δ_min={stats['s_min']:.5f} "
        f"δ_max={stats['s_max']:.5f}"
    )
