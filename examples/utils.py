"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
import random
import time
from typing import Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    accumulate_along_rays_,
    render_weight_from_density,
    rendering,
)

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

# Opt-in lightweight timing for render_image_with_occgrid().
# Enable via: export NERFACC_PROFILE_RENDER=1
# During training prints every 50 calls; always stores last breakdown for callers.
_RENDER_PROFILE_ENABLED = os.environ.get("NERFACC_PROFILE_RENDER", "0") == "1"
_RENDER_PROFILE_CALL = 0
_RENDER_PROFILE_EVERY = 50
_LAST_RENDER_PROFILE = {}


def get_last_render_profile():
    """Return timing dict from the most recent profiled render_image_with_occgrid call."""
    return dict(_LAST_RENDER_PROFILE)


def format_last_render_profile():
    """Human-readable breakdown of the last profiled render (empty string if none)."""
    p = _LAST_RENDER_PROFILE
    if not p:
        return ""
    return (
        f"n_samples={p['n_samples']} | "
        f"sampling_s={p['sampling_s']:.3f}"
        f"(density_mlp={p['density_mlp_s']:.3f}, march={p['sampling_march_s']:.3f}) | "
        f"rendering_s={p['rendering_s']:.3f}"
        f"(rgb_mlp={p['rgb_mlp_s']:.3f}, integrate={p['integrate_s']:.3f}) | "
        f"occgrid_total_s={p['total_s']:.3f}"
    )


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # optional per-ray bounds (e.g. depth-oracle band)
    t_min: Optional[torch.Tensor] = None,
    t_max: Optional[torch.Tensor] = None,
    # If True, skip density early-stop in estimator.sampling (useful with depth oracle).
    skip_sigma_fn: bool = False,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    global _RENDER_PROFILE_CALL, _LAST_RENDER_PROFILE
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
        if t_min is not None:
            t_min = t_min.reshape(num_rays)
        if t_max is not None:
            t_max = t_max.reshape(num_rays)
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )

    # When enabled: always time this call (so eval can attach breakdown to render_time).
    # Print from here only every N training calls to avoid spam.
    do_profile = _RENDER_PROFILE_ENABLED
    do_print = (
        do_profile
        and radiance_field.training
        and (_RENDER_PROFILE_CALL % _RENDER_PROFILE_EVERY == 0)
    )
    _RENDER_PROFILE_CALL += 1
    profile_sampling_s = 0.0
    profile_rendering_s = 0.0
    profile_density_mlp_s = 0.0
    profile_rgb_mlp_s = 0.0
    profile_chunks = 0
    profile_num_rendering_samples = 0
    profile_total_start = None
    if do_profile:
        _cuda_sync()
        profile_total_start = time.perf_counter()

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_t_min = None if t_min is None else t_min[i : i + chunk]
        chunk_t_max = None if t_max is None else t_max[i : i + chunk]

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.viewdirs

        def sigma_fn(t_starts, t_ends, ray_indices):
            nonlocal profile_density_mlp_s
            if t_starts.shape[0] == 0:
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                if do_profile:
                    _cuda_sync()
                    t_mlp = time.perf_counter()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    sigmas = radiance_field.query_density(positions, t)
                else:
                    sigmas = radiance_field.query_density(positions)
                if do_profile:
                    _cuda_sync()
                    profile_density_mlp_s += time.perf_counter() - t_mlp
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            nonlocal profile_rgb_mlp_s
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                if do_profile:
                    _cuda_sync()
                    t_mlp = time.perf_counter()
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = (
                    t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                )
                if timestamps is not None:
                    # dnerf
                    t = (
                        timestamps[ray_indices]
                        if radiance_field.training
                        else timestamps.expand_as(positions[:, :1])
                    )
                    rgbs, sigmas = radiance_field(positions, t, t_dirs)
                else:
                    rgbs, sigmas = radiance_field(positions, t_dirs)
                if do_profile:
                    _cuda_sync()
                    profile_rgb_mlp_s += time.perf_counter() - t_mlp
            return rgbs, sigmas.squeeze(-1)

        if do_profile:
            _cuda_sync()
            t_sampling_start = time.perf_counter()
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=None if skip_sigma_fn else sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            t_min=chunk_t_min,
            t_max=chunk_t_max,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        if do_profile:
            _cuda_sync()
            profile_sampling_s += time.perf_counter() - t_sampling_start

        if do_profile:
            _cuda_sync()
            t_rendering_start = time.perf_counter()
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        if do_profile:
            _cuda_sync()
            profile_rendering_s += time.perf_counter() - t_rendering_start
            profile_chunks += 1
            profile_num_rendering_samples += len(t_starts)

        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    if do_profile and profile_total_start is not None:
        _cuda_sync()
        profile_total_s = time.perf_counter() - profile_total_start
        n_samples_total = sum(n_rendering_samples)
        sampling_march_s = max(profile_sampling_s - profile_density_mlp_s, 0.0)
        integrate_s = max(profile_rendering_s - profile_rgb_mlp_s, 0.0)
        _LAST_RENDER_PROFILE = {
            "call": _RENDER_PROFILE_CALL - 1,
            "num_rays": num_rays,
            "n_samples": n_samples_total,
            "chunks": profile_chunks,
            "sampling_s": profile_sampling_s,
            "density_mlp_s": profile_density_mlp_s,
            "sampling_march_s": sampling_march_s,
            "rendering_s": profile_rendering_s,
            "rgb_mlp_s": profile_rgb_mlp_s,
            "integrate_s": integrate_s,
            "total_s": profile_total_s,
        }

    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )


def render_image_with_propnet(
    # scene
    radiance_field: torch.nn.Module,
    proposal_networks: Sequence[torch.nn.Module],
    estimator: PropNetEstimator,
    rays: Rays,
    # rendering options
    num_samples: int,
    num_samples_per_prop: Sequence[int],
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    sampling_type: Literal["uniform", "lindisp"] = "lindisp",
    opaque_bkgd: bool = True,
    render_bkgd: Optional[torch.Tensor] = None,
    # train options
    proposal_requires_grad: bool = False,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sigmas = proposal_network(positions)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[..., None, :]
        t_dirs = chunk_rays.viewdirs[..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rgb, sigmas = radiance_field(positions, t_dirs)
        if opaque_bkgd:
            sigmas[..., -1, :] = torch.inf
        return rgb, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        t_starts, t_ends = estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            prop_samples=num_samples_per_prop,
            num_samples=num_samples,
            n_rays=chunk_rays.origins.shape[0],
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth]
        results.append(chunk_results)

    colors, opacities, depths = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            torch.Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        extras,
    )


@torch.no_grad()
def render_image_with_occgrid_test(
    max_samples: int,
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    early_stop_eps: float = 1e-4,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = rays.origins[ray_indices]
        t_dirs = rays.viewdirs[ray_indices]
        positions = (
            t_origins + t_dirs * (t_starts[:, None] + t_ends[:, None]) / 2.0
        )
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    device = rays.origins.device
    opacity = torch.zeros(num_rays, 1, device=device)
    depth = torch.zeros(num_rays, 1, device=device)
    rgb = torch.zeros(num_rays, 3, device=device)

    ray_mask = torch.ones(num_rays, device=device).bool()

    # 1 for synthetic scenes, 4 for real scenes
    min_samples = 1 if cone_angle == 0 else 4

    iter_samples = total_samples = 0

    rays_o = rays.origins
    rays_d = rays.viewdirs

    near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
    far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

    t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, estimator.aabbs)

    n_grids = estimator.binaries.size(0)

    if n_grids > 1:
        t_sorted, t_indices = torch.sort(torch.cat([t_mins, t_maxs], -1), -1)
    else:
        t_sorted = torch.cat([t_mins, t_maxs], -1)
        t_indices = torch.arange(
            0, n_grids * 2, device=t_mins.device, dtype=torch.int64
        ).expand(num_rays, n_grids * 2)

    opc_thre = 1 - early_stop_eps

    while iter_samples < max_samples:
        n_alive = ray_mask.sum().item()
        if n_alive == 0:
            break

        # the number of samples to add on each ray
        n_samples = max(min(num_rays // n_alive, 64), min_samples)
        iter_samples += n_samples

        # ray marching
        (intervals, samples, termination_planes) = traverse_grids(
            # rays
            rays_o,  # [n_rays, 3]
            rays_d,  # [n_rays, 3]
            # grids
            estimator.binaries,  # [m, resx, resy, resz]
            estimator.aabbs,  # [m, 6]
            # options
            near_planes,  # [n_rays]
            far_planes,  # [n_rays]
            render_step_size,
            cone_angle,
            n_samples,
            True,
            ray_mask,
            # pre-compute intersections
            t_sorted,  # [n_rays, m*2]
            t_indices,  # [n_rays, m*2]
            hits,  # [n_rays, m]
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices[samples.is_valid]
        packed_info = samples.packed_info

        # get rgb and sigma from radiance field
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # volume rendering using native cuda scan
        weights, _, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=num_rays,
            prefix_trans=1 - opacity[ray_indices].squeeze(-1),
        )
        if alpha_thre > 0:
            vis_mask = alphas >= alpha_thre
            ray_indices, rgbs, weights, t_starts, t_ends = (
                ray_indices[vis_mask],
                rgbs[vis_mask],
                weights[vis_mask],
                t_starts[vis_mask],
                t_ends[vis_mask],
            )

        accumulate_along_rays_(
            weights,
            values=rgbs,
            ray_indices=ray_indices,
            outputs=rgb,
        )
        accumulate_along_rays_(
            weights,
            values=None,
            ray_indices=ray_indices,
            outputs=opacity,
        )
        accumulate_along_rays_(
            weights,
            values=(t_starts + t_ends)[..., None] / 2.0,
            ray_indices=ray_indices,
            outputs=depth,
        )
        # update near_planes using termination planes
        near_planes = termination_planes
        # update rays status
        ray_mask = torch.logical_and(
            # early stopping
            opacity.view(-1) <= opc_thre,
            # remove rays that have reached the far plane
            packed_info[:, 1] == n_samples,
        )
        total_samples += ray_indices.shape[0]

    rgb = rgb + render_bkgd * (1.0 - opacity)
    depth = depth / opacity.clamp_min(torch.finfo(rgbs.dtype).eps)

    return (
        rgb.view((*rays_shape[:-1], -1)),
        opacity.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        total_samples,
    )
