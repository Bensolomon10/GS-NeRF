"""
Visualize oracle depth/opacity maps and optional RGB renders from evenly
spaced viewpoints (NeRF-Synthetic or Mip-NeRF 360).

Outputs under experiments/ (default: experiments/oracles/<scene>/vis_<split>/).
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
import matplotlib.cm as cm
import numpy as np
import torch
import tqdm

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def colorize_scalar(
    values: np.ndarray,
    mask: np.ndarray | None,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Map a 2D scalar field to RGB uint8 with an optional foreground mask."""
    vals = values.astype(np.float32).copy()
    if mask is not None:
        fg = mask.astype(bool)
        if fg.any():
            if vmin is None:
                vmin = float(np.percentile(vals[fg], 2))
            if vmax is None:
                vmax = float(np.percentile(vals[fg], 98))
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
    else:
        if vmin is None:
            vmin = float(np.percentile(vals, 2))
        if vmax is None:
            vmax = float(np.percentile(vals, 98))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    if hasattr(cm, "colormaps"):
        cmap = cm.colormaps.get_cmap(cmap_name)
    else:
        cmap = cm.get_cmap(cmap_name)
    rgb = np.asarray(cmap(norm))[..., :3]  # drop alpha
    if mask is not None:
        rgb = np.where(mask[..., None].astype(bool), rgb, bg_rgb)
    return (rgb * 255.0).astype(np.uint8)


def pick_view_indices(n_total: int, num_views: int) -> np.ndarray:
    """Evenly spaced indices covering the full camera set (~360°)."""
    if num_views >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.linspace(0, n_total - 1, num_views, dtype=np.float64).round().astype(
        np.int64
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root (default: data/nerf_synthetic or data/360_v2 by scene)",
    )
    parser.add_argument(
        "--oracle_path",
        type=str,
        default=None,
        help="Baked depth/opacity .pt (default: experiments/oracles/<scene>/test_depth_opacity.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "trainval", "test"],
        help="Dataset split used for RGB renders and must match oracle N views",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional NeRF checkpoint for RGB multi-view (MLP or NGP Occ)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=12,
        help="Number of evenly spaced views around the object",
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.01,
        help="Foreground mask threshold for depth coloring",
    )
    parser.add_argument(
        "--depth_cmap",
        type=str,
        default="turbo",
        help="Matplotlib colormap for depth",
    )
    parser.add_argument(
        "--opacity_cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for opacity",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/oracles/<scene>/vis_<split>)",
    )
    parser.add_argument("--test_chunk_size", type=int, default=8192)
    parser.add_argument(
        "--make_gif",
        action="store_true",
        help="Also write animated GIFs (depth / opacity / rgb)",
    )
    parser.add_argument(
        "--render_with_oracle",
        action="store_true",
        help="Force RGB rendering with oracle depth bands (needed for oracle_only models)",
    )
    parser.add_argument(
        "--depth_margin",
        type=float,
        default=None,
        help="Margin for oracle RGB (default: 0.1 synthetic / 0.5 for 360)",
    )
    args = parser.parse_args()

    device = "cuda:0"
    set_random_seed(42)
    is_360 = args.scene in MIPNERF360_UNBOUNDED_SCENES
    data_root = args.data_root or str(
        pathlib.Path.cwd() / ("data/360_v2" if is_360 else "data/nerf_synthetic")
    )
    depth_margin_default = 0.5 if is_360 else 0.1

    oracle_path = pathlib.Path(
        args.oracle_path
        if args.oracle_path is not None
        else pathlib.Path.cwd()
        / "experiments"
        / "oracles"
        / args.scene
        / f"{args.split}_depth_opacity.pt"
    )
    out_dir = pathlib.Path(
        args.out_dir
        if args.out_dir is not None
        else pathlib.Path.cwd()
        / "experiments"
        / "oracles"
        / args.scene
        / f"vis_{args.split}"
    )
    depth_dir = out_dir / "depth"
    opacity_dir = out_dir / "opacity"
    rgb_dir = out_dir / "rgb"
    for d in (depth_dir, opacity_dir):
        d.mkdir(parents=True, exist_ok=True)

    oracle = torch.load(oracle_path, map_location="cpu")
    depths = oracle["depths"].numpy()  # [N,H,W]
    opacities = oracle["opacities"].numpy()
    n_views = depths.shape[0]
    indices = pick_view_indices(n_views, args.num_views)
    print(
        f"Oracle {oracle_path}: N={n_views}, writing {len(indices)} views -> {out_dir}"
    )

    # Shared depth scale across selected views (foreground only).
    fg_all = opacities[indices] >= args.opacity_threshold
    if fg_all.any():
        d_fg = depths[indices][fg_all]
        d_vmin = float(np.percentile(d_fg, 2))
        d_vmax = float(np.percentile(d_fg, 98))
    else:
        d_vmin, d_vmax = 0.0, 1.0

    depth_frames = []
    opacity_frames = []
    for k, view_i in enumerate(tqdm.tqdm(indices, desc="Oracle maps")):
        depth = depths[view_i]
        opacity = opacities[view_i]
        fg = opacity >= args.opacity_threshold

        depth_rgb = colorize_scalar(
            depth,
            fg,
            args.depth_cmap,
            vmin=d_vmin,
            vmax=d_vmax,
            bg_rgb=(1.0, 1.0, 1.0),
        )
        opacity_rgb = colorize_scalar(
            opacity,
            mask=None,
            cmap_name=args.opacity_cmap,
            vmin=0.0,
            vmax=1.0,
            bg_rgb=(0.0, 0.0, 0.0),
        )

        depth_name = f"depth_{k:02d}_view{view_i:03d}.png"
        opacity_name = f"opacity_{k:02d}_view{view_i:03d}.png"
        imageio.imwrite(depth_dir / depth_name, depth_rgb)
        imageio.imwrite(opacity_dir / opacity_name, opacity_rgb)
        depth_frames.append(depth_rgb)
        opacity_frames.append(opacity_rgb)

    if args.make_gif:
        imageio.mimsave(out_dir / "depth_360.gif", depth_frames, duration=0.25)
        imageio.mimsave(out_dir / "opacity_360.gif", opacity_frames, duration=0.25)

    # Optional RGB multi-view from a trained checkpoint (MLP or NGP Occ).
    if args.model_path is not None:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        ckpt = torch.load(args.model_path, map_location=device)
        # Heuristic: NGP Occ checkpoints use Instant-NGP; MLP uses PE-MLP.
        # Prefer scene type; also detect hash-grid keys if present.
        use_ngp = is_360 or any(
            k.startswith("mlp_base")
            for k in ckpt.get("radiance_field_state_dict", {})
        )

        if use_ngp:
            from datasets.nerf_360_v2 import SubjectLoader as Loader360
            from datasets.nerf_synthetic import SubjectLoader as LoaderSyn
            from radiance_fields.ngp import NGPRadianceField

            SubjectLoader = Loader360 if is_360 else LoaderSyn
            dataset_kwargs = {"factor": 4} if is_360 else {}
            aabb = (
                torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
                if is_360
                else torch.tensor(
                    [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device
                )
            )
            grid_nlvl = 4 if is_360 else 1
            near_plane = 0.2 if is_360 else 0.0
            render_step_size = 1e-3 if is_360 else 5e-3
            cone_angle = 0.004 if is_360 else 0.0
            alpha_thre = 1e-2 if is_360 else 0.0
            dataset = SubjectLoader(
                subject_id=args.scene,
                root_fp=data_root,
                split=args.split,
                num_rays=None,
                device=device,
                **dataset_kwargs,
            )
            estimator = OccGridEstimator(
                roi_aabb=aabb, resolution=128, levels=grid_nlvl
            ).to(device)
            radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(
                device
            )
        else:
            from datasets.nerf_synthetic import SubjectLoader
            from radiance_fields.mlp import VanillaNeRFRadianceField

            near_plane = 0.0
            render_step_size = 5e-3
            cone_angle = 0.0
            alpha_thre = 0.0
            dataset = SubjectLoader(
                subject_id=args.scene,
                root_fp=data_root,
                split=args.split,
                num_rays=None,
                device=device,
            )
            estimator = OccGridEstimator(
                roi_aabb=torch.tensor(
                    [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device
                ),
                resolution=128,
                levels=1,
            ).to(device)
            radiance_field = VanillaNeRFRadianceField().to(device)

        assert len(dataset) == n_views, (
            f"Oracle N={n_views} vs dataset len={len(dataset)} for split={args.split}"
        )

        radiance_field.load_state_dict(ckpt["radiance_field_state_dict"])
        estimator.load_state_dict(ckpt["estimator_state_dict"])
        oracle_mode = ckpt.get("oracle_mode", "baseline")
        use_oracle_render = args.render_with_oracle or oracle_mode in (
            "oracle_only",
            "oracle_bounds",
        )
        depth_margin = float(
            ckpt.get(
                "depth_margin",
                args.depth_margin
                if args.depth_margin is not None
                else depth_margin_default,
            )
        )
        opacity_thre = float(
            ckpt.get("opacity_threshold", args.opacity_threshold)
        )
        # For oracle_only the saved grid is all-occupied; keep that only if
        # we also constrain rays with oracle bands (otherwise OOM on full images).
        if oracle_mode == "oracle_only":
            estimator.binaries.fill_(True)
            estimator.occs.fill_(1.0)
        radiance_field.eval()
        estimator.eval()

        rgb_frames = []
        with torch.no_grad():
            for k, view_i in enumerate(tqdm.tqdm(indices, desc="RGB renders")):
                data = dataset[int(view_i)]
                render_kwargs = dict(
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=data["color_bkgd"],
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    test_chunk_size=args.test_chunk_size,
                )
                if use_oracle_render:
                    d_map = torch.from_numpy(depths[view_i]).to(device)
                    o_map = torch.from_numpy(opacities[view_i]).to(device)
                    active = o_map >= opacity_thre
                    t_min = torch.clamp(d_map - depth_margin, min=near_plane)
                    t_max = d_map + depth_margin
                    t_min = torch.where(active, t_min, torch.ones_like(t_min))
                    t_max = torch.where(active, t_max, torch.zeros_like(t_max))
                    rgb, acc, _, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        data["rays"],
                        t_min=t_min,
                        t_max=t_max,
                        **render_kwargs,
                    )
                    # Background for inactive pixels.
                    active_img = active[..., None]
                    rgb = torch.where(
                        active_img, rgb, data["color_bkgd"].expand_as(rgb)
                    )
                else:
                    rgb, _, _, _ = render_image_with_occgrid(
                        radiance_field,
                        estimator,
                        data["rays"],
                        **render_kwargs,
                    )
                rgb_u8 = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                name = f"rgb_{k:02d}_view{view_i:03d}.png"
                imageio.imwrite(rgb_dir / name, rgb_u8)
                rgb_frames.append(rgb_u8)

        if args.make_gif:
            imageio.mimsave(out_dir / "rgb_360.gif", rgb_frames, duration=0.25)

    print(f"Done. Wrote visualizations to {out_dir}")
    print(f"  depth/:   {len(indices)} images")
    print(f"  opacity/: {len(indices)} images")
    if args.model_path is not None:
        print(f"  rgb/:     {len(indices)} images")


if __name__ == "__main__":
    main()
