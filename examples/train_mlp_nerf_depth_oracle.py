"""
MLP NeRF trained with a frozen 2D depth/opacity sampling oracle (oracle_only).

Sampling uses only the baked maps: opacity skip + depth band [D-Δ, D+Δ].
The OccGrid is filled occupied and never updated (no density-based skipping).

For baseline OccGrid training use examples/train_mlp_nerf.py instead.

Bake maps first:
  PYTHONPATH=. python examples/bake_mlp_depth_oracle.py --model_path ... --scene lego --split train
  PYTHONPATH=. python examples/bake_mlp_depth_oracle.py --model_path ... --scene lego --split test
"""

import argparse
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from datasets.utils import Rays
from lpips import LPIPS
from radiance_fields.mlp import VanillaNeRFRadianceField

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    format_last_render_profile,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES,
)
parser.add_argument("--test_chunk_size", type=int, default=4096)
parser.add_argument(
    "--oracle_train_path",
    type=str,
    required=True,
    help="Baked train-split depth/opacity .pt from bake_mlp_depth_oracle.py",
)
parser.add_argument(
    "--oracle_test_path",
    type=str,
    required=True,
    help="Baked test-split depth/opacity .pt for oracle-guided eval",
)
parser.add_argument(
    "--depth_margin",
    type=float,
    default=0.1,
    help="Sample along ray in [D-delta, D+delta]",
)
parser.add_argument(
    "--opacity_threshold",
    type=float,
    default=0.01,
    help="Skip rays with oracle opacity below this (background color)",
)
parser.add_argument(
    "--max_num_rays",
    type=int,
    default=8192,
    help="Hard cap on rays/step after dynamic scaling",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Default: experiments/train_mlp_nerf_depth_oracle/oracle_only",
)
args = parser.parse_args()

max_steps = 50000
init_batch_size = 1024
target_sample_batch_size = 1 << 14
max_num_rays = args.max_num_rays
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
grid_resolution = 128
grid_nlvl = 1
render_step_size = 5e-3

out_dir = pathlib.Path(
    args.out_dir
    if args.out_dir is not None
    else pathlib.Path.cwd()
    / "experiments"
    / "train_mlp_nerf_depth_oracle"
    / "oracle_only"
)
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "log.txt"


def log_print(msg: str):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")


train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

oracle_train = torch.load(args.oracle_train_path, map_location=device)
oracle_depths_train = oracle_train["depths"].to(device)
oracle_opacities_train = oracle_train["opacities"].to(device)
assert oracle_depths_train.shape[0] == len(train_dataset), (
    f"Oracle N={oracle_depths_train.shape[0]} vs train images={len(train_dataset)}"
)

oracle_test = torch.load(args.oracle_test_path, map_location=device)
oracle_depths_test = oracle_test["depths"].to(device)
oracle_opacities_test = oracle_test["opacities"].to(device)
assert oracle_depths_test.shape[0] == len(test_dataset), (
    f"Oracle N={oracle_depths_test.shape[0]} vs test images={len(test_dataset)}"
)

# Occupancy grid used only as a traversal scaffold: all cells occupied, never updated.
estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)
estimator.binaries.fill_(True)
estimator.occs.fill_(1.0)

radiance_field = VanillaNeRFRadianceField().to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


def lookup_oracle(depths, opacities, image_ids, xs, ys):
    flat_ids = image_ids.reshape(-1).long()
    flat_xs = xs.reshape(-1).long()
    flat_ys = ys.reshape(-1).long()
    return (
        depths[flat_ids, flat_ys, flat_xs],
        opacities[flat_ids, flat_ys, flat_xs],
    )


def oracle_bounds_from_maps(depth, opacity):
    active = opacity >= args.opacity_threshold
    t_min = torch.clamp(depth - args.depth_margin, min=near_plane)
    t_max = depth + args.depth_margin
    # Inactive rays: empty interval so sampling yields nothing if called.
    t_min = torch.where(active, t_min, torch.ones_like(t_min))
    t_max = torch.where(active, t_max, torch.zeros_like(t_max))
    return t_min, t_max, active


def render_with_oracle(
    radiance_field,
    estimator,
    rays,
    render_bkgd,
    *,
    depths_map,
    opacities_map,
    image_ids,
    xs,
    ys,
    test_chunk_size=None,
):
    """March only in oracle depth bands; inactive rays get background RGB."""
    render_kwargs = dict(
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        # Oracle already restricts to [D-Δ, D+Δ]; skip density early-stop MLP.
        skip_sigma_fn=True,
    )
    if test_chunk_size is not None:
        render_kwargs["test_chunk_size"] = test_chunk_size

    depth_q, opacity_q = lookup_oracle(
        depths_map, opacities_map, image_ids, xs, ys
    )
    t_min, t_max, active = oracle_bounds_from_maps(depth_q, opacity_q)
    rays_shape = rays.origins.shape

    if len(rays_shape) == 3: # for test time rendering, [H,W,3], else [N,3]
        rgb, acc, depth, n_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            t_min=t_min.view(rays_shape[0], rays_shape[1]),
            t_max=t_max.view(rays_shape[0], rays_shape[1]),
            **render_kwargs,
        )
        active_img = active.view(rays_shape[0], rays_shape[1], 1)
        rgb = torch.where(active_img, rgb, render_bkgd.expand_as(rgb))
        acc = torch.where(active_img, acc, torch.zeros_like(acc))
        return rgb, acc, depth, n_samples

    n_rays = rays.origins.shape[0]
    rgb = render_bkgd.expand(n_rays, 3).clone()
    acc = torch.zeros(n_rays, 1, device=device)
    depth = torch.zeros(n_rays, 1, device=device)
    n_samples = 0
    n_active = int(active.sum().item())

    if n_active > 0:
        active_idx = active.nonzero(as_tuple=False).squeeze(-1)
        active_rays = Rays(
            origins=rays.origins[active_idx],
            viewdirs=rays.viewdirs[active_idx],
        )
        rgb_a, acc_a, depth_a, n_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            active_rays,
            t_min=t_min[active_idx],
            t_max=t_max[active_idx],
            **render_kwargs,
        )
        rgb[active_idx] = rgb_a
        acc[active_idx] = acc_a
        depth[active_idx] = depth_a

    return rgb, acc, depth, n_samples, n_active


log_print(
    f"oracle_only depth_margin={args.depth_margin} "
    f"opacity_threshold={args.opacity_threshold} "
    f"max_num_rays={max_num_rays} scene={args.scene} "
    f"max_steps={max_steps} out_dir={out_dir}"
)

tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]
    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    rgb, acc, depth, n_rendering_samples, n_active = render_with_oracle(
        radiance_field,
        estimator,
        rays,
        render_bkgd,
        depths_map=oracle_depths_train,
        opacities_map=oracle_opacities_train,
        image_ids=data["image_ids"],
        xs=data["xs"],
        ys=data["ys"],
    )

    if n_rendering_samples == 0 and n_active == 0:
        continue
    if n_rendering_samples == 0:
        loss = F.smooth_l1_loss(rgb, pixels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        continue

    # if target_sample_batch_size > 0 and n_rendering_samples > 0:
    #     num_rays = len(pixels)
    #     num_rays = int(
    #         num_rays * (target_sample_batch_size / float(n_rendering_samples))
    #     )
    #     num_rays = max(64, min(num_rays, max_num_rays))
    #     train_dataset.update_num_rays(num_rays)

    loss = F.smooth_l1_loss(rgb, pixels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 50 == 0:
        elapsed_time = time.time() - tic
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        log_print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={mse:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | "
            f"num_rays={len(pixels):d} | n_active={n_active:d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        train_time_s = time.time() - tic
        log_print(f"train_time_s={train_time_s:.2f}")

        model_save_path = str(out_dir / f"mlp_nerf_oracle_{step}")
        torch.save(
            {
                "step": step,
                "oracle_mode": "oracle_only",
                "depth_margin": args.depth_margin,
                "opacity_threshold": args.opacity_threshold,
                "radiance_field_state_dict": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
            },
            model_save_path,
        )

        radiance_field.eval()
        psnrs = []
        lpips = []
        render_times = []
        n_test_views = len(test_dataset)
        log_print(f"evaluation: rendering {n_test_views} test views")
        with torch.no_grad():
            for i in tqdm.tqdm(range(n_test_views)):
                data = test_dataset[i]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_render = time.time()
                rgb, acc, depth, _ = render_with_oracle(
                    radiance_field,
                    estimator,
                    data["rays"],
                    data["color_bkgd"],
                    depths_map=oracle_depths_test,
                    opacities_map=oracle_opacities_test,
                    image_ids=data["image_ids"],
                    xs=data["xs"],
                    ys=data["ys"],
                    test_chunk_size=args.test_chunk_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                render_time = time.time() - t_render
                render_times.append(render_time)
                mse = F.mse_loss(rgb, data["pixels"])
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, data["pixels"]).item())

                profile_str = format_last_render_profile()
                log_print(
                    f"eval view={i}/{n_test_views - 1} | "
                    f"psnr={psnr.item():.2f} | lpips={lpips[-1]:.4f} | "
                    f"render_time={render_time:.3f}s | "
                    f"psnr_avg={sum(psnrs) / len(psnrs):.2f} | "
                    f"max_depth={depth.max():.3f} | "
                    + (f"{profile_str} | " if profile_str else "")
                )

                if i == 0:
                    imageio.imwrite(
                        str(out_dir / "rgb_test.png"),
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        str(out_dir / "rgb_error.png"),
                        (
                            (rgb - data["pixels"]).norm(dim=-1).cpu().numpy()
                            * 255
                        ).astype(np.uint8),
                    )

        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        render_fps = len(render_times) / max(sum(render_times), 1e-12)
        log_print(
            f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, "
            f"train_time_s={train_time_s:.2f}, render_fps={render_fps:.3f}"
        )