"""
Bake per-view depth/opacity/frequency maps from a trained NGP+OccGrid checkpoint.

Frequency F(u,v) follows FAGS Laplacian-band energy (see examples/oracle_frequency.py).
Supports NeRF-Synthetic and Mip-NeRF 360 (`data/360_v2`).
Does not modify train_ngp_nerf_occ.py.

Example (360):
  PYTHONPATH=. python examples/bake_ngp_depth_oracle.py \\
    --scene garden --data_root data/360_v2 \\
    --split train --model_path experiments/train_ngp_nerf_occ/garden/ngp_occ_20000
"""

import argparse
import pathlib

import torch
import tqdm
from radiance_fields.ngp import NGPRadianceField

from examples.oracle_frequency import laplacian_frequency_map
from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def scene_config(scene: str, device: str):
    if scene in MIPNERF360_UNBOUNDED_SCENES:
        from datasets.nerf_360_v2 import SubjectLoader

        return {
            "SubjectLoader": SubjectLoader,
            "aabb": torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device),
            "near_plane": 0.2,
            "grid_resolution": 128,
            "grid_nlvl": 4,
            "render_step_size": 1e-3,
            "alpha_thre": 1e-2,
            "cone_angle": 0.004,
            "dataset_kwargs": {"factor": 4},
            "default_data_root": "data/360_v2",
        }
    else:
        from datasets.nerf_synthetic import SubjectLoader

        return {
            "SubjectLoader": SubjectLoader,
            "aabb": torch.tensor(
                [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device
            ),
            "near_plane": 0.0,
            "grid_resolution": 128,
            "grid_nlvl": 1,
            "render_step_size": 5e-3,
            "alpha_thre": 0.0,
            "cone_angle": 0.0,
            "dataset_kwargs": {},
            "default_data_root": "data/nerf_synthetic",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=str,
        default="garden",
        choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root (default depends on scene type)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="train/test (360); train/val/trainval/test (synthetic)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Trained NGP+Occ checkpoint with radiance_field + estimator state dicts",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--test_chunk_size", type=int, default=8192)
    parser.add_argument(
        "--freq_num_levels",
        type=int,
        default=4,
        help="Laplacian pyramid levels for frequency map (FAGS-style)",
    )
    args = parser.parse_args()

    device = "cuda:0"
    set_random_seed(42)
    cfg = scene_config(args.scene, device)
    data_root = args.data_root or str(
        pathlib.Path.cwd() / cfg["default_data_root"]
    )

    SubjectLoader = cfg["SubjectLoader"]
    dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=data_root,
        split=args.split,
        num_rays=None,
        device=device,
        **cfg["dataset_kwargs"],
    )

    estimator = OccGridEstimator(
        roi_aabb=cfg["aabb"],
        resolution=cfg["grid_resolution"],
        levels=cfg["grid_nlvl"],
    ).to(device)
    radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    radiance_field.eval()
    estimator.eval()

    depths = []
    opacities = []
    frequencies = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset)), desc=f"Baking {args.split}"):
            data = dataset[i]
            rgb, acc, depth, _ = render_image_with_occgrid(
                radiance_field,
                estimator,
                data["rays"],
                near_plane=cfg["near_plane"],
                render_step_size=cfg["render_step_size"],
                render_bkgd=data["color_bkgd"],
                cone_angle=cfg["cone_angle"],
                alpha_thre=cfg["alpha_thre"],
                test_chunk_size=args.test_chunk_size,
            )
            depths.append(depth.squeeze(-1).cpu())
            opacities.append(acc.squeeze(-1).cpu())
            frequencies.append(
                laplacian_frequency_map(
                    rgb.cpu(), num_levels=args.freq_num_levels
                )
            )

    depths = torch.stack(depths, dim=0)
    opacities = torch.stack(opacities, dim=0)
    frequencies = torch.stack(frequencies, dim=0)

    out_path = (
        pathlib.Path(args.output_path)
        if args.output_path is not None
        else pathlib.Path.cwd()
        / "experiments"
        / "oracles"
        / args.scene
        / f"{args.split}_depth_opacity.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scene": args.scene,
            "split": args.split,
            "model_path": args.model_path,
            "data_root": data_root,
            "depths": depths,
            "opacities": opacities,
            "frequencies": frequencies,
            "freq_num_levels": args.freq_num_levels,
            "height": depths.shape[1],
            "width": depths.shape[2],
            "is_360": args.scene in MIPNERF360_UNBOUNDED_SCENES,
        },
        out_path,
    )
    print(
        f"Saved oracle maps to {out_path} "
        f"(N={depths.shape[0]}, H={depths.shape[1]}, W={depths.shape[2]})"
    )
    print(
        f"depth stats: min={depths.min():.4f} max={depths.max():.4f} "
        f"mean={depths.mean():.4f} | opacity mean={opacities.mean():.4f} "
        f"frac>=0.01={(opacities >= 0.01).float().mean():.4f} | "
        f"freq mean={frequencies.mean():.4f} max={frequencies.max():.4f}"
    )


if __name__ == "__main__":
    main()
