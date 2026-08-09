"""
Bake per-view depth, opacity, and frequency maps from a trained MLP+OccGrid checkpoint.

These maps are used as a frozen 2D sampling oracle by train_mlp_nerf_depth_oracle.py.
Frequency F(u,v) follows FAGS Laplacian-band energy (see examples/oracle_frequency.py).
Does not modify train_mlp_nerf.py.
"""

import argparse
import pathlib

import torch
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from radiance_fields.mlp import VanillaNeRFRadianceField

from examples.oracle_frequency import laplacian_frequency_map
from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=NERF_SYNTHETIC_SCENES,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "trainval", "test"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained mlp_nerf checkpoint (e.g. experiments/train_mlp_nerf/mlp_nerf_10000)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save oracle .pt (default: experiments/oracles/<scene>/<split>_depth_opacity.pt)",
    )
    parser.add_argument("--test_chunk_size", type=int, default=4096)
    parser.add_argument(
        "--freq_num_levels",
        type=int,
        default=4,
        help="Laplacian pyramid levels for frequency map (FAGS-style)",
    )
    args = parser.parse_args()

    device = "cuda:0"
    set_random_seed(42)

    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    render_step_size = 5e-3
    grid_resolution = 128
    grid_nlvl = 1

    dataset = SubjectLoader(
        subject_id=args.scene,
        root_fp=args.data_root,
        split=args.split,
        num_rays=None,
        device=device,
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)
    radiance_field = VanillaNeRFRadianceField().to(device)

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
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=data["color_bkgd"],
                test_chunk_size=args.test_chunk_size,
            )
            # depth/acc: [H, W, 1] -> [H, W]
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
    payload = {
        "scene": args.scene,
        "split": args.split,
        "model_path": args.model_path,
        "depths": depths,  # [N, H, W]
        "opacities": opacities,  # [N, H, W]
        "frequencies": frequencies,  # [N, H, W] Laplacian energy in [0,1]
        "freq_num_levels": args.freq_num_levels,
        "height": depths.shape[1],
        "width": depths.shape[2],
    }
    torch.save(payload, out_path)
    print(
        f"Saved oracle maps to {out_path} "
        f"(N={depths.shape[0]}, H={depths.shape[1]}, W={depths.shape[2]})"
    )
    print(
        f"depth stats: min={depths.min():.4f} max={depths.max():.4f} "
        f"mean={depths.mean():.4f} | "
        f"opacity mean={opacities.mean():.4f} "
        f"frac>=0.01={(opacities >= 0.01).float().mean():.4f} | "
        f"freq mean={frequencies.mean():.4f} max={frequencies.max():.4f}"
    )


if __name__ == "__main__":
    main()
