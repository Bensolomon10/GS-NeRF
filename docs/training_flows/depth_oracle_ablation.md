# Depth-oracle ablation

Frozen depth/opacity maps guide ray sampling (`[D−Δ, D+Δ]` + opacity skip). Original trainers are unchanged:

- `[examples/train_mlp_nerf.py](../../examples/train_mlp_nerf.py)`
- `[examples/train_ngp_nerf_occ.py](../../examples/train_ngp_nerf_occ.py)`

## Which script?


| Dataset                       | Field         | Oracle trainer                       | Bake script                |
| ----------------------------- | ------------- | ------------------------------------ | -------------------------- |
| NeRF Synthetic                | MLP + OccGrid | `train_mlp_nerf_depth_oracle.py`     | `bake_mlp_depth_oracle.py` |
| Synthetic **or** Mip-NeRF 360 | NGP + OccGrid | `train_ngp_nerf_occ_depth_oracle.py` | `bake_ngp_depth_oracle.py` |


Use the **NGP** path for `data/360_v2` (cascades + cone tracing). MLP oracle is synthetic-only.

## Modes


| Script                                                                                          | Role                                                                              |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `[train_mlp_nerf.py](../../examples/train_mlp_nerf.py)`                                         | **Baseline** OccGrid MLP (no oracle)                                              |
| `[train_mlp_nerf_depth_oracle.py](../../examples/train_mlp_nerf_depth_oracle.py)`               | **oracle_only** MLP: opacity skip + `[D±Δ]`; OccGrid all-occupied, never updated  |
| `[train_ngp_nerf_occ.py](../../examples/train_ngp_nerf_occ.py)`                                 | **Baseline** OccGrid NGP (no oracle)                                              |
| `[train_ngp_nerf_occ_depth_oracle.py](../../examples/train_ngp_nerf_occ_depth_oracle.py)`       | **oracle_only** NGP: opacity skip + `[D±Δ]`; OccGrid all-occupied, never updated  |


Default `--depth_margin`: `0.1` (synthetic) / `0.5` (360 NGP). Both oracle trainers are **oracle_only only** — use the original trainers for baseline.

---



## A. Synthetic MLP

```bash
# Bake
PYTHONPATH=. python examples/bake_mlp_depth_oracle.py \
  --scene lego --split train \
  --model_path experiments/train_mlp_nerf/mlp_nerf_10000
PYTHONPATH=. python examples/bake_mlp_depth_oracle.py \
  --scene lego --split test \
  --model_path experiments/train_mlp_nerf/mlp_nerf_10000

# Baseline (original trainer)
PYTHONPATH=. python examples/train_mlp_nerf.py \
  --scene lego --data_root data/nerf_synthetic

# Oracle-only
PYTHONPATH=. python examples/train_mlp_nerf_depth_oracle.py \
  --scene lego \
  --oracle_train_path experiments/oracles/lego/train_depth_opacity.pt \
  --oracle_test_path experiments/oracles/lego/test_depth_opacity.pt
```

Logs / checkpoints: `experiments/train_mlp_nerf_depth_oracle/oracle_only/`.

---

## B. NGP Occ oracle

Bake from a checkpoint produced by `[train_ngp_nerf_occ.py](../../examples/train_ngp_nerf_occ.py)`.
Assume baseline checkpoints live under scene folders, e.g. `experiments/train_ngp_nerf_occ/<scene>/ngp_occ_20000`.

### B1. Synthetic

```bash
# Baseline NGP Occ (synthetic) — save under lego/
# PYTHONPATH=. python examples/train_ngp_nerf_occ.py \
#   --scene lego --data_root data/nerf_synthetic
# then move/copy checkpoint to:
#   experiments/train_ngp_nerf_occ/lego/ngp_occ_20000

# Bake train + test oracle maps
PYTHONPATH=. python examples/bake_ngp_depth_oracle.py \
  --scene lego --data_root data/nerf_synthetic --split train \
  --model_path experiments/train_ngp_nerf_occ/lego/ngp_occ_20000

PYTHONPATH=. python examples/bake_ngp_depth_oracle.py \
  --scene lego --data_root data/nerf_synthetic --split test \
  --model_path experiments/train_ngp_nerf_occ/lego/ngp_occ_20000

# Oracle-only
PYTHONPATH=. python examples/train_ngp_nerf_occ_depth_oracle.py \
  --scene lego \
  --oracle_train_path experiments/oracles/lego/train_depth_opacity.pt \
  --oracle_test_path experiments/oracles/lego/test_depth_opacity.pt
```

Maps: `experiments/oracles/lego/{train,test}_depth_opacity.pt`.  
Oracle runs: `experiments/train_ngp_nerf_occ_depth_oracle/lego/oracle_only/`.

### B2. Mip-NeRF 360

```bash
conda activate nerfacc

# Baseline NGP Occ (360) — save under garden/
# PYTHONPATH=. python examples/train_ngp_nerf_occ.py \
#   --scene garden --data_root data/360_v2
# then move/copy checkpoint to:
#   experiments/train_ngp_nerf_occ/garden/ngp_occ_20000

# Bake train + test oracle maps
PYTHONPATH=. python examples/bake_ngp_depth_oracle.py \
  --scene garden --data_root data/360_v2 --split train \
  --model_path experiments/train_ngp_nerf_occ/garden/ngp_occ_20000

PYTHONPATH=. python examples/bake_ngp_depth_oracle.py \
  --scene garden --data_root data/360_v2 --split test \
  --model_path experiments/train_ngp_nerf_occ/garden/ngp_occ_20000

# Oracle-only
PYTHONPATH=. python examples/train_ngp_nerf_occ_depth_oracle.py \
  --scene garden --data_root data/360_v2 \
  --oracle_train_path experiments/oracles/garden/train_depth_opacity.pt \
  --oracle_test_path experiments/oracles/garden/test_depth_opacity.pt
```

Safer defaults on 360: `target_sample_batch_size=2^16`, `max_num_rays=8192`. Override with `--target_sample_batch_size` / `--max_num_rays` if needed.

Maps: `experiments/oracles/garden/{train,test}_depth_opacity.pt`.  
Oracle runs: `experiments/train_ngp_nerf_occ_depth_oracle/garden/oracle_only/`.

---



## Visualize



### Synthetic (MLP)

```bash
PYTHONPATH=. python examples/visualize_oracle_views.py \
  --scene lego --split test \
  --oracle_path experiments/oracles/lego/test_depth_opacity.pt \
  --model_path experiments/train_mlp_nerf_depth_oracle/oracle_only/mlp_nerf_oracle_10000 \
  --num_views 12 --make_gif \
  --out_dir experiments/oracles/lego/vis_test_oracle_only
```



### Mip-NeRF 360 (NGP Occ)

Depth/opacity color maps only (no checkpoint needed beyond the baked `.pt`):

```bash
PYTHONPATH=. python examples/visualize_oracle_views.py \
  --scene garden \
  --data_root data/360_v2 \
  --split test \
  --oracle_path experiments/oracles/garden/test_depth_opacity.pt \
  --num_views 12 \
  --make_gif \
  --out_dir experiments/oracles/garden/vis_test
```

Depth + opacity + RGB from an NGP oracle checkpoint (same evenly spaced views):

```bash
PYTHONPATH=. python examples/visualize_oracle_views.py \
  --scene garden \
  --data_root data/360_v2 \
  --split test \
  --oracle_path experiments/oracles/garden/test_depth_opacity.pt \
  --model_path experiments/train_ngp_nerf_occ_depth_oracle/garden/oracle_only/ngp_occ_oracle_20000 \
  --num_views 12 \
  --make_gif \
  --out_dir experiments/oracles/garden/vis_test_oracle_only
```

Outputs under `experiments/oracles/<scene>/...`: `depth/`, `opacity/`, `rgb/` (if `--model_path`), plus optional `*_360.gif`.

## Frequency-adaptive sampling (optional)

Baked \(F(u,v)\) can widen the depth band and shrink the march step. Details:

- [`oracle_frequency_map.md`](oracle_frequency_map.md) — how \(F\) is built (Laplacian / FAGS-style)
- [`oracle_frequency_depth_margin.md`](oracle_frequency_depth_margin.md) — \(\Delta=\Delta_0(1+\lambda_\Delta F)\)
- [`oracle_frequency_step_size.md`](oracle_frequency_step_size.md) — \(\delta=\delta_0/(1+\lambda_\delta F)\)

Disable both with `--freq_margin_scale 0 --freq_step_scale 0`. Train logs every 50 steps report `F_*`, `Δ_*`, `δ_*` on active rays.

## What to compare

From `log.txt`: `n_rendering_samples`, `n_active`, `elapsed_time`, train PSNR, final `psnr_avg` / `lpips_avg`, plus frequency stats (`F_mean`, `Δ_*`, `δ_*`). Sweep `--depth_margin` if quality drops (try larger values on 360).