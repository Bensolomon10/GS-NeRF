# Environment Setup (nerfacc)

Guide to recreate the working Conda environment used on this machine
(**NVIDIA GeForce RTX 4060 Ti**, Ubuntu, system CUDA toolkit 12.0 / `nvcc`).

Verified stack:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| torchvision | 0.16.0+cu121 |
| nerfacc | 0.5.3 (editable from this repo) |
| tinycudann | 2.0 |
| NumPy | 1.26.4 |
| setuptools | 80.10.2 |

---

## 0. System prerequisites

- NVIDIA driver working (`nvidia-smi` shows the GPU)
- CUDA toolkit with `nvcc` available (this machine: Ubuntu `nvidia-cuda-toolkit` 12.0)
- GCC 11 host compiler for CUDA builds (system default GCC 13 breaks torch 2.1 / pybind11):

```bash
sudo apt install g++-11 gcc-11 nvidia-cuda-toolkit
which nvcc gcc-11 g++-11
```

On this machine `nvcc` is at `/usr/bin/nvcc`. There is **no** `/usr/local/cuda` — do **not** set `CUDA_HOME=/usr/local/cuda`.

---

## 1. Create the Conda env

```bash
cd /home/benso/projects/nerfacc

conda create -n nerfacc python=3.10 -y
conda activate nerfacc
```

In Cursor: **Python: Select Interpreter** → pick the `nerfacc` env.

---

## 2. Install PyTorch (CUDA 12.1 wheels)

```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Check GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.1.0+cu121 True NVIDIA GeForce RTX 4060 Ti
```

---

## 3. Pin packages that break torch 2.1 / builds

```bash
# torch 2.1 was built against NumPy 1.x
pip install "numpy<2"

# recent setuptools removed pkg_resources; torch 2.1 still needs it
pip install "setuptools<81" wheel ninja
```

---

## 4. Install nerfacc from this repo (CUDA extension)

Modern pip uses an isolated build env that does **not** see your installed `torch`.
Also, GCC 13 fails compiling torch 2.1’s bundled pybind11 headers.

```bash
unset CUDA_HOME
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11

pip install -e . --no-build-isolation
```

Verify:

```bash
python -c "import torch, nerfacc; print(torch.__version__, nerfacc.__version__, torch.cuda.is_available())"
# expect: 2.1.0+cu121 0.5.3 True
```

Optional: install remaining pinned deps from the freeze file:

```bash
pip install -r requirements.txt
```

(`torch` / `torchvision` lines in `requirements.txt` still need the cu121 index URL from step 2.)

---

## 5. Install example dependencies

For **MLP NeRF** (`train_mlp_nerf.py`) — no tinycudann needed:

```bash
pip install opencv-python imageio "numpy==1.26.4" tqdm scipy lpips
```

> If pip upgrades NumPy to 2.x (e.g. via newer `opencv-python`), pin it back:
> `pip install numpy==1.26.4`

For **Instant-NGP** (`train_ngp_nerf_*.py`) — install **tinycudann**:

```bash
unset CUDA_HOME
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11
export TCNN_CUDA_ARCHITECTURES=89   # Ada / RTX 4060 Ti = sm_89

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
  --no-build-isolation
```

Verify:

```bash
python -c "import tinycudann as tcnn; print('tinycudann OK')"
```

Do **not** rely on plain `pip install -r examples/requirements.txt` alone — that hits the same build-isolation / `pkg_resources` failure unless you pass `--no-build-isolation` and the GCC/CUDA env vars above.

---

## 6. Dataset

Download [NeRF Synthetic (Blender)](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
(`nerf_synthetic.zip`) and unpack so you have:

```text
data/nerf_synthetic/lego/transforms_train.json
data/nerf_synthetic/lego/transforms_test.json
data/nerf_synthetic/lego/train/
data/nerf_synthetic/lego/test/
```

---

## 7. Run examples

Always from the **repo root**. Scripts import `examples.*`, so set `PYTHONPATH`:

```bash
cd /home/benso/projects/nerfacc
conda activate nerfacc
export PYTHONPATH=.
```

### Vanilla MLP NeRF (~1 hour)

```bash
python examples/train_mlp_nerf.py --scene lego --data_root data/nerf_synthetic
```

### Instant-NGP + occupancy grid (~minutes)

```bash
python examples/train_ngp_nerf_occ.py --scene lego --data_root data/nerf_synthetic
```

### Instant-NGP + proposal network

```bash
python examples/train_ngp_nerf_prop.py --scene lego --data_root data/nerf_synthetic
```

Training prints loss / PSNR. At eval time, scripts print `psnr_avg` / `lpips_avg`.
Image dumps (`rgb_test.png`, `rgb_error.png`) are commented out in the eval loops — uncomment if you want saved renders.

---

## Troubleshooting (issues we hit)

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` during `pip install -e .` | Use `--no-build-isolation` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install "setuptools<81"` and `--no-build-isolation` |
| `pybind11/.../cast.h` compile errors with g++ 13 | `export CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11` |
| `No such file or directory: '/usr/local/cuda/bin/nvcc'` | `unset CUDA_HOME` (use system `/usr/bin/nvcc`) |
| NumPy / torch warning `_ARRAY_API not found` | `pip install "numpy<2"` / `numpy==1.26.4` |
| `ModuleNotFoundError: No module named 'examples'` | Run with `PYTHONPATH=.` from repo root |
| `No module named 'tinycudann'` | Install with GCC 11 + `TCNN_CUDA_ARCHITECTURES=89` + `--no-build-isolation` |

---

## One-shot recreate (after system packages are installed)

```bash
cd /home/benso/projects/nerfacc

conda create -n nerfacc python=3.10 -y
conda activate nerfacc

pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install "numpy==1.26.4" "setuptools<81" wheel ninja
pip install opencv-python imageio tqdm scipy lpips rich

unset CUDA_HOME
export CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11

pip install -e . --no-build-isolation

export TCNN_CUDA_ARCHITECTURES=89
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
  --no-build-isolation

export PYTHONPATH=.
python -c "import torch, nerfacc, tinycudann; print(torch.__version__, nerfacc.__version__, torch.cuda.is_available())"
```
