# Installation

Soliton builds from source on Linux with an NVIDIA CUDA toolkit. It is not on PyPI yet.

## Requirements

- Python 3.10 or newer, with NumPy and pytest
- NVIDIA CUDA toolkit: `nvcc`, cuBLAS, NVRTC
- NCCL (linked for data parallelism)
- [LDC](https://ldc-developers.github.io/), the LLVM D compiler

## Build

```bash
git clone https://github.com/vishesh9131/soliton.git
cd soliton

conda create -n soliton python=3.12 numpy pytest
conda activate soliton
conda install -c conda-forge ldc
conda install -c nvidia cuda-toolkit nccl

export CUDA_HOME="$CONDA_PREFIX"
export NCCL_HOME="$CONDA_PREFIX"
make CUDA_ARCH=sm_86        # RTX A6000; use sm_90 for H100, sm_89 for L40S, and so on
```

The build produces `python/soliton/libsoliton.so`, a single shared library holding the D core and the CUDA
kernels. There is no Python build step and no LLVM dependency.

## Verify

```bash
PYTHONPATH=python python -m pytest
PYTHONPATH=python python examples/gpt2.py --plan-only --batch 8
```

The second command should print a memory plan without allocating anything on a GPU — it works on a machine
whose GPUs are busy, and on a machine with no GPU at all.

## Choosing an architecture

`CUDA_ARCH` must match your card's compute capability, because the CUDA kernels are compiled ahead of time.
Fused kernels generated at runtime are compiled by NVRTC for whatever device is present, so they need no
configuration.
