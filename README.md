<p align="center">
  <img src="assets/soliton-banner.png" alt="Soliton — framework that plans the GPU memory required for training before the run starts" width="908">
</p>

<p align="center">
  <strong>Experimental.</strong> Soliton is a research-stage framework with a Python API, a D/LDC core, CUDA kernels, and a deterministic memory allocator.
</p>

## Evidence first

The main claim is deliberately narrow and testable: given the same training step and allocator policy, a Soliton dry run on its `meta` device predicts the real peak reserved GPU memory exactly—without launching GPU kernels.

The checked benchmark is GPT-2 124M, sequence length 1024, AdamW, on one NVIDIA RTX A6000. Soliton, PyTorch, JAX, and TensorFlow were run under matched FP32 and TF32 regimes. Full raw tables are included in [`bench/results/fp32/REPORT.md`](bench/results/fp32/REPORT.md) and [`bench/results/tf32/REPORT.md`](bench/results/tf32/REPORT.md).

| Measured result | Outcome |
| --- | --- |
| Memory-plan accuracy | **0 bytes error** against Soliton's peak reserved memory at batches 1, 2, 4, 8, and 16 |
| Planning cost | **0.01 s** with **0 GPU memory** used for the plan |
| Static memory planning | **9.20 GiB** peak reserved at batch 8 against **10.27 GiB** for the caching pool (−10.5%), packed with **0.00%** waste and unchanged throughput |
| Maximum GPT-2 batch under 24 GiB | **23**, chosen from dry runs and verified by a real training run |
| Auto-fit under the same 24 GiB cap | **Batch 52** by choosing activation checkpointing; predicted and actual peak: **23.97 GiB** |
| FP32 throughput at batch 8 | **16,678 tokens/s**; PyTorch: 15,576; JAX: 18,037; TensorFlow: 11,447 |
| TF32 throughput at batch 16 | **28,927 tokens/s**; PyTorch: 25,440; JAX did not fit at that batch |

Those results do **not** claim that Soliton is universally faster or more mature than existing frameworks. On this machine and workload, JAX remains faster at throughput where it fits, and PyTorch's FlashAttention remains roughly 2× faster than Soliton's attention benchmark. The reproducible evidence is the memory-planning guarantee and the measured budget-fitting workflow.

### What makes the plan exact?

Soliton uses the same deterministic allocator and allocation sequence in a real run and on `meta`:

1. Operations and kernels request scratch memory from the allocator in a fixed order.
2. `meta` replays that allocation trace using fake addresses and skips GPU execution.
3. A hard memory limit uses the same pool policy in both modes.
4. The planner can search checkpoint counts before touching a GPU.
5. `sl.arena` goes one step further: the recorded trace is packed into a single arena, giving every tensor a
   fixed offset, and the run replays those offsets with a per-allocation size check. Tensors whose lifetimes do
   not overlap share the same bytes, so the plan *is* the allocation.

The direct regression coverage is in [`tests/test_memory.py`](tests/test_memory.py) and
[`tests/test_arena.py`](tests/test_arena.py).

## Install

Soliton currently builds from source on Linux with an NVIDIA CUDA toolkit. It is not yet published as a PyPI package.

### Requirements

- Python 3.10+
- NumPy and pytest
- NVIDIA CUDA toolkit with `nvcc`, cuBLAS, and NVRTC
- NCCL (the current shared library links it for data parallelism)
- [LDC](https://ldc-developers.github.io/) — the LLVM D compiler

One Conda-based setup is:

```bash
git clone https://github.com/vishesh9131/soliton.git
cd soliton

conda create -n soliton python=3.12 numpy pytest
conda activate soliton
conda install -c conda-forge ldc
conda install -c nvidia cuda-toolkit nccl

export CUDA_HOME="$CONDA_PREFIX"
export NCCL_HOME="$CONDA_PREFIX"
make CUDA_ARCH=sm_86  # RTX A6000; change for your GPU
```

The build writes `python/soliton/libsoliton.so`. For another NVIDIA architecture, set `CUDA_ARCH` to its CUDA compute capability (for example, `sm_90`).

## Use it

### Plan before running

This creates GPT-2 124M on the `meta` device, computes its allocation trace, and reports the memory budget without running CUDA kernels.

```bash
PYTHONPATH=python python examples/gpt2.py --plan-only --batch 8
```

Example output:

```text
[plan] GPT-2 124M: 124.4M params, batch 8x1024, accum 1
[plan] peak allocated 13.870 GiB, peak reserved 15.637 GiB
```

### Train under a hard memory budget

Prepare the TinyStories data, then let Soliton choose the fewest GPT-2 blocks to checkpoint in order to fit the stated cap.

```bash
PYTHONPATH=python python examples/prepare_tinystories.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python examples/gpt2.py \
  --data data/tinystories \
  --batch 8 \
  --budget-gib 24
```

### Use the Python API

```python
import soliton as sl

def train_step(device):
    x = sl.normal((4096, 4096), device)
    return sl.sum_(sl.gelu(x))

# Inspect the allocation plan without using a GPU.
plan = sl.plan(train_step, budget=24 * 2**30)

# Opt in to TensorFloat-32 matmuls on supported NVIDIA GPUs.
sl.set_tf32(True)
```

### Test the build

```bash
PYTHONPATH=python python -m pytest
```

For multi-GPU data-parallel GPT-2 training, expose the GPUs you intend to use and pass `--nproc`:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python python examples/gpt2.py \
  --data data/tinystories --nproc 2
```

## Current scope

Implemented today:

- Float32 tensors, reverse-mode autograd, common neural-network modules, AdamW, CPU and CUDA backends
- Deterministic caching allocator, `meta` planning, a hard memory cap, and checkpoint auto-fit
- Static arena planning (`sl.arena`): one allocation for the whole run, offsets solved offline from the trace
- GPT-2 124M training with fused causal attention and data parallelism over NCCL
- Opt-in TF32 and an elementwise CUDA fusion path

Not yet implemented: BF16/FP16 mixed precision, true strided views, offload/tensor/pipeline parallelism, and a mature whole-program compiler. See the raw benchmark reports before relying on performance conclusions outside the measured workload.

## Reproduce the benchmark

```bash
PYTHONPATH=python python bench/run.py --help
PYTHONPATH=python python bench/micro.py --help
```

The benchmark compares equivalent GPT-2 workloads in Soliton, PyTorch, JAX, and TensorFlow. Run a precision regime consistently—do not compare JAX's TF32-default configuration to strict FP32 results from other frameworks.

## Project status

Soliton is an early open-source experiment. Contributions, benchmark replications, and bug reports are welcome; claims should remain tied to reproducible measurements and their stated hardware, precision, and workload.
