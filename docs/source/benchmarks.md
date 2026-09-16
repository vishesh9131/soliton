# Benchmarks

Every number here is measured on one NVIDIA RTX A6000, GPT-2 124M at sequence length 1024 with AdamW, with
all frameworks held to the same matmul precision in each regime. Raw tables live in the repository under
`bench/results/`.

:::{note}
JAX's default precision uses TF32 tensor cores and ignores `NVIDIA_TF32_OVERRIDE=0`, which only binds cuBLAS.
Comparing it against everyone else's fp32 flatters it by roughly 1.6×, so the fp32 regime sets
`JAX_DEFAULT_MATMUL_PRECISION=highest`.
:::

## Predicting memory before running

```{image} _static/chart-memory.png
:alt: Peak GPU memory per framework
:width: 100%
```

| Framework | Peak reserved, batch 8 | Predicted in advance? | Cost |
| --- | --- | --- | --- |
| **Soliton (arena)** | **9.20 GiB** | **exact, 0 bytes** | 0.4 s, no GPU |
| Soliton (pool) | 10.27 GiB | exact, 0 bytes | 0.01 s, no GPU |
| PyTorch | 15.83 GiB | 5.0–8.6% too low | 5.4 s, needs a GPU |
| TensorFlow | 22.92 GiB | no API | — |
| JAX | 34.28 GiB (process) | no API | — |

## Largest batch under a 24 GiB budget

```{image} _static/chart-maxbatch.png
:alt: Largest batch under a 24 GiB budget
:width: 100%
```

| Framework | Batch | How |
| --- | --- | --- |
| **Soliton, auto-fit** | **52** | solver picks 13 units, 6 s, no GPU, verified by training |
| **Soliton** | **23** | predicted in 2 s, correct first time |
| JAX | 16 | 6 training runs, 3 crashes, 187 s |
| PyTorch | 12 | its estimator said 13, which OOMs; 7 runs to find 12 |
| TensorFlow | 8 | 7 runs, 4 crashes, 183 s |

## Throughput

```{image} _static/chart-throughput.png
:alt: Training throughput in both precisions
:width: 100%
```

Tokens per second at batch 8:

| Framework | fp32 | TF32 |
| --- | --- | --- |
| JAX | **18,037** | **31,802** |
| Soliton | 16,678 | 26,911 |
| PyTorch | 15,576 | 24,159 |
| TensorFlow | 11,447 | 15,596 |

At batch 16 with TF32, Soliton reaches 28,927 tok/s and JAX runs out of memory.

## Workload micro-benchmarks

| Workload | Soliton | PyTorch | JAX | TensorFlow | tinygrad |
| --- | --- | --- | --- | --- | --- |
| Elementwise chain, 16M floats | **0.31 ms** | 1.05 | 0.35 | 18.3 | 6.40 |
| MLP, 4 × 4096, one step | 73.8 ms | 71.8 | **39.1** | 83.7 | 376 |
| Attention fwd+bwd, seq 1024 | 6.43 ms | **3.40** | 3.54 | — | 23.5 |
| 20 steps, changing sequence length | **915 ms** | 867 | 76,514 | — | — |

## Where Soliton loses

- JAX is about 8% faster on GPT-2 at fp32 and 18% at TF32, because it compiles and fuses the whole step.
- PyTorch's FlashAttention is roughly 2× faster than Soliton's attention: ours keeps score tiles in HBM
  between cuBLAS calls, while FlashAttention keeps them in on-chip memory.
- `torch.compile` could not run in our environment at all — Inductor raises "duplicate template name" on
  torch 2.10 with Triton 3.6, while `aot_eager` compiles fine.

## Reproducing

```bash
PYTHONPATH=python python bench/run.py --gpu 0 --precision fp32 --experiments e1,e2,e3
PYTHONPATH=python python bench/micro.py --framework soliton --workload chain
PYTHONPATH=python python bench/recompute_bench.py --batch 32 --budget-gib 24
PYTHONPATH=python python bench/make_charts.py
```
