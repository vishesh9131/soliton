# Determinism

Soliton is deterministic by construction rather than by option:

- the allocator's behaviour depends only on the request sequence,
- op order is fixed,
- fusion decisions are structural, not timing-dependent,
- and the backward pass avoids float atomics — the embedding backward sorts indices instead, precisely so the
  sum order is fixed.

## Measured

Training GPT-2 124M three steps from a fixed seed, then hashing every parameter:

| Configuration | Result |
| --- | --- |
| fp32, GPU 2, run twice in separate processes | identical, `f99d7122…` |
| fp32, **GPU 3** | identical to GPU 2, `f99d7122…` |
| TF32, GPU 2, twice | identical, `d04ad4ad…` |
| fusion disabled, GPU 2, twice | identical, `5a1e6b4c…` |

So weights are bit-identical **across processes and across two physical GPUs** of the same model. cuBLAS did
not vary its algorithm choice, which is the usual source of run-to-run drift elsewhere.

## What changes the bits

Settings that change arithmetic change results, as they should: TF32 differs from fp32, and fusion changes
rounding because it keeps intermediates in registers. Each setting is self-consistent; compare like with like.

## Not yet verified

- Multi-GPU runs, where all-reduce ordering could vary.
- Different GPU architectures, or a different cuBLAS version.

PyTorch cannot offer this guarantee: `torch.use_deterministic_algorithms(True)` is incomplete, several ops
refuse to run under it, and float atomics make many backward kernels order-dependent.
