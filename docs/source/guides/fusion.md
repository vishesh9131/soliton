# Kernel fusion

A chain of elementwise operations normally means one GPU program per operation, each reading and writing
memory. Soliton records the chain, generates CUDA source for it, and compiles that with **NVRTC** — the
compiler already inside the CUDA driver. No LLVM, no Triton, no Python in the loop.

```python
y = sl.gelu(x * 2.0 + b) * 0.5 - 1.0   # recorded, not executed
y.ptr                                   # first read compiles and runs one kernel
```

## Measured

`gelu(x*2 + y)*0.5 - 1.0` over 16M floats on an RTX A6000:

| | Time | Bandwidth |
| --- | --- | --- |
| Eager, 5 kernels | 1.05 ms | 190 GB/s |
| **Fused, 1 generated kernel** | **0.30 ms** | **666 GB/s** |

666 GB/s is close to this card's practical ceiling, so the fused kernel is essentially memory-bound — the best
an elementwise chain can be. Against other frameworks on the same chain: JAX 0.35 ms, PyTorch 1.05 ms,
tinygrad 6.40 ms, TensorFlow 18.3 ms.

## Shapes do not trigger recompiles

Generated kernels take the element count as an argument instead of baking it in, so changing sequence length
costs nothing. Over 20 steps at varying lengths: Soliton 915 ms, PyTorch 867 ms, **JAX 76,514 ms** — JAX
recompiles for every new shape.

## Interaction with memory plans

Fusion changes which intermediates are allocated, so a plan and the run it predicts must use the same setting:

```python
sl.set_fusion(False)   # both the plan and the run, or neither
```

The decision to fuse is structural — identical on `cpu`, `meta` and `cuda` — so plans stay exact either way.
On GPT-2 specifically fusion changes nothing, because attention, layernorm and cross-entropy are already
hand-fused; the gain is for models whose chains nobody optimised by hand.
