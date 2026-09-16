# Memory planning

## The idea

Peak memory depends on the **shapes** of tensors, not the values inside them. A matmul of 8192×768 by
768×2304 allocates the same bytes whether it holds trained weights or noise.

So Soliton has a third device beside `cpu` and `cuda`: **`meta`**. On it,

- allocation returns a fake address from a counter, and
- no kernel runs,

but everything else is identical — the same Python, the same op order, the same allocator with the same size
classes, block reuse and cache-flush behaviour. The allocator therefore sees an identical stream of requests,
so the peak it reports is the peak the real device will report.

```python
import soliton as sl

stats = sl.plan(lambda device: train_step(device))
print(stats["peak_reserved"])   # bytes the run will reserve
```

## What "exact" means here

The test suite asserts it directly: a hash of the entire allocation trace from the `meta` run must equal the
hash from the real CPU and CUDA runs, along with peak allocated and peak reserved. Measured on GPT-2 124M at
batches 1, 2, 4, 8 and 16, in both precision regimes, the error is **0 bytes**.

## What it does not cover

- The CUDA context and libraries sit outside Soliton's pool — about 0.29 GiB on an A6000. Add that if you are
  comparing against `nvidia-smi` rather than against the allocator.
- A step that branches on data values would break the guarantee. Soliton's ops do not.
- Plans and runs must share settings that change allocation: {func}`soliton.set_fusion` and arena mode.

## Why other frameworks cannot do this

PyTorch's caching allocator behaviour depends on reuse and timing it cannot know in advance; its `MemTracker`
estimate came in 5.0–8.6% below the reserved peak in our benchmark, in the dangerous direction. JAX and
TensorFlow expose no equivalent API at all.
