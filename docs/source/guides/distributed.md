# Data parallel

One process per GPU, gradients averaged with NCCL in a single grouped all-reduce.

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python python examples/gpt2.py \
  --data data/tinystories --nproc 2
```

Under the hood:

```python
from soliton import distributed

distributed.init(rank, world, init_file)     # rank 0 publishes the NCCL id
...
loss.backward()
distributed.all_reduce_grads(opt.params)     # one group, averaged
opt.step()
```

## Measured

Two RTX A6000s over PCIe, GPT-2 124M:

| | Throughput | Memory plan |
| --- | --- | --- |
| 1 GPU | 13.0k tok/s | exact |
| 2 GPUs | **23.6k tok/s** (1.8×) | exact per rank |

The test suite checks correctness rather than only speed: two ranks must produce the **same parameters** as a
single process training on the combined batch.

## Memory planning with ranks

Collectives allocate nothing from the pool — NCCL uses its own buffers — so a single-rank dry run is the exact
per-rank plan. NCCL's buffers appear as fixed per-process overhead outside the pool, like the CUDA context.

## Not implemented

Tensor, pipeline and sharded-optimizer parallelism. Only data parallelism exists today.
