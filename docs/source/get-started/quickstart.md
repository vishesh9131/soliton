# Quickstart

## Plan a step without a GPU

```bash
PYTHONPATH=python python examples/gpt2.py --plan-only --batch 8
```

```text
[plan] GPT-2 124M: 124.4M params, batch 8x1024, accum 1
[plan] peak allocated 9.226 GiB, peak reserved 10.268 GiB (dry run took 0.01s, no GPU used)
```

That number is not an estimate. Run the same configuration on a GPU and the peak reserved bytes match exactly.

## Check a configuration before you queue it

```python
import soliton as sl

report = sl.preflight.check(build_and_train, budget=24 * 2**30)
print(report)
if not report:
    raise SystemExit("would not fit")
```

The check runs the whole step on the `meta` device, so it catches shape errors and budget overruns in
milliseconds. See [Pre-flight checks](../guides/preflight.md).

## Train under a hard budget

```bash
PYTHONPATH=python python examples/prepare_tinystories.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python examples/gpt2.py \
  --data data/tinystories --batch 8 --budget-gib 24
```

Soliton solves which attention and MLP halves to recompute so the step fits the cap, then trains under it.
See [Budgets and recompute](../guides/budgets.md).

## Use the API directly

```python
import soliton as sl

x = sl.normal((4096, 4096), 0.02, "cuda").requires_grad_()
y = sl.gelu(x * 2.0 + 1.0)      # one generated kernel, not three
loss = sl.sum_(y)
loss.backward()

print(sl.memory_stats("cuda")["peak_reserved"])
```

## Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python python examples/gpt2.py \
  --data data/tinystories --nproc 2
```

One process per GPU, gradients averaged with NCCL. The memory plan is per rank and stays exact.
