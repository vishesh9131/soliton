---
sd_hide_title: true
---

# Soliton

:::{div} hero

# Know the memory before the run

Soliton is a deep learning framework that tells you exactly how much GPU memory a training step needs
**before it runs** — to the byte, in hundredths of a second, without touching a GPU.

:::

:::{div} ledger

GPT-2 124M · batch 8 × 1024 · fp32 · RTX A6000

planned, no GPU visible &nbsp;&nbsp; **11,025,126,400 bytes**
measured on the GPU &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **11,025,126,400 bytes**
difference &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **0 bytes**

:::

```python
import soliton as sl

def train_step(device):
    model = GPT(device=device)
    ...

print(sl.preflight.check(train_step, budget=24 * 2**30))
# soliton preflight OK — 2202 allocations, 11 ms, no GPU used
#   peak allocated 9.226 GiB   peak reserved 10.268 GiB
#   budget 24.000 GiB — fits
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stopwatch` Plan before you run
Run the step on the `meta` device: same code, same allocator, no kernels. The peak it reports is the peak
you get.
+++
[Memory planning](guides/memory-planning.md)
:::

:::{grid-item-card} {octicon}`shield-check` Train under a hard budget
State a cap. Soliton solves which parts to recompute so the step fits, then verifies it.
+++
[Budgets and recompute](guides/budgets.md)
:::

:::{grid-item-card} {octicon}`container` One allocation for the whole run
Every tensor gets a fixed offset, solved offline. 0.00% packing waste on GPT-2.
+++
[Static arenas](guides/arena.md)
:::

:::{grid-item-card} {octicon}`repo-forked` Reproducible by construction
Bit-identical weights across processes and across GPUs.
+++
[Determinism](guides/determinism.md)
:::

::::

## Why it matters

Everywhere else you learn a configuration's memory by running it and waiting for the crash, then shrinking the
batch and trying again. Each attempt costs minutes and a GPU you may be sharing. Soliton answers the question
up front, and turns a memory budget into something the framework enforces rather than something you discover.

```{toctree}
:hidden:
:caption: Get started

get-started/installation
get-started/quickstart
```

```{toctree}
:hidden:
:caption: Guides

guides/memory-planning
guides/budgets
guides/arena
guides/preflight
guides/determinism
guides/fusion
guides/distributed
```

```{toctree}
:hidden:
:caption: Reference

api/index
benchmarks
```
