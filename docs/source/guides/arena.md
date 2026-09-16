# Static arenas

A caching allocator reuses blocks by size and always strands some memory. Soliton's wasted 11% on GPT-2;
PyTorch's wastes about 5%. But when the entire allocation trace is known in advance, the allocator can stop
guessing: give every tensor a fixed offset in one arena, and let tensors whose lifetimes never overlap share
the same bytes. It is what a compiler does with registers.

```python
from soliton import arena

plan = arena.plan(train_step)      # record on meta, solve offsets. No GPU.
print(plan)                        # Plan(2202 blocks, arena 9.195 GiB, lower bound 9.195 GiB, waste 0.00%)

arena.install(plan, "cuda")        # one allocation for the whole run
...                                # arena.mark() at each step boundary
arena.check("cuda")                # raises if the run ever diverged from the plan
arena.off("cuda")
```

## Measured

GPT-2 124M, peak reserved:

| Batch | Caching pool | Static arena | Saved |
| --- | --- | --- | --- |
| 1 | 3.32 GiB | **2.60** | −21.7% |
| 4 | 6.34 GiB | **5.38** | −15.1% |
| 8 | 10.27 GiB | **9.20** | −10.5% |
| 16 | 18.11 GiB | **16.81** | −7.2% |

At batch 8 the solver packs 2202 blocks into 9.195 GiB against a 9.195 GiB lower bound — **0.00% waste** —
in 0.45 s with no GPU, and throughput is unchanged (0.998×). Savings are largest at small batch, where fixed
blocks like parameters and optimizer state dominate.

## Rules the arena depends on

- **A step must be self-contained.** Anything allocated inside a step is released inside it, so one plan can
  repeat. The example trainer releases gradients at the end of the step for this reason.
- **`arena.mark()` goes before the step's first allocation**, since the cursor rewinds there.
- **Every allocation is size-checked against the plan.** A run that diverges fails loudly instead of being
  handed the wrong buffer.

## What it did not buy

At GPT-2's scale, saving 1.07 GiB is less than one batch costs, so the largest batch fitting 24 GiB stayed at
23. The arena buys headroom, not another sequence.
