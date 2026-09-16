# Budgets and recompute

## A budget the framework enforces

```python
sl.set_memory_limit("cuda", 24 * 2**30)
```

The pool refuses to reserve past the cap, flushing cached blocks first — and `meta` enforces the same cap, so
a dry run under a budget answers *"would this train?"* exactly, rather than comparing an estimate to a number.

```python
stats = sl.plan(train_step, budget=24 * 2**30)
if stats is None:
    print("does not fit")
```

## Solving what to recompute

If it does not fit, some activations must be recomputed in backward instead of kept. Choosing *which* is an
optimisation problem, and Soliton has the information to solve it: the dry run gives the exact peak for any
choice, and the lifetime graph gives what each choice saves.

```python
from soliton import recompute

units = [f"{i}.{part}" for i in range(12) for part in ("attn", "mlp", "block")]
chosen, peak = recompute.solve(make_run, units, budget=24 * 2**30, cost=measured_cost)
```

`solve` picks the cheapest set whose savings cover the overshoot, then **verifies the answer with another
exact dry run**, adding units until it really fits. Nothing is guessed.

### Measured

GPT-2 124M, batch 32, 24 GiB cap, on an RTX A6000:

| Policy | Units recomputed | Step time | Peak |
| --- | --- | --- | --- |
| Whole blocks (the usual approach) | 14 | 2102 ms | 23.94 GiB |
| **Solved** | **11** | **2049 ms** | 23.95 GiB |

The solver chose mostly MLP halves even though they are the *more* expensive ones to recompute (35.3 ms
against 29.6 ms), because they free enough more memory to be worth it — the kind of trade nobody makes by hand.

With auto-fit, the same 24 GiB cap trains **batch 52**, up from 23 with no recomputation.

## In the example trainer

```bash
python examples/gpt2.py --data data/tinystories --batch 8 --budget-gib 24
```

```text
[plan] fits 24.0 GiB by recomputing 13 of 36 units (23.98 GiB), solved in 6.39s with no GPU
```
