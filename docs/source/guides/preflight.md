# Pre-flight checks

A shape mismatch usually surfaces minutes into a job, deep in a stack trace, after the GPU was already busy.
Soliton can run the entire step on `meta` in milliseconds, so it can check first.

```python
import soliton as sl

print(sl.preflight.check(train_step, budget=12 * 2**30))
```

```text
soliton preflight OK — 2202 allocations, 11 ms, no GPU used
  peak allocated 9.226 GiB   peak reserved 10.268 GiB
  budget 12.000 GiB — fits
```

When something is wrong it names the module it happened in, not just the framework line that raised:

```text
soliton preflight FAILED in GPT → Block → Linear
  AssertionError: matmul shape mismatch (8, 1024, 768) (512, 999)
  checked in 0 ms, no GPU used
```

## The report

{class}`soliton.preflight.Report` is truthy when the step is valid *and* fits the budget, so it drops into a
guard:

```python
report = sl.preflight.check(train_step, budget=24 * 2**30, arena=True)
if not report:
    raise SystemExit(str(report))
```

| Field | Meaning |
| --- | --- |
| `ok` | the step ran to completion on `meta` |
| `error`, `scope` | what failed, and the module path it failed in |
| `peak_allocated`, `peak_reserved` | bytes the real run will use |
| `arena_bytes` | size of the static plan, when `arena=True` |
| `fits` | against `budget`; `None` when no budget was given |
| `seconds`, `n_alloc` | cost of the check, and how many allocations it saw |

Because it needs no GPU, this belongs in CI, in a job submission script, or in front of an expensive sweep.
