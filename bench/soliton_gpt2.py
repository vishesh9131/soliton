"""GPT-2 training step in Soliton (model from examples/gpt2.py). --predict is a meta-device dry run."""
import argparse
import json
import os
import sys
import time

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(root, "python"), os.path.join(root, "examples")]
import soliton as sl  # noqa: E402
from gpt2 import fit_checkpoints, run as gpt_run  # noqa: E402

V = 50257


def choose_checkpoints(batch, seq, budget, autofit):
    """Solver picks which attention/MLP halves to recompute; None when nothing fits."""
    return fit_checkpoints("124M", batch, seq, 1, budget)[0] if autofit else 0


def arena_plan(batch, seq, k):
    return sl.arena.plan(lambda d: gpt_run(d, "124M", batch, seq, 1, 2, checkpoint=k))


def predict(batch, seq, budget_gib=None, autofit=False, arena=False):
    """Dry run on meta. With a budget, meta enforces the same cap (and cache-flush-on-pressure) as the
    real device, so `fits` is an exact yes/no rather than a peak compared against a number.

    With --arena the prediction is the size of the single planned arena, which is what the run will reserve."""
    t0 = time.time()
    budget = int(budget_gib * 2**30) if budget_gib else 0
    k = choose_checkpoints(batch, seq, budget, autofit)
    name = "soliton+arena" if arena else "soliton"
    if k is None:
        return dict(framework=name, batch=batch, seq=seq, fits=False, checkpoint=None, predict_s=time.time() - t0)
    if arena:
        p = arena_plan(batch, seq, k)
        return dict(framework=name, batch=batch, seq=seq, fits=not budget or p.bytes <= budget, checkpoint=k if isinstance(k, int) else len(k),
                    predicted=p.bytes, predicted_alloc=p.live_peak, predict_s=time.time() - t0)
    stats = sl.plan(lambda d: gpt_run(d, "124M", batch, seq, 1, 2, checkpoint=k), budget)
    out = dict(framework=name, batch=batch, seq=seq, fits=stats is not None,
           checkpoint=k if isinstance(k, int) else len(k), predict_s=time.time() - t0)
    if stats is not None:
        out.update(predicted=stats["peak_reserved"], predicted_alloc=stats["peak_allocated"])
    return out


def run(batch, seq, steps, budget_gib=None, autofit=False, tf32=False, arena=False):
    out = dict(framework="soliton+arena" if arena else "soliton", batch=batch, seq=seq, oom=False)
    if tf32:
        sl.set_tf32(True)
    budget = int(budget_gib * 2**30) if budget_gib else 0
    k = choose_checkpoints(batch, seq, budget, autofit)
    if k is None:
        return dict(out, oom=True, error="planned: does not fit even with every block checkpointed")
    out["checkpoint"] = k if isinstance(k, int) else len(k)
    plan = None
    if arena:  # one allocation for the whole run, sized by the plan
        plan = arena_plan(batch, seq, k)
        if budget and plan.bytes > budget:
            return dict(out, oom=True, error=f"planned arena {plan.bytes} exceeds budget {budget}")
        try:
            sl.arena.install(plan, "cuda")
        except MemoryError as e:
            return dict(out, oom=True, error=str(e)[:200])
    elif budget:  # hard cap on the pool's reserved memory
        sl.set_memory_limit("cuda", budget)
    rng = np.random.default_rng(0)

    def get_batch():
        tok = rng.integers(0, V, size=(batch, seq + 1)).astype(np.int32)
        return tok[:, :-1], tok[:, 1:]

    times = []
    try:
        stats, losses, _, _ = gpt_run("cuda", "124M", batch, seq, 1, steps, get_batch,
                                      log=lambda step, loss, norm, dt: times.append(dt), checkpoint=k)
        if arena:
            sl.arena.check("cuda")
        warm = times[2:] or times
        out.update(loss_first=losses[0], loss_last=losses[-1], first_step_s=times[0],
                   step_ms=1000 * sum(warm) / len(warm), tok_s=batch * seq * len(warm) / sum(warm),
                   native_peak=plan.bytes if arena else stats["peak_reserved"],
                   native_peak_alloc=plan.live_peak if arena else stats["peak_allocated"])
    except MemoryError as e:
        out.update(oom=True, error=str(e)[:200])
    finally:
        if arena:
            sl.arena.off("cuda")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--autofit", action="store_true", help="checkpoint as few blocks as needed to fit the budget")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor cores for matmul (lower precision)")
    ap.add_argument("--arena", action="store_true", help="plan one arena and replay fixed offsets")
    args = ap.parse_args()
    assert not (args.arena and args.autofit), "--arena with --autofit is not wired up yet"
    if args.predict:
        res = predict(args.batch, args.seq, args.budget_gib, args.autofit, args.arena)
    else:
        res = run(args.batch, args.seq, args.steps, args.budget_gib, args.autofit, args.tf32, args.arena)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
