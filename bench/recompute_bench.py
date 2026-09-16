"""Whole-block checkpointing vs a solved per-unit choice, at the same memory budget.

Both must fit the budget; the question is how much compute each one gives up to get there.
"""
import argparse
import os
import sys
import time

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(root, "python"), os.path.join(root, "examples")]
import soliton as sl  # noqa: E402
from gpt2 import CONFIGS, fit_checkpoints, run as gpt_run, units_of  # noqa: E402


def timed_step(batch, seq, ckpt, steps=4):
    rng = np.random.default_rng(0)

    def gb():
        tok = rng.integers(0, 50257, (batch, seq + 1)).astype(np.int32)
        return tok[:, :-1], tok[:, 1:]

    times = []
    stats = gpt_run("cuda", "124M", batch, seq, 1, steps, gb,
                    log=lambda s, l, n, dt: times.append(dt), checkpoint=ckpt)[0]
    return min(times[1:]), stats["peak_reserved"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--budget-gib", type=float, default=24)
    args = ap.parse_args()
    budget, n = int(args.budget_gib * 2**30), CONFIGS["124M"]["n_layer"]
    plan_peak = lambda ck: sl.plan(lambda d: gpt_run(d, "124M", args.batch, args.seq, 1, 2, checkpoint=ck))["peak_reserved"]  # noqa: E731

    base_ms, base_peak = timed_step(args.batch, args.seq, 0)
    print(f"no recompute      : {base_ms*1000:7.0f} ms/step  peak {base_peak/2**30:6.2f} GiB"
          f"  {'fits' if base_peak <= budget else 'OVER BUDGET'}")

    # cost per kind, measured once: blocks are identical, so one attn and one mlp timing covers all units
    cost_kind = {}
    for kind in ("attn", "mlp"):
        ms, _ = timed_step(args.batch, args.seq, frozenset([f"0.{kind}"]))
        cost_kind[kind] = max(ms - base_ms, 1e-6)
    cost_kind["block"] = cost_kind["attn"] + cost_kind["mlp"]
    print(f"measured cost     : attn {cost_kind['attn']*1000:.1f} ms, mlp {cost_kind['mlp']*1000:.1f} ms per unit")

    k = next((k for k in range(n + 1) if plan_peak(k) <= budget), None)
    blocks_ms, blocks_peak = timed_step(args.batch, args.seq, k)
    print(f"whole blocks (k={k}): {blocks_ms*1000:7.0f} ms/step  peak {blocks_peak/2**30:6.2f} GiB"
          f"  {2*k} units recomputed")

    cost = {u: cost_kind[u.split('.')[1]] for u in units_of("124M")}
    t0 = time.time()
    units, solved_peak_plan = fit_checkpoints("124M", args.batch, args.seq, 1, budget, cost)
    solve_s = time.time() - t0
    solved_ms, solved_peak = timed_step(args.batch, args.seq, units)
    kinds = sorted(u.split(".")[1] for u in units)
    print(f"solved            : {solved_ms*1000:7.0f} ms/step  peak {solved_peak/2**30:6.2f} GiB"
          f"  {len(units)} units recomputed ({kinds.count('attn')} attn, {kinds.count('mlp')} mlp)"
          f", solved in {solve_s:.1f}s with no GPU")
    print(f"=> solver is {blocks_ms/solved_ms:.3f}x the speed of whole-block checkpointing "
          f"({(blocks_ms-solved_ms)*1000:.0f} ms/step saved), both under {args.budget_gib} GiB")


if __name__ == "__main__":
    main()
