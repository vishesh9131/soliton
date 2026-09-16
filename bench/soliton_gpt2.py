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
    return fit_checkpoints("124M", batch, seq, 1, budget) if autofit else 0


def predict(batch, seq, budget_gib=None, autofit=False):
    """Dry run on meta. With a budget, meta enforces the same cap (and cache-flush-on-pressure) as the
    real device, so `fits` is an exact yes/no rather than a peak compared against a number."""
    t0 = time.time()
    budget = int(budget_gib * 2**30) if budget_gib else 0
    k = choose_checkpoints(batch, seq, budget, autofit)
    stats = None if k is None else sl.plan(lambda d: gpt_run(d, "124M", batch, seq, 1, 2, checkpoint=k), budget)
    out = dict(framework="soliton", batch=batch, seq=seq, fits=stats is not None, checkpoint=k, predict_s=time.time() - t0)
    if stats is not None:
        out.update(predicted=stats["peak_reserved"], predicted_alloc=stats["peak_allocated"])
    return out


def run(batch, seq, steps, budget_gib=None, autofit=False, tf32=False):
    out = dict(framework="soliton", batch=batch, seq=seq, oom=False)
    if tf32:
        sl.set_tf32(True)
    k = choose_checkpoints(batch, seq, int(budget_gib * 2**30) if budget_gib else 0, autofit)
    if k is None:
        return dict(out, oom=True, error="planned: does not fit even with every block checkpointed")
    out["checkpoint"] = k
    if budget_gib:  # hard cap on the pool's reserved memory
        sl.set_memory_limit("cuda", int(budget_gib * 2**30))
    rng = np.random.default_rng(0)

    def get_batch():
        tok = rng.integers(0, V, size=(batch, seq + 1)).astype(np.int32)
        return tok[:, :-1], tok[:, 1:]

    times = []
    try:
        stats, losses, _, _ = gpt_run("cuda", "124M", batch, seq, 1, steps, get_batch,
                                      log=lambda step, loss, norm, dt: times.append(dt), checkpoint=k)
        warm = times[2:] or times
        out.update(loss_first=losses[0], loss_last=losses[-1], first_step_s=times[0],
                   step_ms=1000 * sum(warm) / len(warm), tok_s=batch * seq * len(warm) / sum(warm),
                   native_peak=stats["peak_reserved"], native_peak_alloc=stats["peak_allocated"])
    except MemoryError as e:
        out.update(oom=True, error=str(e)[:200])
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
    args = ap.parse_args()
    if args.predict:
        res = predict(args.batch, args.seq, args.budget_gib, args.autofit)
    else:
        res = run(args.batch, args.seq, args.steps, args.budget_gib, args.autofit, args.tf32)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
