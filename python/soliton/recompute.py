"""Choose which parts of a model to recompute so a training step fits a memory budget at least cost.

Whole-block checkpointing is blunt: it trades far more compute than a budget usually needs. Because a dry run
reports the exact peak for any choice, the choice can be solved instead of guessed -- pick the cheapest set of
units whose savings cover the overshoot, then verify the answer with one more exact plan.

A unit is anything the model can wrap in sl.checkpoint (for GPT-2: each block's attention and MLP halves).
"""
from . import tensor as T

MiB = 2**20


def peak_for(make_run, chosen, budget=0):
    """Peak reserved bytes for one choice of checkpointed units, from a meta dry run.

    With a budget the dry run enforces the same cap as the real device, cache flushing included, so None
    means "this choice would not train under that cap" rather than "its unconstrained peak is larger"."""
    stats = T.plan(make_run(frozenset(chosen)), budget)
    return None if stats is None else stats["peak_reserved"]


def savings(make_run, units, baseline=None):
    """Bytes each unit saves on its own. One dry run per unit; no GPU."""
    base = baseline if baseline is not None else peak_for(make_run, ())
    return base, {u: max(base - peak_for(make_run, (u,)), 0) for u in units}


def _knapsack(need, saving, cost):
    """Cheapest subset whose savings cover `need` bytes. DP over MiB of coverage."""
    units = list(saving)
    cap = max(1, -(-need // MiB))
    INF = float("inf")
    best = [INF] * (cap + 1)
    pick = [None] * (cap + 1)
    best[0] = 0.0
    for u in units:
        s = min(cap, saving[u] // MiB)
        if s <= 0:
            continue
        c = cost[u]
        for have in range(cap, -1, -1):  # 0/1: iterate downward so each unit is used once
            if best[have] == INF:
                continue
            nxt = min(cap, have + s)
            if best[have] + c < best[nxt]:
                best[nxt] = best[have] + c
                pick[nxt] = (have, u)
    if best[cap] == INF:
        return None
    chosen, at = [], cap
    while at and pick[at]:
        prev, u = pick[at]
        chosen.append(u)
        at = prev
    return chosen


def solve(make_run, units, budget, cost=None):
    """Least-cost set of units to checkpoint so the step fits `budget` bytes.

    `cost` maps unit -> relative recompute cost; without it every unit counts the same, which minimises how
    many units are recomputed. Returns (chosen_set, exact_peak) or (None, None) if even everything fails."""
    cost = cost or {u: 1.0 for u in units}
    base, saved = savings(make_run, units)
    if base <= budget:
        return frozenset(), base

    chosen = _knapsack(base - budget, saved, cost) or list(units)
    # Savings interact, so the additive solve is only a starting point: verify exactly, and if it still
    # overshoots add the best remaining unit by saving-per-cost until it fits.
    rest = sorted(set(units) - set(chosen), key=lambda u: -saved[u] / cost[u])
    while True:
        peak = peak_for(make_run, chosen, budget)  # under the cap, exactly as the run will execute
        if peak is not None:
            return frozenset(chosen), peak
        if not rest:
            return None, None
        chosen.append(rest.pop(0))


def measure_costs(time_step, units):
    """Relative recompute cost per unit, timed on the real device: how much slower one step gets."""
    base = time_step(frozenset())
    return {u: max(time_step(frozenset([u])) - base, 1e-6) for u in units}
