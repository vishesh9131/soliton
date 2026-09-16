"""A recorded plan must reproduce the run exactly, in one allocation, with no fragmentation."""
import numpy as np
import pytest

import soliton as sl
from soliton._C import lib
from gpt2 import run

DEVICES = ["cpu"] + (["cuda"] if lib.sl_cuda_count() > 0 else [])
B, T, STEPS = 4, 32, 3


def batches():
    rng = np.random.default_rng(0)
    return lambda: tuple(rng.integers(0, 512, size=(B, T)).astype(np.int32) for _ in range(2))


def train(device):
    return run(device, "tiny", B, T, 1, STEPS, None if device == "meta" else batches())


def test_plan_packs_below_the_pool():
    p = sl.arena.plan(lambda dev: train(dev))
    assert len(p.sizes) > 100 and p.bytes > 0
    assert p.bytes >= p.live_peak, "an arena cannot be smaller than the bytes live at once"
    assert p.waste < 0.10, f"greedy packing wasted {p.waste:.1%}, expected under 10%"


@pytest.mark.parametrize("device", DEVICES)
def test_arena_run_matches_normal_run(device):
    plan = sl.arena.plan(lambda dev: train(dev))

    sl.empty_cache(device)
    _, losses_ref, model_ref, _ = train(device)
    ref = [p.numpy().copy() for p in model_ref.parameters()]
    del model_ref

    sl.empty_cache(device)
    sl.arena.install(plan, device)
    try:
        _, losses, model, _ = train(device)
        got = [p.numpy().copy() for p in model.parameters()]
        sl.arena.check(device)
        del model
    finally:
        sl.arena.off(device)

    np.testing.assert_allclose(losses, losses_ref, rtol=1e-6)
    for a, b in zip(got, ref):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_arena_uses_less_memory_than_the_caching_pool(device):
    plan = sl.arena.plan(lambda dev: train(dev))
    sl.empty_cache(device)
    sl.reset_peak_stats(device)
    base = sl.memory_stats(device)["reserved"]
    train(device)
    pooled = sl.memory_stats(device)["peak_reserved"] - base
    assert plan.bytes < pooled, f"arena {plan.bytes} should beat the pool's {pooled}"
