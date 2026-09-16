"""The core Soliton claim: a dry run on the meta device reproduces the real allocation trace exactly."""
import numpy as np
import pytest

import soliton as sl
from soliton._C import lib
from gpt2 import GPT, CONFIGS, fit_checkpoints, run

REAL = ["cpu"] + (["cuda"] if lib.sl_cuda_count() > 0 else [])
B, T, ACCUM, STEPS = 2, 16, 2, 3


def batches():
    rng = np.random.default_rng(0)
    return lambda: tuple(rng.integers(0, 512, size=(B, T)).astype(np.int32) for _ in range(2))


def measure(device, checkpoint=0):
    sl.empty_cache(device)
    base = sl.memory_stats(device)
    stats, losses, _, _ = run(device, "tiny", B, T, ACCUM, STEPS, None if device == "meta" else batches(),
                              checkpoint=checkpoint)
    return {k: stats[k] - base["allocated"] if k.startswith("peak_") else stats[k] for k in ("peak_allocated", "trace", "n_alloc")}, stats, losses


@pytest.mark.parametrize("checkpoint", [0, 1])
@pytest.mark.parametrize("device", REAL)
def test_meta_trace_matches_real(device, checkpoint):
    meta, _, _ = measure("meta", checkpoint)
    real, _, losses = measure(device, checkpoint)
    assert all(np.isfinite(losses)) and losses[-1] < losses[0] + 1.0
    assert real == meta


def test_checkpoint_gives_same_grads_and_less_memory():
    # Big enough that activations, not the fixed cuBLAS workspace, dominate the peak.
    tok = np.random.default_rng(0).integers(0, 512, size=(2, 8, 64)).astype(np.int32)
    grads, peaks = {}, {}
    for k in (0, 2):
        sl.empty_cache("cpu")
        base = sl.memory_stats("cpu")["allocated"]
        sl.reset_peak_stats("cpu")
        model = GPT(**{**CONFIGS["tiny"], "device": "cpu", "checkpoint": k})
        model(sl.tensor(tok[0]), sl.tensor(tok[1])).backward()
        grads[k] = [p.grad.numpy() for p in model.parameters()]
        peaks[k] = sl.memory_stats("cpu")["peak_allocated"] - base
        del model
    for a, b in zip(grads[0], grads[2]):
        np.testing.assert_array_equal(a, b)
    assert peaks[2] < peaks[0]


def test_fit_checkpoints_is_minimal():
    # The real config on the meta device: free, and big enough that checkpointing changes the peak.
    cfg, b, t, n = "124M", 4, 1024, CONFIGS["124M"]["n_layer"]
    need = lambda k: sl.plan(lambda d: run(d, cfg, b, t, 1, 2, checkpoint=k))["peak_reserved"]  # noqa: E731
    assert need(n) < need(0)
    budget = (need(0) + need(n)) // 2  # fits with full checkpointing, not without
    k = fit_checkpoints(cfg, b, t, 1, budget)
    assert k is not None and k > 0
    assert sl.plan(lambda d: run(d, cfg, b, t, 1, 2, checkpoint=k), budget) is not None
    assert sl.plan(lambda d: run(d, cfg, b, t, 1, 2, checkpoint=k - 1), budget) is None


def test_limit_below_plan_fails_and_at_plan_succeeds():
    _, stats, _ = measure("meta")
    need = stats["peak_reserved"]
    try:
        sl.empty_cache("meta")
        sl.set_memory_limit("meta", need - (1 << 20))
        with pytest.raises(MemoryError):
            measure("meta")
        sl.empty_cache("meta")
        sl.set_memory_limit("meta", need)
        measure("meta")
    finally:
        sl.set_memory_limit("meta", 0)
