"""Fused elementwise chains must match the unfused result exactly in value, and exactly in allocations."""
import numpy as np
import pytest

import soliton as sl
from soliton._C import lib

DEVICES = ["cpu"] + (["cuda"] if lib.sl_cuda_count() > 0 else [])

CHAINS = {
    "scale_add_gelu": lambda x, y: sl.gelu(x * 2.0 + y),
    "long_chain": lambda x, y: sl.gelu((x * 0.5 - 1.0) * y + x) * 3.0,
    "residual": lambda x, y: x + y * 2.0,
    "div_mul": lambda x, y: (x / (y * y + 1.0)) * x,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", CHAINS)
def test_fused_matches_eager(device, name):
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal((4, 128)).astype(np.float32) for _ in range(2))
    g = rng.standard_normal((4, 128)).astype(np.float32)
    out = {}
    for fuse in (True, False):
        sl.set_fusion(fuse)
        try:
            x, y = (sl.tensor(v, device).requires_grad_() for v in (a, b))
            z = CHAINS[name](x, y)
            z.backward(sl.tensor(g, device))
            out[fuse] = (z.numpy(), x.grad.numpy(), y.grad.numpy())
        finally:
            sl.set_fusion(True)
    for fused, eager in zip(out[True], out[False]):
        np.testing.assert_allclose(fused, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fusion_allocates_less_and_matches_meta(device):
    def work(dev):
        x = sl.tensor(np.ones((32, 256), np.float32), dev) if dev != "meta" else sl.empty((32, 256), dev)
        y = sl.gelu(x * 2.0 + 1.0) * 0.5
        y.ptr  # force the chain to run
        return y

    stats = {}
    for fuse in (True, False):
        sl.set_fusion(fuse)
        try:
            for dev in ("meta", device):
                sl.empty_cache(dev)
                sl.reset_peak_stats(dev)
                work(dev)
                s = sl.memory_stats(dev)
                stats[(fuse, dev)] = (s["n_alloc"], s["trace"])
        finally:
            sl.set_fusion(True)

    for fuse in (True, False):  # the plan must match the real run either way
        assert stats[(fuse, "meta")] == stats[(fuse, device)], f"meta != {device} with fusion={fuse}"
    assert stats[(True, device)][0] < stats[(False, device)][0], "fusion should allocate fewer buffers"
