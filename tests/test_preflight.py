"""Preflight must catch a bad model on the meta device, and say where it failed."""
import soliton as sl
from gpt2 import run


def test_ok_report_has_the_numbers():
    r = sl.preflight.check(lambda d: run(d, "tiny", 2, 16, 1, 2), budget=8 * 2**30)
    assert r and r.ok and r.fits
    assert r.peak_reserved > 0 and r.n_alloc > 0


def test_budget_too_small_does_not_fit():
    r = sl.preflight.check(lambda d: run(d, "tiny", 2, 16, 1, 2), budget=1 << 20)
    assert not r, "a 1 MiB budget cannot fit anything"


def test_shape_error_names_the_module():
    class Bad(sl.nn.Module):
        def __init__(self):
            self.lin = sl.nn.Linear(8, 8, device="meta")

        def forward(self, x):
            return sl.matmul(self.lin(x), sl.normal((3, 3), 0.02, "meta"))

    r = sl.preflight.check(lambda d: Bad()(sl.normal((4, 8), 0.02, d)))
    assert not r.ok and "Bad" in r.scope and "shape mismatch" in r.error
