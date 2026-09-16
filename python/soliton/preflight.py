"""Check a training step before it touches a GPU: shapes, dtypes, and whether it fits a memory budget.

The whole step runs on the meta device, so this costs milliseconds and no GPU. It reports where a failure
happened in the model, not just which line of the framework raised.
"""
import time

from . import arena as _arena
from . import tensor as T


class Report:
    __slots__ = ("ok", "error", "scope", "peak_allocated", "peak_reserved", "arena_bytes", "budget", "seconds",
                 "n_alloc")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def fits(self):
        """None when no budget was given."""
        if not self.budget:
            return None
        peak = self.arena_bytes or self.peak_reserved
        return False if peak is None else peak <= self.budget  # no numbers means it overshot while running

    def __bool__(self):
        return bool(self.ok) and self.fits is not False

    def __str__(self):
        gib = lambda n: "—" if n is None else f"{n / 2**30:.3f} GiB"  # noqa: E731
        if not self.ok:
            where = " → ".join(self.scope) if self.scope else "outside any module"
            return (f"soliton preflight FAILED in {where}\n"
                    f"  {self.error}\n"
                    f"  checked in {self.seconds * 1000:.0f} ms, no GPU used")
        lines = [f"soliton preflight OK — {self.n_alloc} allocations, {self.seconds * 1000:.0f} ms, no GPU used",
                 f"  peak allocated {gib(self.peak_allocated)}   peak reserved {gib(self.peak_reserved)}"]
        if self.arena_bytes:
            lines.append(f"  static arena   {gib(self.arena_bytes)}")
        if self.budget:
            lines.append(f"  budget {gib(self.budget)} — {'fits' if self.fits else 'DOES NOT FIT'}")
        return "\n".join(lines)


def check(fn, budget=0, arena=False):
    """Run fn("meta") and report. `budget` in bytes; `arena` also solves the static plan."""
    T._fail_scope = []
    t0 = time.perf_counter()
    try:
        if arena:
            plan = _arena.plan(fn)
            stats = T.memory_stats("meta")
            return Report(ok=True, arena_bytes=plan.bytes, peak_allocated=plan.live_peak,
                          peak_reserved=plan.bytes, budget=budget, seconds=time.perf_counter() - t0,
                          n_alloc=len(plan.sizes))
        stats = T.plan(fn, budget)
        if stats is None:  # hit the budget while running
            return Report(ok=True, peak_allocated=None, peak_reserved=None, budget=budget,
                          seconds=time.perf_counter() - t0, n_alloc=None, error="exceeded the budget")
        return Report(ok=True, peak_allocated=stats["peak_allocated"], peak_reserved=stats["peak_reserved"],
                      budget=budget, seconds=time.perf_counter() - t0, n_alloc=stats["n_alloc"])
    except Exception as e:
        return Report(ok=False, error=f"{type(e).__name__}: {e}", scope=list(T._fail_scope),
                      seconds=time.perf_counter() - t0)
