"""Static memory planning.

Because a Soliton training step allocates the same sizes in the same order every time, the whole trace can be
recorded once on the meta device and every tensor given a fixed offset inside a single arena. Blocks whose
lifetimes do not overlap share the same bytes, so there is no fragmentation and nothing to reclaim at runtime.

A framework that cannot predict its allocations cannot do this.

    plan = arena.plan(lambda dev: train_step(dev))   # no GPU used
    arena.install(plan, "cuda")                      # one allocation for the whole run
    ...                                              # call arena.mark() at each step boundary
"""
import ctypes

from . import tensor as T
from ._C import lib, longs


class Plan:
    __slots__ = ("offsets", "sizes", "body", "bytes", "live_peak", "_keep")

    def __init__(self, offsets, sizes, body, nbytes, live_peak):
        self.offsets, self.sizes, self.body = offsets, sizes, body
        self.bytes, self.live_peak = nbytes, live_peak
        self._keep = None  # holds the ctypes arrays the pool points at, once installed

    @property
    def waste(self):
        """Fraction of the arena that packing could not reuse, against the live-bytes lower bound."""
        return (self.bytes - self.live_peak) / self.bytes if self.bytes else 0.0

    def __repr__(self):
        return (f"Plan({len(self.sizes)} blocks, arena {self.bytes / 2**30:.3f} GiB, "
                f"lower bound {self.live_peak / 2**30:.3f} GiB, waste {self.waste * 100:.2f}%)")


def record(fn):
    """Run fn("meta") with the block cache disabled, returning (sizes, starts, ends, body_start)."""
    meta = T.DEVICES.index("meta")
    T._dev(meta)
    lib.sl_record_start(meta)
    try:
        fn("meta")
    finally:
        lib.sl_record_stop(meta)
    n = lib.sl_record_count(meta)
    sizes, starts, ends = ((ctypes.c_int64 * n)() for _ in range(3))
    lib.sl_record_get(meta, sizes, starts, ends)
    return list(sizes), list(starts), list(ends), lib.sl_record_body(meta)


def solve(sizes, starts, ends):
    """Give every block an offset, largest first, at the lowest gap no live neighbour occupies.

    The classic greedy for this (NP-hard) packing problem; it lands within a few percent of the
    max-live-bytes lower bound on training traces."""
    n = len(sizes)
    end = [e if e >= 0 else n + 1 for e in ends]  # never freed -> live to the end
    offsets = [0] * n
    placed = []
    for i in sorted(range(n), key=lambda k: -sizes[k]):
        overlapping = [(offsets[j], sizes[j]) for j in placed if starts[i] < end[j] and starts[j] < end[i]]
        overlapping.sort()
        off = 0
        for o, sz in overlapping:
            if o >= off + sizes[i]:
                break  # the gap below this block is big enough
            off = max(off, o + sz)
        offsets[i] = off
        placed.append(i)

    arena = max((offsets[i] + sizes[i] for i in range(n)), default=0)
    live, peak = 0, 0  # lower bound: the most bytes ever live at once
    events = sorted([(starts[i], sizes[i]) for i in range(n)] + [(end[i], -sizes[i]) for i in range(n)])
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return offsets, arena, peak


def plan(fn):
    """Record on meta and solve. Uses no GPU."""
    sizes, starts, ends, body = record(fn)
    offsets, nbytes, live_peak = solve(sizes, starts, ends)
    return Plan(offsets, sizes, body, nbytes, live_peak)


def install(p, device):
    """Allocate the arena on `device` and replay the plan's offsets."""
    dev = T._dev(device)
    p._keep = (longs(p.offsets), longs(p.sizes))  # keep the arrays alive while the pool points at them
    off, sz = p._keep
    if lib.sl_arena_install(dev, off, sz, len(p.sizes), p.body, p.bytes) != 0:
        raise MemoryError(f"soliton: cannot allocate a {p.bytes / 2**30:.2f} GiB arena on {device}")


def mark(device="cuda"):
    """Step boundary: while planning it marks the repeating body, while running it rewinds to it."""
    lib.sl_record_mark(T._dev(device))


def check(device="cuda"):
    """Raise if the run has asked for an allocation the plan did not predict."""
    if lib.sl_arena_diverged(T._dev(device)):
        raise RuntimeError("soliton: the run diverged from its memory plan (an unplanned allocation). "
                           "Re-plan with the same configuration, or turn the arena off.")


def off(device="cuda"):
    lib.sl_arena_off(T._dev(device))
