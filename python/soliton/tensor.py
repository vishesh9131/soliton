"""Tensor, autograd and ops.

Tensors are contiguous float32 (or int32 for indices). Only reshape shares storage; everything else
allocates, so the sequence of allocations depends only on shapes -- which is what lets a dry run on the
meta device predict real memory exactly.

Autograd invariants, which make memory both low and deterministic:
- The graph is made of Nodes that point at parent Nodes, never at parent Tensors, so an activation stays
  alive only if some backward closure saved it.
- A backward closure never captures its own output Tensor (only its Storage), so there are no reference
  cycles and everything is freed by refcounting at the same point on every device.
"""
import ctypes
import math
import os
from contextlib import contextmanager

import numpy as np

from ._C import lib, longs

DEVICES = ("cpu", "meta", "cuda")
CPU, META, CUDA = 0, 1, 2

_scope, _fail_scope = [], []  # module breadcrumbs, for preflight error messages
_ordinal = int(os.environ.get("LOCAL_RANK", 0))
_ready = set()
_grad_on = True


def set_device(ordinal):
    """Pick the CUDA device for this process (before first CUDA use)."""
    global _ordinal
    if CUDA in _ready and ordinal != _ordinal:
        raise RuntimeError("soliton: CUDA device already initialized")
    _ordinal = ordinal


def _dev(device):
    d = device if isinstance(device, int) else DEVICES.index(device)
    if d not in _ready:
        if d == CUDA and lib.sl_cuda_count() <= _ordinal:
            raise RuntimeError(f"soliton: no CUDA device {_ordinal}")
        if lib.sl_init_device(d, _ordinal) != 0:
            raise RuntimeError(f"soliton: cannot initialize {DEVICES[d]}")
        _ready.add(d)
    return d


def _check(d):
    err = lib.sl_check(d)
    if err:
        raise RuntimeError(f"soliton: {DEVICES[d]} error code {err}")


def _ok(rc):
    if rc:
        raise MemoryError("soliton: out of memory allocating op scratch space")


def _fmt(n):
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.1f} MiB"


# ---------------------------------------------------------------- memory

_STAT_KEYS = ("allocated", "reserved", "peak_allocated", "peak_reserved", "n_alloc", "n_raw", "trace")


# Memory APIs initialize the device first, so its fixed reservations (cuBLAS workspace) always land
# before any measurement window instead of inside whichever one happens to touch the device first.


def memory_stats(device="cuda"):
    buf = (ctypes.c_int64 * 7)()
    lib.sl_stats(_dev(device), buf)
    return dict(zip(_STAT_KEYS, buf))


def reset_peak_stats(device="cuda"):
    lib.sl_reset_stats(_dev(device))


def empty_cache(device="cuda"):
    lib.sl_empty_cache(_dev(device))


def set_memory_limit(device, nbytes):
    """Cap reserved bytes (0 = no cap). On meta this simulates a smaller GPU."""
    lib.sl_set_limit(_dev(device), int(nbytes))


def set_tf32(on=True):
    """Let matmul use TF32 tensor cores: much faster on Ampere, ~10 mantissa bits instead of 24. Off by default."""
    lib.sl_set_tf32(_dev(CUDA), 1 if on else 0)


def cuda_mem_info():
    """(free, total) bytes as the driver sees them."""
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    _dev(CUDA)
    lib.sl_mem_info(ctypes.byref(free), ctypes.byref(total))
    return free.value, total.value


def synchronize():
    if CUDA in _ready:
        lib.sl_sync(CUDA)
        _check(CUDA)


def plan(fn, budget=0):
    """Dry-run fn("meta") -- build the model and run training steps -- optionally under a reserved-memory
    budget in bytes. Returns the memory stats of the run, or None if it would not fit the budget.

    Exact for the real device because meta runs the same allocator policy, including the cache flush the
    pool does when it hits the limit."""
    empty_cache("meta")
    reset_peak_stats("meta")
    set_memory_limit("meta", budget)
    try:
        fn("meta")
    except MemoryError:
        return None
    finally:
        set_memory_limit("meta", 0)
    return memory_stats("meta")


class Storage:
    __slots__ = ("dev", "ptr", "nbytes")

    def __init__(self, dev, nbytes):
        ptr = lib.sl_alloc(dev, nbytes)
        if not ptr:
            s = memory_stats(DEVICES[dev])
            raise MemoryError(
                f"soliton: out of memory on {DEVICES[dev]} allocating {_fmt(nbytes)} "
                f"(allocated {_fmt(s['allocated'])}, reserved {_fmt(s['reserved'])})"
            )
        self.dev, self.ptr, self.nbytes = dev, ptr, nbytes

    def __del__(self):
        try:
            lib.sl_free(self.dev, self.ptr, self.nbytes)
        except Exception:  # interpreter shutdown, or __init__ failed
            pass


# ---------------------------------------------------------------- tensor


class Tensor:
    __slots__ = ("storage", "shape", "dtype", "requires_grad", "grad", "_ctx", "_chain", "_dev")

    def __init__(self, storage, shape, dtype="float32", dev=None):
        self.storage, self.shape, self.dtype = storage, tuple(shape), dtype
        self._dev = storage.dev if storage is not None else dev
        self.requires_grad, self.grad, self._ctx, self._chain = False, None, None, None

    @property
    def device(self):
        return DEVICES[self._dev]

    @property
    def dev(self):
        return self._dev

    @property
    def ptr(self):
        """Reading the pointer is what forces a recorded chain to actually run."""
        if self._chain is not None:
            _run_chain(self)
        return self.storage.ptr

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return math.prod(self.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})"

    def requires_grad_(self, flag=True):
        self.requires_grad = flag
        return self

    def detach(self):
        return Tensor(self.storage, self.shape, self.dtype)

    def numpy(self):
        if self.dev == META:
            raise RuntimeError("soliton: meta tensors have no data")
        a = np.empty(self.shape, np.float32 if self.dtype == "float32" else np.int32)
        lib.sl_to_host(self.dev, a.ctypes.data, self.ptr, a.nbytes)
        _check(self.dev)
        return a

    def item(self):
        assert self.numel == 1, "item() needs a single element"
        return float("nan") if self.dev == META else float(self.numpy().reshape(-1)[0])

    def to(self, device):
        d = _dev(device)
        if d == self.dev:
            return self
        out = empty(self.shape, d, self.dtype)
        if META not in (d, self.dev):
            a = self.numpy()
            lib.sl_from_host(d, out.ptr, a.ctypes.data, a.nbytes)
        return out

    def backward(self, grad=None):
        backward(self, grad)

    __add__ = lambda a, b: add(a, b)
    __radd__ = lambda a, b: add(b, a)
    __sub__ = lambda a, b: sub(a, b)
    __rsub__ = lambda a, b: sub(b, a)
    __mul__ = lambda a, b: mul(a, b)
    __rmul__ = lambda a, b: mul(b, a)
    __truediv__ = lambda a, b: div(a, b)
    __neg__ = lambda a: scale(a, -1.0)
    __matmul__ = lambda a, b: matmul(a, b)
    reshape = lambda self, *shape: reshape(self, shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
    permute = lambda self, *dims: permute(self, dims)
    transpose = lambda self, a, b: transpose(self, a, b)
    sum = lambda self: sum_(self)
    mean = lambda self: scale(sum_(self), 1.0 / self.numel)


def empty(shape, device="cpu", dtype="float32"):
    shape = tuple(shape)
    return Tensor(Storage(_dev(device), max(math.prod(shape), 1) * 4), shape, dtype)


def full(shape, value, device="cpu"):
    t = empty(shape, device)
    lib.sl_fill(t.dev, t.ptr, t.numel, float(value))
    return t


def zeros(shape, device="cpu"):
    return full(shape, 0.0, device)


def tensor(data, device="cpu"):
    a = np.ascontiguousarray(data)
    is_int = a.dtype.kind in "iu"
    a = a.astype(np.int32 if is_int else np.float32, copy=False)
    t = empty(a.shape, device, "int32" if is_int else "float32")
    lib.sl_from_host(t.dev, t.ptr, a.ctypes.data, a.nbytes)
    return t


def normal(shape, std=1.0, device="cpu", rng=None):
    if _dev(device) == META:  # skip generating host data nobody will read
        return empty(shape, META)
    rng = rng if rng is not None else np.random.default_rng()
    return tensor(rng.standard_normal(tuple(shape), dtype=np.float32) * std, device)


# ---------------------------------------------------------------- autograd


@contextmanager
def no_grad():
    global _grad_on
    prev, _grad_on = _grad_on, False
    try:
        yield
    finally:
        _grad_on = prev


class Node:
    """One recorded op: its backward fn (with whatever that fn saved) and its parents' Nodes.
    A leaf parent (a parameter) is referenced as the Tensor itself; it outlives the graph anyway."""

    __slots__ = ("fn", "parents")

    def __init__(self, fn, parents):
        self.fn, self.parents = fn, parents


# ---------------------------------------------------------------- fused elementwise chains
#
# A chain of elementwise ops is recorded instead of run, then compiled into one CUDA kernel the first time
# anything reads the result. Three GPU round trips become one.
#
# The decision to fuse depends only on shapes and ops, never on the device, so meta and CPU take the same
# path: exactly one buffer is allocated per chain on every device, and memory plans stay exact.

_fusion = True
_jit_cache = {}
_BIN_EXPR = {0: "+", 1: "-", 2: "*", 3: "/"}


def set_fusion(on=True):
    """Turn elementwise fusion on or off. A plan and the run it predicts must use the same setting,
    because fusion changes which intermediates get allocated."""
    global _fusion
    _fusion = on


def _mat(x):
    """Force a pending chain to run, for the few places that touch .storage directly."""
    if isinstance(x, Tensor) and x._chain is not None:
        x.ptr
    return x


def _kernel_src(stmts, cur, nin):
    params = "".join(f", const float* v{k}" for k in range(nin))
    body = "\n".join(stmts)
    return (f'extern "C" __global__ void fused(long n, float* out{params}) {{\n'
            f"  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"  if (i >= n) return;\n{body}\n  out[i] = {cur};\n}}\n")


def _run_chain(t):
    stmts, cur, leaves, steps = t._chain
    t._chain = None
    ptrs = [leaf.ptr for leaf in leaves]  # leaves are always materialized first
    t.storage = Storage(t._dev, max(math.prod(t.shape), 1) * 4)
    n = t.numel
    if t._dev == CUDA:
        src = _kernel_src(stmts, cur, len(leaves))
        handle = _jit_cache.get(src)
        if handle is None:
            out = ctypes.c_void_p()
            rc = lib.sl_jit(CUDA, src.encode(), b"fused", ctypes.byref(out))
            if rc:
                raise RuntimeError(f"soliton: fused kernel failed to compile ({rc}): "
                                   f"{lib.sl_jit_log().decode()[:400]}\n{src}")
            handle = _jit_cache[src] = out.value
        vals = [ctypes.c_long(n), ctypes.c_void_p(t.storage.ptr)] + [ctypes.c_void_p(p) for p in ptrs]
        args = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
        lib.sl_jit_launch(CUDA, handle, n, args)
    elif t._dev == CPU:  # reference path: the same single buffer, updated in place step by step
        p, nd = t.storage.ptr, len(t.shape)
        sh, st = longs(t.shape), longs(_strides(t.shape))
        for step in steps:
            kind = step[0]
            if kind == "copy":
                lib.sl_copy(CPU, p, ptrs[step[1]], n * 4)
            elif kind == "scale":
                lib.sl_scale(CPU, p, p, n, step[1])
            elif kind == "addc":
                lib.sl_add_scalar(CPU, p, n, step[1])
            elif kind == "gelu":
                lib.sl_gelu(CPU, p, p, n)
            else:  # bin / rbin
                other = ptrs[step[2]]
                a, b = (p, other) if kind == "bin" else (other, p)
                lib.sl_binary(CPU, step[1], a, b, p, nd, sh, st, st)


def _chain_of(x):
    if x._chain is not None:
        stmts, cur, leaves, steps = x._chain
        return list(stmts), cur, list(leaves), list(steps)
    return [], "v0[i]", [x], [("copy", 0)]


def _record(x, shape, expr_of, step, other=None):
    stmts, cur, leaves, steps = _chain_of(x)
    ref = None
    if other is not None:
        leaves.append(other)
        ref = f"v{len(leaves) - 1}[i]"
        step = (*step, len(leaves) - 1)
    name = f"t{len(stmts)}"
    stmts.append(f"  float {name} = {expr_of(cur, ref)};")
    out = Tensor(None, shape, "float32", dev=x.dev)
    out._chain = (stmts, name, leaves, steps + [step])
    return out


def _fusable(a, b=None):
    if not (_fusion and isinstance(a, Tensor) and a.dtype == "float32"):
        return False
    if b is None:
        return True
    return isinstance(b, Tensor) and b.dtype == "float32" and b.shape == a.shape and b.dev == a.dev


def _ew2(op, a, b):
    """Elementwise binary forward, fused when both sides have the same shape."""
    if _fusable(a, b):
        _mat(b)  # v1 records linear chains only: the right-hand side must already exist
        return _record(a, a.shape, lambda cur, ref: f"{cur} {_BIN_EXPR[op]} {ref}", ("bin", op), b)
    return _binary(op, a, b)


def checkpoint(fn, *inputs):
    """Run ``fn(*inputs)`` without keeping its activations; recompute them during backward.
    Trades one extra forward of fn for its activation memory."""
    with no_grad():
        out = fn(*inputs)

    def bwd(g):
        global _grad_on
        detached = [Tensor(_mat(x).storage, x.shape, x.dtype).requires_grad_(x.requires_grad) for x in inputs]
        prev, _grad_on = _grad_on, True
        try:
            y = fn(*detached)
        finally:
            _grad_on = prev
        backward(y, g)  # parameters inside fn accumulate their .grad here
        return tuple(d.grad if d.requires_grad else None for d in detached)

    return _track(out, inputs, bwd)


def _track(out, parents, backward_fn):
    if _grad_on and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._ctx = Node(backward_fn, tuple(
            (p._ctx if p._ctx is not None else p) if p.requires_grad else None for p in parents))
    return out


def backward(root, grad=None):
    start = root._ctx if root._ctx is not None else root
    order, seen, stack = [], set(), [(start, False)]
    while stack:  # iterative DFS: topological order without recursion limits
        obj, done = stack.pop()
        if done:
            order.append(obj)
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        stack.append((obj, True))
        if isinstance(obj, Node):
            if obj.fn is None:
                raise RuntimeError("soliton: backward through a graph that was already freed")
            stack.extend((p, False) for p in obj.parents if p is not None and id(p) not in seen)

    grads = {id(start): grad if grad is not None else full(root.shape, 1.0, root.dev)}
    leaf_storages = set()
    with no_grad():
        for obj in reversed(order):
            g = grads.pop(id(obj), None)
            if g is None:
                continue
            if isinstance(obj, Tensor):
                # Backward fns may hand the same storage to several parents; leaf grads get mutated
                # in place later (accumulation), so they must own their storage.
                _mat(g)
                if id(g.storage) in leaf_storages:
                    g = clone(g)
                if obj.grad is None:
                    obj.grad = Tensor(g.storage, obj.shape)
                    leaf_storages.add(id(g.storage))
                else:
                    lib.sl_axpy(obj.dev, g.ptr, obj.grad.ptr, obj.numel, 1.0)
                continue
            gs = obj.fn(g)
            obj.fn = None  # frees this node's saved activations right away
            del g
            for p, pg in zip(obj.parents, gs):
                if p is None or pg is None:
                    continue
                old = grads.get(id(p))
                grads[id(p)] = pg if old is None else _ew2(0, old, pg)
            del gs


# ---------------------------------------------------------------- ops


def _strides(shape):
    out, acc = [], 1
    for n in reversed(shape):
        out.append(acc)
        acc *= n
    return out[::-1]


def _bstrides(shape, out_shape):
    """Strides of `shape` read as `out_shape` under broadcasting (0 on broadcast dims)."""
    pad = (1,) * (len(out_shape) - len(shape)) + tuple(shape)
    return [0 if n == 1 and o != 1 else s for n, o, s in zip(pad, out_shape, _strides(pad))]


def _lift(x, like):
    return x if isinstance(x, Tensor) else full((), float(x), like.dev)


def _binary(op, a, b):
    assert a.dev == b.dev, f"device mismatch: {a.device} vs {b.device}"
    shape = tuple(np.broadcast_shapes(a.shape, b.shape))
    o = empty(shape, a.dev)
    lib.sl_binary(a.dev, op, a.ptr, b.ptr, o.ptr, len(shape), longs(shape),
                  longs(_bstrides(a.shape, shape)), longs(_bstrides(b.shape, shape)))
    return o


def _unbroadcast(g, shape):
    shape = tuple(shape)
    if g.shape == shape:
        return g
    pad = (1,) * (g.ndim - len(shape)) + shape
    o = empty(shape, g.dev)
    _ok(lib.sl_reduce_sum(g.dev, g.ptr, o.ptr, o.numel, g.ndim, longs(g.shape), longs(_bstrides(pad, g.shape))))
    return o


def clone(x):
    o = empty(x.shape, x.dev, x.dtype)
    lib.sl_copy(x.dev, o.ptr, x.ptr, x.numel * 4)
    return _track(o, (x,), lambda g: (g,))


def add(a, b):
    if not isinstance(b, Tensor) and _fusable(a):  # x + constant
        c = float(b)
        return _track(_record(a, a.shape, lambda cur, _: f"{cur} + {c}f", ("addc", c)), (a,), lambda g: (g,))
    a, b = _lift(a, b), _lift(b, a)
    sa, sb = a.shape, b.shape
    return _track(_ew2(0, a, b), (a, b), lambda g: (_unbroadcast(g, sa), _unbroadcast(g, sb)))


def sub(a, b):
    if not isinstance(b, Tensor) and _fusable(a):  # x - constant
        c = -float(b)
        return _track(_record(a, a.shape, lambda cur, _: f"{cur} + {c}f", ("addc", c)), (a,), lambda g: (g,))
    a, b = _lift(a, b), _lift(b, a)
    sa, sb = a.shape, b.shape
    return _track(_ew2(1, a, b), (a, b), lambda g: (_unbroadcast(g, sa), _unbroadcast(scale(g, -1.0), sb)))


def scale(x, alpha):
    if _fusable(x):
        o = _record(x, x.shape, lambda cur, _: f"{cur} * {float(alpha)}f", ("scale", float(alpha)))
    else:
        o = empty(x.shape, x.dev)
        lib.sl_scale(x.dev, x.ptr, o.ptr, x.numel, alpha)
    return _track(o, (x,), lambda g: (scale(g, alpha),))


def mul(a, b):
    if not isinstance(b, Tensor):
        return scale(a, float(b))
    if not isinstance(a, Tensor):
        return scale(b, float(a))

    def bwd(g):
        return (_unbroadcast(_ew2(2, g, b), a.shape) if a.requires_grad else None,
                _unbroadcast(_ew2(2, g, a), b.shape) if b.requires_grad else None)

    return _track(_ew2(2, a, b), (a, b), bwd)


def div(a, b):
    if not isinstance(b, Tensor):
        return scale(a, 1.0 / float(b))
    a = _lift(a, b)

    def bwd(g):
        ga = _unbroadcast(_binary(3, g, b), a.shape) if a.requires_grad else None
        gb = _unbroadcast(scale(_binary(3, _binary(2, g, a), _binary(2, b, b)), -1.0), b.shape) if b.requires_grad else None
        return ga, gb

    return _track(_binary(3, a, b), (a, b), bwd)


def sum_(x):
    o = _unbroadcast(x, ()) if x.shape != () else clone(x)
    shape = x.shape

    def bwd(g):
        out = empty(shape, g.dev)
        lib.sl_strided_copy(g.dev, g.ptr, 0, out.ptr, len(shape), longs(shape), longs([0] * len(shape)))
        return (out,)

    return _track(o, (x,), bwd)


def _mm(a, b, ta, tb):
    """op(a) @ op(b) over matching leading dims, where op transposes the last two dims."""
    ar, ac = a.shape[-2:]
    br, bc = b.shape[-2:]
    n, k = (ac, ar) if ta else (ar, ac)
    k2, m = (bc, br) if tb else (br, bc)
    assert k == k2 and a.shape[:-2] == b.shape[:-2], f"matmul shape mismatch {a.shape} {b.shape}"
    o = empty((*a.shape[:-2], n, m), a.dev)
    lib.sl_matmul(a.dev, a.ptr, b.ptr, o.ptr, math.prod(a.shape[:-2]), n, k, m, ta, tb)
    return o


def matmul(a, b):
    return _track(_mm(a, b, 0, 0), (a, b), lambda g: (
        _mm(g, b, 0, 1) if a.requires_grad else None,
        _mm(a, g, 1, 0) if b.requires_grad else None,
    ))


def linear(x, w, b=None):
    """x (..., in) -> x @ w.T + b, without materializing w.T."""
    n_in, n_out = x.shape[-1], w.shape[0]
    rows = x.numel // n_in
    x2 = Tensor(_mat(x).storage, (rows, n_in))
    y = _mm(x2, w, 0, 1)
    if b is not None:  # in-place bias add: output is contiguous, so o == a is safe
        lib.sl_binary(y.dev, 0, y.ptr, b.ptr, y.ptr, 2, longs(y.shape), longs(_strides(y.shape)),
                      longs(_bstrides(b.shape, y.shape)))
    out = Tensor(y.storage, (*x.shape[:-1], n_out))
    xs = x.shape

    def bwd(g):
        g2 = Tensor(_mat(g).storage, (rows, n_out))
        gx = Tensor(_mm(g2, w, 0, 0).storage, xs) if x.requires_grad else None
        gw = _mm(g2, x2, 1, 0) if w.requires_grad else None
        if b is None:
            return gx, gw
        return gx, gw, (_unbroadcast(g2, b.shape) if b.requires_grad else None)

    return _track(out, (x, w) if b is None else (x, w, b), bwd)


def reshape(x, shape):
    shape = list(shape)
    if -1 in shape:
        i = shape.index(-1)
        shape[i] = x.numel // math.prod(s for j, s in enumerate(shape) if j != i)
    assert math.prod(shape) == x.numel, f"cannot reshape {x.shape} to {tuple(shape)}"
    xs = x.shape
    _mat(x)
    return _track(Tensor(x.storage, shape, x.dtype), (x,), lambda g: (Tensor(_mat(g).storage, xs),))


def permute(x, dims):
    dims = list(dims)
    st = _strides(x.shape)
    shape = [x.shape[d] for d in dims]
    o = empty(shape, x.dev)
    lib.sl_strided_copy(x.dev, x.ptr, 0, o.ptr, len(shape), longs(shape), longs(st[d] for d in dims))
    inv = [dims.index(i) for i in range(len(dims))]
    return _track(o, (x,), lambda g: (permute(g, inv),))


def transpose(x, a, b):
    dims = list(range(x.ndim))
    dims[a], dims[b] = dims[b], dims[a]
    return permute(x, dims)


def softmax(x):
    """Softmax over the last dim."""
    dim = x.shape[-1]
    outer = x.numel // dim
    o = empty(x.shape, x.dev)
    _ok(lib.sl_softmax(x.dev, x.ptr, o.ptr, outer, dim))
    ys, shape = o.storage, o.shape

    def bwd(g):
        dx = empty(shape, g.dev)
        _ok(lib.sl_softmax_bwd(g.dev, ys.ptr, g.ptr, dx.ptr, outer, dim))
        return (dx,)

    return _track(o, (x,), bwd)


def gelu(x):
    if _fusable(x):
        o = _record(x, x.shape, lambda cur, _: f"0.5f * ({cur}) * (1.0f + erff(({cur}) * 0.70710678f))", ("gelu",))
        return _track(o, (x,), lambda g: (_gelu_bwd(x, g),))
    o = empty(x.shape, x.dev)
    lib.sl_gelu(x.dev, x.ptr, o.ptr, x.numel)
    return _track(o, (x,), lambda g: (_gelu_bwd(x, g),))


def _gelu_bwd(x, g):
    """Backward reads x, so a fused chain that produced x is recomputed here. Deterministic, and the meta
    device replays the same allocation."""
    dx = empty(x.shape, g.dev)
    lib.sl_gelu_bwd(g.dev, x.ptr, g.ptr, dx.ptr, x.numel)
    return dx


def layernorm(x, w, b, eps=1e-5):
    c = x.shape[-1]
    outer = x.numel // c
    y, mean, invstd = empty(x.shape, x.dev), empty((outer,), x.dev), empty((outer,), x.dev)
    _ok(lib.sl_layernorm(x.dev, x.ptr, w.ptr, b.ptr, y.ptr, mean.ptr, invstd.ptr, outer, c, eps))

    def bwd(g):
        dx, dw, db = empty(x.shape, g.dev), empty(w.shape, g.dev), empty(b.shape, g.dev)
        _ok(lib.sl_layernorm_bwd(g.dev, g.ptr, x.ptr, w.ptr, mean.ptr, invstd.ptr, dx.ptr, dw.ptr, db.ptr, outer, c))
        return dx, dw, db

    return _track(y, (x, w, b), bwd)


def embedding(w, idx):
    assert idx.dtype == "int32", "embedding indices must be int32"
    vocab, c = w.shape
    o = empty((*idx.shape, c), w.dev)
    lib.sl_embedding(w.dev, w.ptr, idx.ptr, o.ptr, idx.numel, c)

    def bwd(g):
        dw = empty(w.shape, g.dev)
        if lib.sl_embedding_bwd(g.dev, g.ptr, idx.ptr, dw.ptr, vocab, idx.numel, c) != 0:
            raise MemoryError("soliton: out of memory in embedding backward")
        return dw, None

    return _track(o, (w, idx), bwd)


def cross_entropy(logits, targets):
    """Mean cross entropy. logits (..., V), int32 targets (...)."""
    v = logits.shape[-1]
    n = logits.numel // v
    assert targets.dtype == "int32" and targets.numel == n
    err = ctypes.c_int()
    loss = lib.sl_cross_entropy(logits.dev, logits.ptr, targets.ptr, n, v, ctypes.byref(err))
    if err.value:
        raise MemoryError("soliton: out of memory in cross_entropy")
    out = full((), loss if logits.dev != META else 0.0, logits.dev)

    def bwd(g):
        # Materialize g on every device: reading it only off-meta would make the plan disagree with the run.
        _mat(g)
        dl = empty(logits.shape, g.dev)
        _ok(lib.sl_cross_entropy_bwd(g.dev, logits.ptr, targets.ptr, dl.ptr, n, v, g.item() if g.dev != META else 1.0))
        return dl, None

    return _track(out, (logits, targets), bwd)


def attention(qkv, scale=None):
    """Fused causal attention. qkv is packed (B, T, 3, H, hs) as one projection produces it; returns (B, T, H, hs).

    Never materializes the (B, T, T) scores: they are recomputed tile by tile in backward."""
    b, t, three, h, hs = qkv.shape
    assert three == 3, "attention expects packed (B, T, 3, H, hs)"
    scale = scale if scale is not None else 1.0 / math.sqrt(hs)
    out, lse = empty((b, t, h, hs), qkv.dev), empty((b, h, t), qkv.dev)
    _ok(lib.sl_attention(qkv.dev, qkv.ptr, out.ptr, lse.ptr, b, t, h, hs, scale))
    ostore, shape = out.storage, qkv.shape

    def bwd(g):
        dqkv = empty(shape, g.dev)
        _ok(lib.sl_attention_bwd(g.dev, qkv.ptr, ostore.ptr, lse.ptr, g.ptr, dqkv.ptr, b, t, h, hs, scale))
        return (dqkv,)

    return _track(out, (qkv,), bwd)


def sumsq(x):
    return lib.sl_sumsq(x.dev, x.ptr, x.numel)
