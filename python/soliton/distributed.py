"""Data parallel over NCCL: one process per GPU, gradients averaged after backward.

Collectives allocate nothing from the pool (NCCL uses its own buffers), so a single-rank meta dry run is the
exact per-rank memory plan.
"""
import ctypes
import os
import time

from . import tensor as T
from ._C import lib

_world = 1


def init(rank, world, init_file):
    """Rank 0 creates the NCCL id and publishes it through init_file; the other ranks wait for it."""
    global _world
    T.set_device(rank)
    T._dev(T.CUDA)
    buf = ctypes.create_string_buffer(128)
    if rank == 0:
        if lib.sl_dist_unique_id(buf):
            raise RuntimeError("soliton: ncclGetUniqueId failed")
        with open(init_file + ".tmp", "wb") as f:
            f.write(buf.raw)
        os.replace(init_file + ".tmp", init_file)  # readers never see a partial id
    else:
        while not os.path.exists(init_file):
            time.sleep(0.05)
        with open(init_file, "rb") as f:
            ctypes.memmove(buf, f.read(128), 128)
    if lib.sl_dist_init(rank, world, buf):
        raise RuntimeError(f"soliton: NCCL init failed on rank {rank}")
    _world = world


def world_size():
    return _world


def all_reduce_grads(params):
    """Average gradients across ranks in one NCCL group. No-op for a single process."""
    grads = [p.grad for p in params if p.grad is not None]
    if _world == 1 or not grads:
        return
    dev = grads[0].dev
    ok = lib.sl_dist_group(dev, 1) == 0
    for g in grads:
        ok = lib.sl_dist_allreduce_mean(dev, g.ptr, g.numel) == 0 and ok
    ok = lib.sl_dist_group(dev, 0) == 0 and ok
    if not ok:
        raise RuntimeError("soliton: NCCL all-reduce failed")
