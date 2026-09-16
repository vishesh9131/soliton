"""Two-GPU data parallel must produce the same parameters as one process on the concatenated batch."""
import multiprocessing as mp
import os
import tempfile

import numpy as np
import pytest

from soliton._C import lib

STEPS, B, T = 3, 4, 32

pytestmark = pytest.mark.skipif(lib.sl_cuda_count() < 2, reason="needs 2 GPUs")


def data():
    return np.random.default_rng(0).integers(0, 512, size=(STEPS, 2, 2 * B, T)).astype(np.int32)


def train(rank, world, init_file, q):
    import soliton as sl
    from soliton import distributed
    from soliton.optim import AdamW
    from gpt2 import GPT, CONFIGS

    if world > 1:
        distributed.init(rank, world, init_file)
    else:
        sl.set_device(0)
    model = GPT(**{**CONFIGS["tiny"], "device": "cuda"})
    opt = AdamW(model.parameters(), lr=1e-3)
    for idx, tgt in data():
        lo, hi = (rank * B, (rank + 1) * B) if world > 1 else (0, 2 * B)
        opt.zero_grad()
        model(sl.tensor(idx[lo:hi], "cuda"), sl.tensor(tgt[lo:hi], "cuda")).backward()
        distributed.all_reduce_grads(opt.params)
        opt.step()
    q.put((rank, world, [p.numpy() for p in model.parameters()]))


def test_data_parallel_matches_single_process():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = os.path.join(tempfile.mkdtemp(), "nccl_id")
    procs = [ctx.Process(target=train, args=(r, 2, init_file, q)) for r in range(2)]
    procs.append(ctx.Process(target=train, args=(0, 1, None, q)))
    for p in procs:
        p.start()
    results = {(r, w): params for r, w, params in (q.get(timeout=300) for _ in procs)}
    for p in procs:
        p.join(timeout=60)
    single = results[(0, 1)]
    for rank in (0, 1):
        for a, b in zip(results[(rank, 2)], single):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
