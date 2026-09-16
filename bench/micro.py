"""Workload benchmarks across frameworks. One process per (framework, workload); prints one RESULT line.

    python bench/micro.py --framework soliton --workload mlp

Workloads
  mlp      4-layer MLP, forward + backward + SGD step        (op overhead, dense gemms)
  chain    elementwise chain on 16M floats, forward only     (kernel fusion / bandwidth)
  attn     causal attention fwd+bwd at several seq lengths   (attention implementation)
  dynamic  20 steps whose sequence length changes each step  (compile cost on shape change)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEQS = (512, 1024, 2048)
DYN_SEQS = [256, 384, 512, 640, 768, 896, 1024, 512, 768, 256] * 2


def timed(step, reps, warmup=3):
    for _ in range(warmup):
        step()
    t = time.perf_counter()
    for _ in range(reps):
        step()
    return 1000 * (time.perf_counter() - t) / reps


# ------------------------------------------------------------------ soliton

def soliton_bench(workload):
    sys.path[:0] = [os.path.join(ROOT, "python"), os.path.join(ROOT, "examples")]
    import soliton as sl
    from soliton._C import lib
    from soliton import nn

    sync = lambda: lib.sl_sync(2)  # noqa: E731
    rng = np.random.default_rng(0)

    if workload == "chain":
        n = 1 << 24
        x, y = (sl.tensor(rng.standard_normal(n, dtype=np.float32), "cuda") for _ in range(2))

        def step():
            out = sl.gelu(x * 2.0 + y) * 0.5 - 1.0
            out.ptr
        return timed(lambda: (step(), sync()), 20)

    if workload == "mlp":
        d, b = 4096, 4096
        layers = [nn.Linear(d, d, device="cuda", rng=rng) for _ in range(4)]
        params = [p for l in layers for p in (l.weight, l.bias)]
        x = sl.tensor(rng.standard_normal((b, d), dtype=np.float32) * 0.1, "cuda")

        def step():
            h = x
            for l in layers:
                h = sl.gelu(l(h))
            loss = sl.scale(sl.sum_(sl.mul(h, h)), 1.0 / (b * d))
            loss.backward()
            for p in params:
                if p.grad is not None:
                    lib.sl_axpy(p.dev, p.grad.ptr, p.ptr, p.numel, -1e-4)
                    p.grad = None
        return timed(lambda: (step(), sync()), 10)

    if workload == "attn":
        out = {}
        for t in SEQS:
            b, h, hs = 4, 12, 64
            qkv = sl.tensor(rng.standard_normal((b, t, 3, h, hs), dtype=np.float32), "cuda").requires_grad_()
            g = sl.tensor(rng.standard_normal((b, t, h, hs), dtype=np.float32), "cuda")

            def step(qkv=qkv, g=g):
                o = sl.attention(qkv)
                o.backward(g)
                qkv.grad = None
            out[t] = timed(lambda: (step(), sync()), 10)
        return out

    if workload == "dynamic":
        from gpt2 import GPT, CONFIGS
        from soliton.optim import AdamW
        cfg = {**CONFIGS["124M"], "n_layer": 2, "device": "cuda"}
        model = GPT(**cfg)
        opt = AdamW(model.parameters())
        t0 = time.perf_counter()
        for t in DYN_SEQS:
            tok = rng.integers(0, 50257, (2, t + 1)).astype(np.int32)
            opt.zero_grad()
            model(sl.tensor(tok[:, :-1], "cuda"), sl.tensor(tok[:, 1:], "cuda")).backward()
            opt.step()
        sync()
        return 1000 * (time.perf_counter() - t0)


# ------------------------------------------------------------------ pytorch (eager or compiled)

def torch_bench(workload, compiled):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"
    sync = torch.cuda.synchronize
    rng = np.random.default_rng(0)

    if workload == "chain":
        n = 1 << 24
        x, y = (torch.tensor(rng.standard_normal(n, dtype=np.float32), device=dev) for _ in range(2))
        f = lambda: F.gelu(x * 2.0 + y) * 0.5 - 1.0  # noqa: E731
        if compiled:
            f = torch.compile(f)
        return timed(lambda: (f(), sync()), 20)

    if workload == "mlp":
        d, b = 4096, 4096
        layers = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(4)]).to(dev)
        x = torch.tensor(rng.standard_normal((b, d), dtype=np.float32) * 0.1, device=dev)

        def fwd(x):
            h = x
            for l in layers:
                h = F.gelu(l(h))
            return (h * h).mean()

        fn = torch.compile(fwd) if compiled else fwd

        def step():
            loss = fn(x)
            loss.backward()
            with torch.no_grad():
                for p in layers.parameters():
                    p -= 1e-4 * p.grad
                    p.grad = None
        return timed(lambda: (step(), sync()), 10)

    if workload == "attn":
        out = {}
        for t in SEQS:
            b, h, hs = 4, 12, 64
            qkv = torch.tensor(rng.standard_normal((b, t, 3, h, hs), dtype=np.float32), device=dev, requires_grad=True)
            g = torch.tensor(rng.standard_normal((b, t, h, hs), dtype=np.float32), device=dev)

            def fwd(qkv=qkv):
                q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
                return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)

            fn = torch.compile(fwd) if compiled else fwd

            def step(fn=fn, qkv=qkv, g=g):
                o = fn()
                o.backward(g)
                qkv.grad = None
            out[t] = timed(lambda: (step(), sync()), 10)
        return out

    if workload == "dynamic":
        sys.path.insert(0, HERE)
        import torch_gpt2 as tg
        tg.L = 2
        model = tg.GPT().to(dev)
        opt = tg.make_opt(model)
        fn = torch.compile(model) if compiled else model
        t0 = time.perf_counter()
        for t in DYN_SEQS:
            tok = torch.from_numpy(rng.integers(0, 50257, (2, t + 1)).astype(np.int64)).to(dev)
            loss = fn(tok[:, :-1], tok[:, 1:])
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        sync()
        return 1000 * (time.perf_counter() - t0)


# ------------------------------------------------------------------ jax

def jax_bench(workload):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    sync = lambda r: jax.block_until_ready(r)  # noqa: E731

    if workload == "chain":
        n = 1 << 24
        x, y = (jnp.asarray(rng.standard_normal(n, dtype=np.float32)) for _ in range(2))
        f = jax.jit(lambda x, y: jax.nn.gelu(x * 2.0 + y, approximate=False) * 0.5 - 1.0)
        return timed(lambda: sync(f(x, y)), 20)

    if workload == "mlp":
        d, b = 4096, 4096
        ws = [jnp.asarray(rng.standard_normal((d, d), dtype=np.float32) * 0.02) for _ in range(4)]
        bs = [jnp.zeros(d) for _ in range(4)]
        x = jnp.asarray(rng.standard_normal((b, d), dtype=np.float32) * 0.1)

        def loss_fn(ws, bs, x):
            h = x
            for w, bb in zip(ws, bs):
                h = jax.nn.gelu(h @ w + bb, approximate=False)
            return jnp.mean(h * h)

        grad = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

        def step():
            gw, gb = grad(ws, bs, x)
            for i in range(4):
                ws[i] = ws[i] - 1e-4 * gw[i]
                bs[i] = bs[i] - 1e-4 * gb[i]
            sync(ws[0])
        return timed(step, 10)

    if workload == "attn":
        import math
        out = {}
        for t in SEQS:
            b, h, hs = 4, 12, 64
            qkv = jnp.asarray(rng.standard_normal((b, t, 3, h, hs), dtype=np.float32))
            g = jnp.asarray(rng.standard_normal((b, t, h, hs), dtype=np.float32))
            mask = jnp.triu(jnp.full((t, t), -jnp.inf), 1)

            def fwd(qkv):
                q, k, v = (jnp.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3))
                att = jax.nn.softmax((q @ jnp.swapaxes(k, -1, -2)) / math.sqrt(hs) + mask, axis=-1)
                return jnp.transpose(att @ v, (0, 2, 1, 3))

            vjp = jax.jit(lambda qkv, g: jax.vjp(fwd, qkv)[1](g)[0])
            out[t] = timed(lambda: sync(vjp(qkv, g)), 10)
        return out

    if workload == "dynamic":
        sys.path.insert(0, HERE)
        import jax_gpt2 as jg
        jg.L = 2
        params = jg.init()
        import optax
        opt = optax.adamw(6e-4, b1=0.9, b2=0.95, weight_decay=0.1)
        state = opt.init(params)

        @jax.jit
        def step(p, s, idx, tgt):
            loss, g = jax.value_and_grad(jg.loss_fn)(p, idx, tgt)
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, loss

        t0 = time.perf_counter()
        for t in DYN_SEQS:
            tok = rng.integers(0, jg.V, (2, t + 1)).astype(np.int32)
            params, state, loss = step(params, state, jnp.asarray(tok[:, :-1]), jnp.asarray(tok[:, 1:]))
        sync(loss)
        return 1000 * (time.perf_counter() - t0)


# ------------------------------------------------------------------ tinygrad

def tinygrad_bench(workload):
    from tinygrad import Tensor, dtypes
    from tinygrad.device import Device

    rng = np.random.default_rng(0)
    Tensor.manual_seed(0)
    _dev = Device[Device.DEFAULT]

    def sync(t):  # wait for the GPU without copying the result to the host, like the other adapters
        t.realize()
        _dev.synchronize()

    if workload == "chain":
        n = 1 << 24
        x = Tensor(rng.standard_normal(n, dtype=np.float32))
        y = Tensor(rng.standard_normal(n, dtype=np.float32))
        return timed(lambda: sync((x * 2.0 + y).gelu() * 0.5 - 1.0), 20)

    if workload == "mlp":
        d, b = 4096, 4096
        ws = [Tensor(rng.standard_normal((d, d), dtype=np.float32) * 0.02, requires_grad=True) for _ in range(4)]
        x = Tensor(rng.standard_normal((b, d), dtype=np.float32) * 0.1)
        Tensor.training = True

        def step():
            h = x
            for w in ws:
                h = (h @ w).gelu()
            loss = (h * h).mean()
            for w in ws:
                w.grad = None
            loss.backward()
            for w in ws:
                w.assign(w.detach() - 1e-4 * w.grad)
            sync(ws[0])
        return timed(step, 10)

    if workload == "attn":
        out = {}
        for t in SEQS:
            b, h, hs = 4, 12, 64
            q, k, v = (Tensor(rng.standard_normal((b, h, t, hs), dtype=np.float32), requires_grad=True)
                       for _ in range(3))
            Tensor.training = True

            def step(q=q, k=k, v=v):
                o = q.scaled_dot_product_attention(k, v, is_causal=True)
                loss = o.sum()
                for p in (q, k, v):
                    p.grad = None
                loss.backward()
                sync(q.grad)
            out[t] = timed(step, 5)
        return out

    return None  # dynamic: no GPT-2 port here


# ------------------------------------------------------------------ tensorflow

def tf_bench(workload):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("tensorflow: no GPU visible")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    rng = np.random.default_rng(0)

    if workload == "chain":
        n = 1 << 24
        x = tf.constant(rng.standard_normal(n, dtype=np.float32))
        y = tf.constant(rng.standard_normal(n, dtype=np.float32))

        @tf.function
        def f():
            return tf.nn.gelu(x * 2.0 + y, approximate=False) * 0.5 - 1.0
        return timed(lambda: f().numpy()[:1], 20)

    if workload == "mlp":
        d, b = 4096, 4096
        ws = [tf.Variable(rng.standard_normal((d, d), dtype=np.float32) * 0.02) for _ in range(4)]
        bs = [tf.Variable(tf.zeros(d)) for _ in range(4)]
        x = tf.constant(rng.standard_normal((b, d), dtype=np.float32) * 0.1)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                h = x
                for w, bb in zip(ws, bs):
                    h = tf.nn.gelu(tf.matmul(h, w) + bb, approximate=False)
                loss = tf.reduce_mean(h * h)
            gs = tape.gradient(loss, ws + bs)
            for v, g in zip(ws + bs, gs):
                v.assign_sub(1e-4 * g)
            return loss
        return timed(lambda: step().numpy(), 10)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True,
                    choices=("soliton", "pytorch", "pytorch-compile", "jax", "tensorflow", "tinygrad"))
    ap.add_argument("--workload", required=True, choices=("mlp", "chain", "attn", "dynamic"))
    args = ap.parse_args()
    fns = {
        "soliton": lambda w: soliton_bench(w),
        "pytorch": lambda w: torch_bench(w, False),
        "pytorch-compile": lambda w: torch_bench(w, True),
        "jax": lambda w: jax_bench(w),
        "tensorflow": lambda w: tf_bench(w),
        "tinygrad": lambda w: tinygrad_bench(w),
    }
    res = dict(framework=args.framework, workload=args.workload)
    try:
        value = fns[args.framework](args.workload)
        res["result"] = value if value is not None else "unsupported"
    except Exception as e:  # a framework failing one workload shouldn't kill the sweep
        res["error"] = f"{type(e).__name__}: {e}"[:300]
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
