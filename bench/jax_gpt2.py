"""GPT-2 training step in JAX, same math as examples/gpt2.py. Prints one RESULT json line."""
import argparse
import json
import math
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # otherwise JAX grabs 75% up front

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

V, CTX, L, H, C = 50257, 1024, 12, 12, 768


def init(seed=0):
    rng = np.random.default_rng(seed)
    n = lambda *s, std=0.02: jnp.asarray(rng.standard_normal(s, dtype=np.float32) * std)  # noqa: E731
    lin = lambda i, o, std=0.02: (n(o, i, std=std), jnp.zeros(o))  # noqa: E731
    ln = lambda: (jnp.ones(C), jnp.zeros(C))  # noqa: E731
    pstd = 0.02 / math.sqrt(2 * L)
    blocks = [dict(ln1=ln(), ln2=ln(), q=lin(C, C), k=lin(C, C), v=lin(C, C), proj=lin(C, C, pstd),
                   fc=lin(C, 4 * C), fc_proj=lin(4 * C, C, pstd)) for _ in range(L)]
    return dict(wte=n(V, C), wpe=n(CTX, C), blocks=blocks, lnf=ln())


def layernorm(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * w + b


def linear(x, wb):
    return x @ wb[0].T + wb[1]


def loss_fn(p, idx, tgt):
    b, t = idx.shape
    hs = C // H
    x = p["wte"][idx] + p["wpe"][:t]
    mask = jnp.triu(jnp.full((t, t), -jnp.inf), 1)
    for blk in p["blocks"]:
        y = layernorm(x, *blk["ln1"])
        q, k, v = (linear(y, blk[n]).reshape(b, t, H, hs).transpose(0, 2, 1, 3) for n in "qkv")
        att = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hs) + mask, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, C)
        x = x + linear(y, blk["proj"])
        x = x + linear(jax.nn.gelu(linear(layernorm(x, *blk["ln2"]), blk["fc"]), approximate=False), blk["fc_proj"])
    logits = layernorm(x, *p["lnf"]) @ p["wte"].T
    return optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()
    out = dict(framework="jax", batch=args.batch, seq=args.seq, oom=False)
    try:
        params = init()
        decay = jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(6e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1, mask=decay))
        state = opt.init(params)

        def step(p, s, idx, tgt):
            loss, g = jax.value_and_grad(loss_fn)(p, idx, tgt)
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, loss

        step = jax.jit(step, donate_argnums=(0, 1))

        rng = np.random.default_rng(0)
        losses, times = [], []
        for i in range(args.steps):
            tok = rng.integers(0, V, size=(args.batch, args.seq + 1)).astype(np.int32)
            t0 = time.time()
            params, state, loss = step(params, state, jnp.asarray(tok[:, :-1]), jnp.asarray(tok[:, 1:]))
            losses.append(float(loss))  # forces sync
            times.append(time.time() - t0)
        warm = times[2:] or times
        out.update(loss_first=losses[0], loss_last=losses[-1], first_step_s=times[0],
                   step_ms=1000 * sum(warm) / len(warm), tok_s=args.batch * args.seq * len(warm) / sum(warm),
                   native_peak=None)
    except Exception as e:  # XlaRuntimeError RESOURCE_EXHAUSTED
        msg = str(e)
        if "RESOURCE_EXHAUSTED" not in msg and "out of memory" not in msg.lower():
            raise
        out.update(oom=True, error=msg.splitlines()[0][:200])
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
