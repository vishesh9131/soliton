"""GPT-2 training step in TensorFlow (tf.function graph mode), same math as examples/gpt2.py."""
import argparse
import json
import math
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

V, CTX, L, H, C = 50257, 1024, 12, 12, 768


def init(seed=0):
    rng = np.random.default_rng(seed)
    n = lambda *s, std=0.02: tf.Variable(rng.standard_normal(s, dtype=np.float32) * std)  # noqa: E731
    z = lambda *s: tf.Variable(tf.zeros(s))  # noqa: E731
    o = lambda *s: tf.Variable(tf.ones(s))  # noqa: E731
    lin = lambda i, out, std=0.02: (n(out, i, std=std), z(out))  # noqa: E731
    pstd = 0.02 / math.sqrt(2 * L)
    blocks = [dict(ln1=(o(C), z(C)), ln2=(o(C), z(C)), q=lin(C, C), k=lin(C, C), v=lin(C, C),
                   proj=lin(C, C, pstd), fc=lin(C, 4 * C), fc_proj=lin(4 * C, C, pstd)) for _ in range(L)]
    return dict(wte=n(V, C), wpe=n(CTX, C), blocks=blocks, lnf=(o(C), z(C)))


def layernorm(x, w, b):
    m = tf.reduce_mean(x, -1, keepdims=True)
    v = tf.reduce_mean(tf.square(x - m), -1, keepdims=True)
    return (x - m) / tf.sqrt(v + 1e-5) * w + b


def linear(x, wb):
    return tf.matmul(x, wb[0], transpose_b=True) + wb[1]


def loss_fn(p, idx, tgt):
    b, t = idx.shape
    hs = C // H
    x = tf.gather(p["wte"], idx) + p["wpe"][:t]
    mask = tf.constant(np.triu(np.full((t, t), -np.inf, np.float32), 1))
    for blk in p["blocks"]:
        y = layernorm(x, *blk["ln1"])
        q, k, v = (tf.transpose(tf.reshape(linear(y, blk[n]), (b, t, H, hs)), (0, 2, 1, 3)) for n in "qkv")
        att = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / math.sqrt(hs) + mask, axis=-1)
        y = tf.reshape(tf.transpose(tf.matmul(att, v), (0, 2, 1, 3)), (b, t, C))
        x = x + linear(y, blk["proj"])
        x = x + linear(tf.nn.gelu(linear(layernorm(x, *blk["ln2"]), blk["fc"]), approximate=False), blk["fc_proj"])
    logits = tf.matmul(layernorm(x, *p["lnf"]), p["wte"], transpose_b=True)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt, logits=logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor cores for matmul (lower precision)")
    args = ap.parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("tensorflow: no GPU visible, refusing to benchmark on CPU")
    for gpu in gpus:
        if args.budget_gib:  # hard cap on TF's allocator
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(args.budget_gib * 1024))])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)  # otherwise TF grabs the whole GPU
    tf.config.experimental.enable_tensor_float_32_execution(args.tf32)
    out = dict(framework="tensorflow", batch=args.batch, seq=args.seq, oom=False)
    try:
        params = init()
        vars_ = tf.nest.flatten(params)
        m = [tf.Variable(tf.zeros_like(v)) for v in vars_]
        s = [tf.Variable(tf.zeros_like(v)) for v in vars_]
        t_step = tf.Variable(0.0)
        lr, b1, b2, eps, wd = 6e-4, 0.9, 0.95, 1e-8, 0.1

        @tf.function
        def step(idx, tgt):
            with tf.GradientTape() as tape:
                loss = loss_fn(params, idx, tgt)
            grads, _ = tf.clip_by_global_norm(tape.gradient(loss, vars_), 1.0)
            t_step.assign_add(1.0)
            bc1, bc2 = 1 - b1 ** t_step, 1 - b2 ** t_step
            for v, g, mv, sv in zip(vars_, grads, m, s):
                mv.assign(b1 * mv + (1 - b1) * g)
                sv.assign(b2 * sv + (1 - b2) * tf.square(g))
                upd = (mv / bc1) / (tf.sqrt(sv / bc2) + eps) + (wd * v if len(v.shape) >= 2 else 0.0)
                v.assign_sub(lr * upd)
            return loss

        rng = np.random.default_rng(0)
        losses, times = [], []
        for i in range(args.steps):
            tok = rng.integers(0, V, size=(args.batch, args.seq + 1)).astype(np.int32)
            t0 = time.time()
            loss = step(tf.constant(tok[:, :-1]), tf.constant(tok[:, 1:]))
            losses.append(float(loss.numpy()))
            times.append(time.time() - t0)
        warm = times[2:] or times
        out.update(loss_first=losses[0], loss_last=losses[-1], first_step_s=times[0],
                   step_ms=1000 * sum(warm) / len(warm), tok_s=args.batch * args.seq * len(warm) / sum(warm),
                   native_peak=tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        out.update(oom=True, error=str(e).splitlines()[0][:200])
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
