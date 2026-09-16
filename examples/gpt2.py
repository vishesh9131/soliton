"""GPT-2 on Soliton: plan memory on the meta device, then train and compare the plan with reality.

    python examples/gpt2.py --plan-only                   # memory plan, no GPU touched
    CUDA_VISIBLE_DEVICES=2 python examples/gpt2.py --data data/fineweb_edu --steps 200
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import soliton as sl  # noqa: E402
from soliton import distributed, nn  # noqa: E402
from soliton.optim import AdamW, clip_grad_norm  # noqa: E402


class Block(nn.Module):
    def __init__(self, c, n_head, n_layer, device, rng, idx=0):
        self.n_head, self.idx, self.ckpt = n_head, idx, frozenset()
        proj_std = 0.02 / math.sqrt(2 * n_layer)
        self.ln1, self.ln2 = nn.LayerNorm(c, device), nn.LayerNorm(c, device)
        self.qkv = nn.Linear(c, 3 * c, device=device, rng=rng)  # one gemm, packed for fused attention
        self.proj = nn.Linear(c, c, device=device, std=proj_std, rng=rng)
        self.fc = nn.Linear(c, 4 * c, device=device, rng=rng)
        self.fc_proj = nn.Linear(4 * c, c, device=device, std=proj_std, rng=rng)

    def attn(self, x):
        b, t, c = x.shape
        h, hs = self.n_head, c // self.n_head
        # Packed (B,T,3,H,hs) goes straight into fused causal attention: no transposes, no (B,H,T,T) tensor.
        qkv = sl.reshape(self.qkv(self.ln1(x)), (b, t, 3, h, hs))
        return x + self.proj(sl.reshape(sl.attention(qkv), (b, t, c)))

    def mlp(self, x):
        return x + self.fc_proj(sl.gelu(self.fc(self.ln2(x))))

    def both(self, x):
        return self.mlp(self.attn(x))

    def forward(self, x):
        """Recomputable as halves or as a whole block. Whole is cheaper in memory (one boundary activation
        instead of two), halves are finer, so the solver gets to choose per block."""
        if f"{self.idx}.block" in self.ckpt:
            return sl.checkpoint(self.both, x)
        x = sl.checkpoint(self.attn, x) if f"{self.idx}.attn" in self.ckpt else self.attn(x)
        return sl.checkpoint(self.mlp, x) if f"{self.idx}.mlp" in self.ckpt else self.mlp(x)


class GPT(nn.Module):
    def __init__(self, vocab=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, device="cpu", seed=0,
                 checkpoint=0):
        rng = np.random.default_rng(seed)
        self.device = device
        self.checkpoint = checkpoint  # recompute the first `checkpoint` blocks in backward
        self.wte = nn.Embedding(vocab, n_embd, device, rng=rng)
        self.wpe = nn.Embedding(block_size, n_embd, device, rng=rng)
        self.blocks = [Block(n_embd, n_head, n_layer, device, rng, i) for i in range(n_layer)]
        for b in self.blocks:
            b.ckpt = checkpoint_units(checkpoint, n_layer)
        self.ln_f = nn.LayerNorm(n_embd, device)

    def forward(self, idx, targets):
        _, t = idx.shape
        x = self.wte(idx) + self.wpe(sl.tensor(np.arange(t, dtype=np.int32), self.device))
        for block in self.blocks:
            x = block(x)
        logits = sl.linear(self.ln_f(x), self.wte.weight)  # tied weights
        return sl.cross_entropy(logits, targets)


CONFIGS = {
    "124M": dict(n_layer=12, n_head=12, n_embd=768),
    "tiny": dict(vocab=512, block_size=64, n_layer=2, n_head=2, n_embd=32),
}


def train_step(model, opt, batches, clip=1.0):
    """One optimizer step over micro-batches. Identical on meta and real devices.

    Every allocation made here is also released here, which is what lets one arena plan repeat per step."""
    total = 0.0
    for idx, tgt in batches:
        loss = model(idx, tgt)
        total += loss.item() / len(batches)
        sl.scale(loss, 1.0 / len(batches)).backward()
        del loss
    distributed.all_reduce_grads(opt.params)  # no-op for one process
    norm = clip_grad_norm(opt.params, clip)
    opt.step()
    opt.zero_grad()  # at the end, so gradients do not outlive the step
    return total, norm


def run(device, cfg, batch, seq, accum, steps, get_batch=None, lr=6e-4, log=None, checkpoint=0):
    """Build model + optimizer on `device` and run `steps` steps. Returns memory stats and losses."""
    sl.empty_cache(device)
    sl.reset_peak_stats(device)
    model = GPT(**{**CONFIGS[cfg], "device": device, "checkpoint": checkpoint})
    opt = AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        # The step boundary has to come before the batch tensors are allocated: they are part of the
        # repeating body, so the arena cursor must rewind ahead of them.
        sl.arena.mark(device)
        if get_batch is None:  # meta: shapes only, but one allocation per tensor exactly like the real path
            batches = [(sl.empty((batch, seq), device, "int32"), sl.empty((batch, seq), device, "int32"))
                       for _ in range(accum)]
        else:
            batches = [tuple(sl.tensor(a, device) for a in get_batch()) for _ in range(accum)]
        t0 = time.time()
        loss, norm = train_step(model, opt, batches)
        del batches
        losses.append(loss)
        if log:
            sl.synchronize()
            log(step, loss, norm, time.time() - t0)
    stats = sl.memory_stats(device)
    return stats, losses, model, opt


def checkpoint_units(checkpoint, n_layer):
    """An int means the first k blocks entirely; otherwise an explicit set of "<block>.attn"/"<block>.mlp" ids."""
    if isinstance(checkpoint, int):
        return frozenset(f"{i}.block" for i in range(checkpoint))
    return frozenset(checkpoint)


def units_of(cfg):
    return [f"{i}.{p}" for i in range(CONFIGS[cfg]["n_layer"]) for p in ("attn", "mlp", "block")]


def fit_checkpoints(cfg, batch, seq, accum, budget, cost=None):
    """Least-cost set of attention/MLP halves to recompute so the step fits `budget`. Returns (set, peak)."""
    make = lambda ck: (lambda dev: run(dev, cfg, batch, seq, accum, 2, checkpoint=ck))  # noqa: E731
    units, peak = sl.recompute.solve(make, units_of(cfg), budget, cost)
    if units:  # a whole-block choice subsumes that block's halves; drop them and re-verify
        blocks = {u.split(".")[0] for u in units if u.endswith(".block")}
        trimmed = frozenset(u for u in units if u.endswith(".block") or u.split(".")[0] not in blocks)
        if trimmed != units:
            units, peak = trimmed, sl.recompute.peak_for(make, trimmed, budget)
    return units, peak


def gib(n):
    return f"{n / 2**30:.3f} GiB"


def launch_workers(nproc):
    """Re-run this script once per visible GPU with rank env vars. Rank 0 does the printing."""
    init_file = os.path.join(tempfile.mkdtemp(), "nccl_id")
    procs = [subprocess.Popen([sys.executable, *sys.argv],
                              env=dict(os.environ, RANK=str(r), WORLD_SIZE=str(nproc), SOLITON_INIT_FILE=init_file))
             for r in range(nproc)]
    return max(p.wait() for p in procs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="124M", choices=CONFIGS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--data", default=None, help="dir with train.bin (uint16 GPT-2 tokens)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--budget-gib", type=float, help="hard memory cap; checkpoints as few blocks as needed to fit")
    ap.add_argument("--nproc", type=int, default=1, help="data-parallel processes, one per visible GPU")
    args = ap.parse_args()
    if args.nproc > 1 and "RANK" not in os.environ:
        sys.exit(launch_workers(args.nproc))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    say = print if rank == 0 else (lambda *a, **k: None)

    t0 = time.time()
    budget = int(args.budget_gib * 2**30) if args.budget_gib else 0
    ckpt = 0
    if budget:
        ckpt, fit_peak = fit_checkpoints(args.config, args.batch, args.seq, args.accum, budget)
        if ckpt is None:
            sys.exit(f"[plan] does not fit {args.budget_gib} GiB even with everything recomputed")
        say(f"[plan] fits {args.budget_gib} GiB by recomputing {len(ckpt)} of {len(units_of(args.config))} "
            f"units ({fit_peak / 2**30:.3f} GiB), solved in {time.time() - t0:.2f}s with no GPU")
    plan = sl.plan(lambda dev: run(dev, args.config, args.batch, args.seq, args.accum, 2, checkpoint=ckpt), budget)
    params = nn.num_params(GPT(**{**CONFIGS[args.config], "device": "meta"}))
    sl.empty_cache("meta")
    say(f"[plan] GPT-2 {args.config}: {params / 1e6:.1f}M params, batch {args.batch}x{args.seq}, accum {args.accum}"
        + (f", {world} data-parallel ranks (plan is per rank)" if world > 1 else ""))
    say(f"[plan] peak allocated {gib(plan['peak_allocated'])}, peak reserved {gib(plan['peak_reserved'])} "
        f"(dry run took {time.time() - t0:.2f}s, no GPU used)")
    if args.plan_only:
        return
    if world > 1:
        distributed.init(rank, world, os.environ["SOLITON_INIT_FILE"])
    if budget:
        sl.set_memory_limit("cuda", budget)

    data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(rank)  # each rank sees different batches

    def get_batch():
        ix = rng.integers(0, len(data) - args.seq - 1, size=args.batch)
        x = np.stack([data[i:i + args.seq] for i in ix]).astype(np.int32)
        y = np.stack([data[i + 1:i + 1 + args.seq] for i in ix]).astype(np.int32)
        return x, y

    _, total = sl.cuda_mem_info()
    tokens = args.batch * args.seq * args.accum * world

    def log(step, loss, norm, dt):
        say(f"step {step:5d} | loss {loss:.4f} | norm {norm:.3f} | {dt * 1000:.0f} ms | {tokens / dt:,.0f} tok/s", flush=True)

    real, losses, _, _ = run("cuda", args.config, args.batch, args.seq, args.accum, args.steps, get_batch, args.lr, log,
                             checkpoint=ckpt)
    free1, _ = sl.cuda_mem_info()
    say(f"[real] peak allocated {gib(real['peak_allocated'])}, peak reserved {gib(real['peak_reserved'])}")
    for key in ("peak_allocated", "peak_reserved"):
        say(f"[check] {key}: predicted {plan[key]:,} bytes vs actual {real[key]:,} bytes -> "
            f"{'exact' if plan[key] == real[key] else f'off by {real[key] - plan[key]:+,} bytes'}")
    say(f"[driver] process uses {gib(total - free1)} total; context/library overhead outside the pool "
        f"{gib(total - free1 - real['reserved'])}")


if __name__ == "__main__":
    main()
