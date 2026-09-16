"""GPT-2 training step in PyTorch eager, same math as examples/gpt2.py.

--predict estimates peak memory before running, with PyTorch's own MemTracker under FakeTensorMode.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

V, CTX, L, H, C = 50257, 1024, 12, 12, 768


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(C, eps=1e-5), nn.LayerNorm(C, eps=1e-5)
        self.q, self.k, self.v, self.proj = (nn.Linear(C, C) for _ in range(4))
        self.fc, self.fc_proj = nn.Linear(C, 4 * C), nn.Linear(4 * C, C)

    def forward(self, x, mask):
        b, t, _ = x.shape
        hs = C // H
        y = self.ln1(x)
        q, k, v = (lin(y).view(b, t, H, hs).transpose(1, 2) for lin in (self.q, self.k, self.v))
        att = torch.softmax((q @ k.transpose(2, 3)) / math.sqrt(hs) + mask, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, C)
        x = x + self.proj(y)
        return x + self.fc_proj(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, C), nn.Embedding(CTX, C)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.lnf = nn.LayerNorm(C, eps=1e-5)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * L) if name.endswith("proj.weight") else 0.02)
            elif name.endswith("bias") and "ln" not in name:
                nn.init.zeros_(p)

    def forward(self, idx, tgt):
        _, t = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        logits = F.linear(self.lnf(x), self.wte.weight)
        return F.cross_entropy(logits.view(-1, V), tgt.reshape(-1))


def make_opt(model):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                             lr=6e-4, betas=(0.9, 0.95), eps=1e-8)


def train_step(model, opt, idx, tgt):
    loss = model(idx, tgt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


def predict(batch, seq):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.distributed._tools.mem_tracker import MemTracker

    t0 = time.time()
    with FakeTensorMode():
        with torch.device("cuda"):
            model, idx, tgt = GPT(), torch.randint(0, V, (batch, seq)), torch.randint(0, V, (batch, seq))
            opt = make_opt(model)
        tracker = MemTracker()
        tracker.track_external(model, opt)
        with tracker:
            for _ in range(2):  # step 2 includes optimizer state
                train_step(model, opt, idx, tgt)
                tracker.reset_mod_stats()  # required between iterations; global peak is kept
        snap = tracker.get_tracker_snapshot("peak")
    peak = max(v["Total"] for d, v in snap.items() if torch.device(d).type == "cuda")
    return dict(framework="pytorch", batch=batch, seq=seq, predicted=peak, predict_s=time.time() - t0)


def run(batch, seq, steps, budget_gib=None, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.manual_seed(0)
    out = dict(framework="pytorch", batch=batch, seq=seq, oom=False)
    if budget_gib:  # hard cap on the caching allocator's reserved memory
        torch.cuda.set_per_process_memory_fraction(budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    try:
        model = GPT().cuda()
        opt = make_opt(model)
        rng = np.random.default_rng(0)
        losses, times = [], []
        for _ in range(steps):
            tok = torch.from_numpy(rng.integers(0, V, size=(batch, seq + 1)).astype(np.int64)).cuda()
            t0 = time.time()
            losses.append(train_step(model, opt, tok[:, :-1], tok[:, 1:]).item())
            times.append(time.time() - t0)
        warm = times[2:] or times
        out.update(loss_first=losses[0], loss_last=losses[-1], first_step_s=times[0],
                   step_ms=1000 * sum(warm) / len(warm), tok_s=batch * seq * len(warm) / sum(warm),
                   native_peak=torch.cuda.max_memory_reserved(), native_peak_alloc=torch.cuda.max_memory_allocated())
    except torch.OutOfMemoryError as e:
        out.update(oom=True, error=str(e).splitlines()[0][:200])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor cores for matmul (lower precision)")
    args = ap.parse_args()
    res = predict(args.batch, args.seq) if args.predict else run(args.batch, args.seq, args.steps, args.budget_gib,
                                                                 args.tf32)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
