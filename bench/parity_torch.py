"""Train Soliton and PyTorch GPT-2 124M from identical weights on identical TinyStories batches; compare losses.

    CUDA_VISIBLE_DEVICES=0 python bench/parity_torch.py --steps 50
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "python"), os.path.join(ROOT, "examples"), HERE]
import soliton as sl  # noqa: E402
import torch_gpt2 as tg  # noqa: E402
from gpt2 import CONFIGS, GPT, train_step  # noqa: E402
from soliton.optim import AdamW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--control", action="store_true",
                    help="compare PyTorch with PyTorch whose weights get 1e-6 relative noise, to see how chaotic training is")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = GPT(**{**CONFIGS["124M"], "device": "cuda"})
    opt = AdamW(model.parameters(), lr=6e-4)
    tmodel = tg.GPT().cuda()
    tparams = dict(tmodel.named_parameters())
    with torch.no_grad():
        for name, p in model.named_parameters():
            a = torch.from_numpy(p.numpy())
            if ".qkv." in name:  # Soliton packs q, k and v into one projection; PyTorch keeps three
                c = a.shape[0] // 3
                for i, part in enumerate(("q", "k", "v")):
                    tparams[name.replace("qkv", part)].copy_(a[i * c:(i + 1) * c])
            else:
                tparams[name.replace("ln_f", "lnf")].copy_(a)
    topt = tg.make_opt(tmodel)
    if args.control:
        other = tg.GPT().cuda()
        with torch.no_grad():
            for p, q in zip(tmodel.parameters(), other.parameters()):
                q.copy_(p * (1 + 1e-6 * torch.randn_like(p)))
        oopt = tg.make_opt(other)

    data = np.memmap(os.path.join(ROOT, "data", "tinystories", "train.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    diffs = []
    label = "control" if args.control else "soliton"
    for step in range(args.steps):
        ix = rng.integers(0, len(data) - args.seq - 1, size=args.batch)
        tok = np.stack([data[i:i + args.seq + 1] for i in ix]).astype(np.int32)
        x, y = tok[:, :-1], tok[:, 1:]
        tx, ty = (torch.from_numpy(a.astype(np.int64)).cuda() for a in (x, y))
        if args.control:
            loss = tg.train_step(other, oopt, tx, ty).item()
        else:
            loss, _ = train_step(model, opt, [(sl.tensor(x, "cuda"), sl.tensor(y, "cuda"))])
        tloss = tg.train_step(tmodel, topt, tx, ty).item()
        diffs.append(abs(loss - tloss))
        print(f"step {step:3d} | {label} {loss:.5f} | pytorch {tloss:.5f} | diff {loss - tloss:+.2e}", flush=True)
    print(f"max |loss diff| over {args.steps} steps: {max(diffs):.2e} (first 10 steps: {max(diffs[:10]):.2e})")


if __name__ == "__main__":
    main()
