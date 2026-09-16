"""Every op's forward and backward checked against PyTorch on CPU, on each available Soliton device."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import soliton as sl
from soliton._C import lib

DEVICES = ["cpu"] + (["cuda"] if lib.sl_cuda_count() > 0 else [])


def check(device, fn_sl, fn_t, shapes, positive=(), atol=1e-4):
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    for i in positive:
        xs[i] = np.abs(xs[i]) + 0.5
    sx = [sl.tensor(x, device).requires_grad_() for x in xs]
    tx = [torch.tensor(x, requires_grad=True) for x in xs]
    so, to = fn_sl(*sx), fn_t(*tx)
    np.testing.assert_allclose(so.numpy(), to.detach().numpy(), rtol=1e-4, atol=atol)
    gy = np.asarray(rng.standard_normal(to.shape), dtype=np.float32)
    so.backward(sl.tensor(gy, device))
    to.backward(torch.tensor(gy))
    for s, t in zip(sx, tx):
        np.testing.assert_allclose(s.grad.numpy(), t.grad.numpy(), rtol=1e-3, atol=atol)


CASES = {
    "add_broadcast": (lambda a, b: a + b, lambda a, b: a + b, [(2, 3, 4), (4,)]),
    "sub_broadcast": (lambda a, b: a - b, lambda a, b: a - b, [(2, 3, 4), (3, 1)]),
    "mul_broadcast": (lambda a, b: a * b, lambda a, b: a * b, [(2, 1, 4), (3, 4)]),
    "div": (lambda a, b: a / b, lambda a, b: a / b, [(3, 4), (4,)], (1,)),
    "scalar_math": (lambda a: (a * 2.0 - 1.0) / 3.0 + a, lambda a: (a * 2.0 - 1.0) / 3.0 + a, [(5,)]),
    "same_input_twice": (lambda a: a * a + a, lambda a: a * a + a, [(3, 3)]),
    "matmul_batched": (sl.matmul, torch.matmul, [(2, 3, 4, 5), (2, 3, 5, 6)]),
    "linear": (sl.linear, F.linear, [(2, 3, 5), (4, 5), (4,)]),
    "reshape_permute": (lambda x: sl.permute(sl.reshape(x, (2, 3, 4)), (2, 0, 1)),
                        lambda x: x.reshape(2, 3, 4).permute(2, 0, 1), [(6, 4)]),
    "transpose": (lambda x: sl.transpose(x, 1, 3), lambda x: x.transpose(1, 3), [(2, 3, 4, 5)]),
    "softmax": (sl.softmax, lambda x: torch.softmax(x, -1), [(3, 4, 7)]),
    "gelu": (sl.gelu, F.gelu, [(4, 9)]),
    "layernorm": (sl.layernorm, lambda x, w, b: F.layer_norm(x, (8,), w, b, 1e-5), [(2, 3, 8), (8,), (8,)]),
    "sum_mean": (lambda x: x.sum() + x.mean(), lambda x: x.sum() + x.mean(), [(3, 5)]),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", CASES)
def test_op(device, name):
    fn_sl, fn_t, shapes, *pos = CASES[name]
    check(device, fn_sl, fn_t, shapes, *pos)


@pytest.mark.parametrize("device", DEVICES)
def test_causal_attention(device):
    t = 6
    mask = np.triu(np.full((t, t), -np.inf, np.float32), 1)

    def f_sl(q, k, v):
        return sl.matmul(sl.softmax(sl.scale(sl.matmul(q, sl.transpose(k, 2, 3)), 0.5) + sl.tensor(mask, device)), v)

    def f_t(q, k, v):
        return torch.softmax((q @ k.transpose(2, 3)) * 0.5 + torch.tensor(mask), -1) @ v

    check(device, f_sl, f_t, [(2, 3, t, 4)] * 3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("t", [6, 256, 512])  # 256 and 512 exercise the multi-tile path
def test_fused_attention(device, t):
    b, h, hs = 2, 3, 16
    rng = np.random.default_rng(4)
    packed = rng.standard_normal((b, t, 3, h, hs)).astype(np.float32)
    gy = rng.standard_normal((b, t, h, hs)).astype(np.float32)
    sq = sl.tensor(packed, device).requires_grad_()
    out = sl.attention(sq)
    out.backward(sl.tensor(gy, device))

    tq = torch.tensor(packed, requires_grad=True)
    q, k, v = (tq[:, :, i].transpose(1, 2) for i in range(3))  # (b,h,t,hs)
    tout = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    tout.backward(torch.tensor(gy))
    np.testing.assert_allclose(out.numpy(), tout.detach().numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sq.grad.numpy(), tq.grad.numpy(), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n,vocab", [(7, 5), (4096, 37)])  # the large case stresses duplicate indices
def test_embedding(device, n, vocab):
    rng = np.random.default_rng(1)
    w = rng.standard_normal((vocab, 16)).astype(np.float32)
    idx = rng.integers(0, vocab, size=(n,)).astype(np.int32)
    gy = rng.standard_normal((n, 16)).astype(np.float32)
    sw = sl.tensor(w, device).requires_grad_()
    out = sl.embedding(sw, sl.tensor(idx, device))
    tw = torch.tensor(w, requires_grad=True)
    tout = F.embedding(torch.tensor(idx, dtype=torch.long), tw)
    np.testing.assert_allclose(out.numpy(), tout.detach().numpy())
    out.backward(sl.tensor(gy, device))
    tout.backward(torch.tensor(gy))
    np.testing.assert_allclose(sw.grad.numpy(), tw.grad.numpy(), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n,vocab", [(6, 11), (512, 4099)])
def test_cross_entropy(device, n, vocab):
    rng = np.random.default_rng(2)
    logits = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    tgt = rng.integers(0, vocab, size=(n,)).astype(np.int32)
    sx = sl.tensor(logits, device).requires_grad_()
    loss = sl.cross_entropy(sx, sl.tensor(tgt, device))
    tx = torch.tensor(logits, requires_grad=True)
    tloss = F.cross_entropy(tx, torch.tensor(tgt, dtype=torch.long))
    assert abs(loss.item() - tloss.item()) < 1e-4
    loss.backward()
    tloss.backward()
    np.testing.assert_allclose(sx.grad.numpy(), tx.grad.numpy(), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_adamw_matches_torch(device):
    rng = np.random.default_rng(3)
    p0 = rng.standard_normal((4, 5)).astype(np.float32)
    sp = sl.tensor(p0, device).requires_grad_()
    tp = torch.nn.Parameter(torch.tensor(p0))
    opt = sl.optim.AdamW([sp], lr=1e-2, weight_decay=0.1)
    topt = torch.optim.AdamW([tp], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(5):
        g = rng.standard_normal(p0.shape).astype(np.float32)
        sp.grad, tp.grad = sl.tensor(g, device), torch.tensor(g)
        opt.step()
        topt.step()
    np.testing.assert_allclose(sp.numpy(), tp.detach().numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_accumulates_across_backward_calls(device):
    x = sl.tensor(np.ones((3,), np.float32), device).requires_grad_()
    for _ in range(2):
        (x * 3.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((3,), 6.0))
