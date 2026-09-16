import math

from . import tensor as T
from ._C import lib


class AdamW:
    """Fused AdamW. Weight decay applies to params with ndim >= 2 (matrices, embeddings), not biases/norms."""

    def __init__(self, params, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.t = 0
        # State is allocated up front so it shows up in the memory plan before step 1.
        self.state = [(T.zeros(p.shape, p.dev), T.zeros(p.shape, p.dev)) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, (m, v) in zip(self.params, self.state):
            if p.grad is None:
                continue
            wd = self.wd if p.ndim >= 2 else 0.0
            lib.sl_adamw(p.dev, p.ptr, p.grad.ptr, m.ptr, v.ptr, p.numel, self.lr, b1, b2, self.eps, wd, self.t)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def clip_grad_norm(params, max_norm):
    """Scales grads in place so their global L2 norm is at most max_norm. Returns the pre-clip norm."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = math.sqrt(sum(T.sumsq(g) for g in grads))
    if norm > max_norm:
        s = max_norm / (norm + 1e-6)
        for g in grads:
            lib.sl_scale(g.dev, g.ptr, g.ptr, g.numel, s)
    return norm
