import math

from . import tensor as T


class Module:
    def named_parameters(self, prefix=""):
        for name, v in vars(self).items():
            if isinstance(v, T.Tensor) and v.requires_grad:
                yield prefix + name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(f"{prefix}{name}.")
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, device="cpu", std=0.02, rng=None):
        self.weight = T.normal((n_out, n_in), std, device, rng).requires_grad_()
        self.bias = T.zeros((n_out,), device).requires_grad_() if bias else None

    def forward(self, x):
        return T.linear(x, self.weight, self.bias)


class LayerNorm(Module):
    def __init__(self, dim, device="cpu", eps=1e-5):
        self.weight = T.full((dim,), 1.0, device).requires_grad_()
        self.bias = T.zeros((dim,), device).requires_grad_()
        self.eps = eps

    def forward(self, x):
        return T.layernorm(x, self.weight, self.bias, self.eps)


class Embedding(Module):
    def __init__(self, num, dim, device="cpu", std=0.02, rng=None):
        self.weight = T.normal((num, dim), std, device, rng).requires_grad_()

    def forward(self, idx):
        return T.embedding(self.weight, idx)


def num_params(module):
    return sum(math.prod(p.shape) for p in module.parameters())
