"""Soliton: a deep learning framework that tells you how much memory training needs before you run it."""
from . import arena, distributed, nn, optim, preflight, recompute
from .tensor import (
    Tensor, add, attention, checkpoint, clone, cross_entropy, cuda_mem_info, div, embedding, empty, empty_cache, full, gelu,
    layernorm, linear, matmul, memory_stats, mul, no_grad, normal, permute, plan, reset_peak_stats, reshape,
    scale, set_device, set_fusion, set_memory_limit, set_tf32, softmax, sub, sum_, sumsq, synchronize, tensor,
    transpose, zeros,
)

__version__ = "0.0.1"
