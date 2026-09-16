"""ctypes bindings to libsoliton.so (D core + CUDA kernels)."""
import ctypes as C
import os

lib = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsoliton.so"))

P, L, I, F, D, S = C.c_void_p, C.c_int64, C.c_int, C.c_float, C.c_double, C.c_size_t
LP = C.POINTER(C.c_int64)

# ponytail: ctypes per-op overhead is a few µs; move to a CPython extension when profiles show it.
_SIGS = {
    "sl_alloc": (P, [I, S]),
    "sl_free": (None, [I, P, S]),
    "sl_stats": (None, [I, LP]),
    "sl_reset_stats": (None, [I]),
    "sl_empty_cache": (None, [I]),
    "sl_set_limit": (None, [I, L]),
    "sl_init_device": (I, [I, I]),
    "sl_cuda_count": (I, []),
    "sl_check": (I, [I]),
    "sl_sync": (None, [I]),
    "sl_mem_info": (None, [C.POINTER(S), C.POINTER(S)]),
    "sl_set_tf32": (None, [I, I]),
    "sl_from_host": (None, [I, P, P, S]),
    "sl_to_host": (None, [I, P, P, S]),
    "sl_copy": (None, [I, P, P, S]),
    "sl_fill": (None, [I, P, L, F]),
    "sl_binary": (None, [I, I, P, P, P, I, LP, LP, LP]),
    "sl_strided_copy": (None, [I, P, L, P, I, LP, LP]),
    "sl_reduce_sum": (I, [I, P, P, L, I, LP, LP]),
    "sl_axpy": (None, [I, P, P, L, F]),
    "sl_scale": (None, [I, P, P, L, F]),
    "sl_add_scalar": (None, [I, P, L, F]),
    "sl_sumsq": (D, [I, P, L]),
    "sl_matmul": (None, [I, P, P, P, L, L, L, L, I, I]),
    "sl_softmax": (I, [I, P, P, L, L]),
    "sl_softmax_bwd": (I, [I, P, P, P, L, L]),
    "sl_gelu": (None, [I, P, P, L]),
    "sl_gelu_bwd": (None, [I, P, P, P, L]),
    "sl_layernorm": (I, [I, P, P, P, P, P, P, L, L, F]),
    "sl_layernorm_bwd": (I, [I, P, P, P, P, P, P, P, P, L, L]),
    "sl_embedding": (None, [I, P, P, P, L, L]),
    "sl_embedding_bwd": (I, [I, P, P, P, L, L, L]),
    "sl_cross_entropy": (D, [I, P, P, L, L, C.POINTER(I)]),
    "sl_cross_entropy_bwd": (I, [I, P, P, P, L, L, F]),
    "sl_adamw": (None, [I, P, P, P, P, L, F, F, F, F, F, I]),
    "sl_attention": (I, [I, P, P, P, L, L, L, L, F]),
    "sl_attention_bwd": (I, [I, P, P, P, P, P, L, L, L, L, F]),
    "sl_jit": (I, [I, C.c_char_p, C.c_char_p, C.POINTER(P)]),
    "sl_jit_launch": (None, [I, P, L, C.POINTER(P)]),
    "sl_jit_free": (None, [I, P]),
    "sl_jit_log": (C.c_char_p, []),
    "sl_dist_unique_id": (I, [C.c_char_p]),
    "sl_dist_init": (I, [I, I, C.c_char_p]),
    "sl_dist_group": (I, [I, I]),
    "sl_dist_allreduce_mean": (I, [I, P, L]),
    "sl_dist_destroy": (None, []),
}
for _name, (_res, _args) in _SIGS.items():
    _fn = getattr(lib, _name)
    _fn.restype, _fn.argtypes = _res, _args


def longs(xs):
    xs = list(xs)
    return (C.c_int64 * max(len(xs), 1))(*xs)
