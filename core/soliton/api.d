/// The C ABI Python talks to. Every op takes a device id and routes to CPU, CUDA, or nothing (meta).
///
/// Rule that makes memory prediction exact: any scratch memory an op needs is taken from the pool here, in the
/// same order on every device, so meta replays the same allocation trace as the real device.
module soliton.api;

import soliton.alloc;
static import cpu = soliton.cpu;

extern (C) nothrow @nogc
{
    int sl_cuda_set_device(int ordinal);
    int sl_cuda_device_count();
    int sl_cuda_check();
    void sl_cuda_sync();
    void sl_cuda_mem_info(size_t* free, size_t* total);
    void sl_cuda_set_workspace(void* p, size_t n);
    void sl_cuda_set_tf32(int on);
    void sl_cuda_h2d(void* dst, const(void)* src, size_t n);
    void sl_cuda_d2h(void* dst, const(void)* src, size_t n);
    void sl_cuda_d2d(void* dst, const(void)* src, size_t n);
    void sl_cuda_fill(float* p, long n, float v);
    void sl_cuda_binary(int op, const(float)* a, const(float)* b, float* o, int nd, const(long)* shape,
        const(long)* sa, const(long)* sb);
    void sl_cuda_strided_copy(const(float)* src, long base, float* o, int nd, const(long)* shape, const(long)* ss);
    void sl_cuda_reduce_sum(const(float)* src, float* dst, long ndst, int nd, const(long)* shape, const(long)* sd,
        float* ones);
    void sl_cuda_axpy(const(float)* x, float* y, long n, float alpha);
    void sl_cuda_scale(const(float)* x, float* y, long n, float alpha);
    void sl_cuda_add_scalar(float* p, long n, float c);
    double sl_cuda_sumsq(const(float)* x, long n);
    void sl_cuda_matmul(const(float)* a, const(float)* b, float* o, long bt, long n, long k, long m, int ta, int tb);
    void sl_cuda_softmax(const(float)* x, float* y, long outer, long dim, long chunks, float* cmax, float* rmax,
        float* rsum, float* ones);
    void sl_cuda_softmax_bwd(const(float)* y, const(float)* dy, float* dx, long outer, long dim, float* dot, float* ones);
    void sl_cuda_gelu(const(float)* x, float* y, long n);
    void sl_cuda_gelu_bwd(const(float)* x, const(float)* dy, float* dx, long n);
    void sl_cuda_layernorm(const(float)* x, const(float)* w, const(float)* b, float* y, float* mean,
        float* invstd, long outer, long c, float eps, float* ones);
    void sl_cuda_layernorm_bwd(const(float)* dy, const(float)* x, const(float)* w, const(float)* mean,
        const(float)* invstd, float* dx, float* dw, float* db, long outer, long c, float* tmp, float* sg,
        float* sgx, float* onesC, float* onesRows);
    void sl_cuda_embedding(const(float)* w, const(int)* idx, float* o, long n, long c);
    void sl_cuda_embedding_bwd(const(float)* dy, const(int)* idx, float* dw, long vocab, long n, long c,
        int* keys, int* perm, int* permOut, void* temp, size_t tempBytes);
    double sl_cuda_cross_entropy(const(float)* logits, const(int)* t, long n, long v, float* rowloss);
    void sl_cuda_cross_entropy_bwd(const(float)* logits, const(int)* t, float* dl, long n, long v, float dloss);
    void sl_cuda_adamw(float* p, const(float)* g, float* m, float* v, long n, float lr, float b1, float b2,
        float eps, float wd, int step);
    long sl_cuda_attn_tile(long T);
    void sl_cuda_attention(const(float)* qkv, float* o, float* L, long B, long T, long H, long hs, float scale,
        long tile, float* s, float* acc, float* m, float* l, float* c, float* rowbuf, float* ones, void* ptrs);
    void sl_cuda_attention_bwd(const(float)* qkv, const(float)* o, const(float)* L, const(float)* dout, float* dqkv,
        long B, long T, long H, long hs, float scale, long tile, float* s, float* dp, float* doc, float* oc,
        float* lrow, float* D, float* rowbuf, float* ones, void* ptrs);
    int sl_cuda_jit(const(char)* src, const(char)* name, void** o);
    void sl_cuda_jit_launch(void* handle, long n, void** args);
    void sl_cuda_jit_free(void* handle);
    const(char)* sl_cuda_jit_log();
    int sl_cuda_nccl_unique_id(char* o);
    int sl_cuda_nccl_init(int rank, int world, const(char)* id);
    int sl_cuda_nccl_group(int start);
    int sl_cuda_nccl_allreduce_mean(float* p, long n);
    void sl_cuda_nccl_destroy();
}

nothrow @nogc:

private void route(alias cpuFn, alias cudaFn, Args...)(int dev, Args args)
{
    if (dev == DEV_CPU)
        cpuFn(args);
    else if (dev == DEV_CUDA)
        cudaFn(args);
}

/// Scratch buffers for one op, released in reverse order when the op returns.
private struct Scratch
{
    Pool* pool;
    void*[16] ptr;
    size_t[16] size;
    int count;
    bool ok = true;

nothrow @nogc:

    void* bytes(size_t n)
    {
        auto p = pool.alloc(n);
        ok = ok && p !is null;
        ptr[count] = p;
        size[count++] = n;
        return p;
    }

    float* floats(long n)
    {
        return cast(float*) bytes(cast(size_t) n * float.sizeof);
    }

    ~this()
    {
        foreach_reverse (i; 0 .. count)
            if (ptr[i])
                pool.release(ptr[i], size[i]);
    }
}

enum OOM = -2;

/// Chunk count for row maxima: loops of ~sqrt(dim) keep GPU threads short.
private long chunksFor(long dim)
{
    long c = 1;
    while (c * c < dim)
        c++;
    return c;
}

/// Scratch for CUB's radix sort. CUB's own size query returns 0 when no GPU is visible, which made plans made
/// without a GPU differ from the real run; this bound depends only on n, and the CUDA kernel checks CUB's
/// real need against it.
private size_t sortTempBytes(long n)
{
    return cast(size_t) n * 16 + (1 << 20);
}

enum CUBLAS_WORKSPACE = 32 << 20;

// ---- memory ----

export extern (C) void* sl_alloc(int dev, size_t n)
{
    return pools[dev].alloc(n);
}

export extern (C) void sl_free(int dev, void* p, size_t n)
{
    pools[dev].release(p, n);
}

/// out: allocated, reserved, peak_allocated, peak_reserved, n_alloc, n_raw, trace_hash
export extern (C) void sl_stats(int dev, long* o)
{
    auto p = &pools[dev];
    o[0] = p.allocated;
    o[1] = p.reserved;
    o[2] = p.peakAllocated;
    o[3] = p.peakReserved;
    o[4] = p.nAlloc;
    o[5] = p.nRaw;
    o[6] = cast(long) p.traceHash;
}

export extern (C) void sl_reset_stats(int dev)
{
    auto p = &pools[dev];
    p.peakAllocated = p.allocated;
    p.peakReserved = p.reserved;
    p.nAlloc = p.nRaw = 0;
    p.traceHash = Pool.init.traceHash;
}

export extern (C) void sl_empty_cache(int dev)
{
    pools[dev].emptyCache();
}

// ---- static memory planning: record a trace, then replay it against one arena ----

export extern (C) void sl_record_start(int dev)
{
    auto p = &pools[dev];
    p.recCount = p.nmarks = 0;
    p.recording = true;
}

export extern (C) void sl_record_stop(int dev)
{
    pools[dev].recording = false;
}

/// Note a step boundary while recording; the planner uses the last one as the start of the repeating body.
export extern (C) void sl_record_mark(int dev)
{
    auto p = &pools[dev];
    if (p.recording && p.nmarks < p.marks.length)
        p.marks[p.nmarks++] = cast(long) p.recCount;
    else if (p.arenaMode)
        p.planCursor = p.planBody; // replay: the next step reuses the same offsets
}

export extern (C) long sl_record_count(int dev)
{
    return cast(long) pools[dev].recCount;
}

export extern (C) long sl_record_body(int dev)
{
    auto p = &pools[dev];
    return p.nmarks ? p.marks[p.nmarks - 1] : 0;
}

export extern (C) void sl_record_get(int dev, long* size, long* start, long* end)
{
    auto p = &pools[dev];
    foreach (i; 0 .. p.recCount)
    {
        size[i] = p.recSize[i];
        start[i] = p.recStart[i];
        end[i] = p.recEnd[i];
    }
}

/// Install a plan: one arena allocation, then fixed offsets in the recorded order.
export extern (C) int sl_arena_install(int dev, const(long)* offsets, const(long)* sizes, long count,
    long body_, long arenaBytes)
{
    auto p = &pools[dev];
    p.emptyCache();
    void* a = p.rawAlloc(cast(size_t) arenaBytes);
    if (!a)
        return OOM;
    p.arena = a;
    p.arenaBytes = arenaBytes;
    p.planOffset = offsets;
    p.planSize = sizes;
    p.planCount = cast(size_t) count;
    p.planBody = cast(size_t) body_;
    p.planCursor = 0;
    p.arenaMode = true;
    p.diverged = false;
    p.reserved += arenaBytes;
    if (p.reserved > p.peakReserved)
        p.peakReserved = p.reserved;
    return 0;
}

/// 0 while the run matches its plan; 1 once an allocation asked for a size the plan did not predict.
export extern (C) int sl_arena_diverged(int dev)
{
    return pools[dev].diverged ? 1 : 0;
}

export extern (C) void sl_arena_off(int dev)
{
    auto p = &pools[dev];
    if (!p.arenaMode)
        return;
    p.arenaMode = false;  // stop handing out offsets; new allocations go back to the pool
    if (p.allocated == 0)
        p.releaseArena();  // otherwise the last arena-backed release frees it
}

export extern (C) void sl_set_limit(int dev, long bytes)
{
    pools[dev].limit = bytes;
}

/// Meta pretends to be CUDA: both reserve the cuBLAS workspace from the pool.
export extern (C) int sl_init_device(int dev, int ordinal)
{
    if (dev == DEV_CPU)
        return 0;
    if (dev == DEV_CUDA && sl_cuda_set_device(ordinal) != 0)
        return -1;
    void* ws = pools[dev].alloc(CUBLAS_WORKSPACE);
    if (!ws)
        return OOM;
    if (dev == DEV_CUDA)
        sl_cuda_set_workspace(ws, CUBLAS_WORKSPACE);
    return 0;
}

export extern (C) int sl_cuda_count()
{
    return sl_cuda_device_count();
}

export extern (C) int sl_check(int dev)
{
    return dev == DEV_CUDA ? sl_cuda_check() : 0;
}

export extern (C) void sl_sync(int dev)
{
    if (dev == DEV_CUDA)
        sl_cuda_sync();
}

/// Opt-in: trade gemm precision for speed. Off by default, so results stay bit-comparable with fp32.
export extern (C) void sl_set_tf32(int dev, int on)
{
    if (dev == DEV_CUDA)
        sl_cuda_set_tf32(on);
}

export extern (C) void sl_mem_info(size_t* free, size_t* total)
{
    sl_cuda_mem_info(free, total);
}

// ---- data movement ----

private void cpuMemcpy(void* dst, const(void)* src, size_t n)
{
    import core.stdc.string : memcpy;

    memcpy(dst, src, n);
}

export extern (C) void sl_from_host(int dev, void* dst, const(void)* src, size_t n)
{
    route!(cpuMemcpy, sl_cuda_h2d)(dev, dst, src, n);
}

export extern (C) void sl_to_host(int dev, void* dst, const(void)* src, size_t n)
{
    route!(cpuMemcpy, sl_cuda_d2h)(dev, dst, src, n);
}

export extern (C) void sl_copy(int dev, void* dst, const(void)* src, size_t n)
{
    route!(cpuMemcpy, sl_cuda_d2d)(dev, dst, src, n);
}

// ---- ops ----

export extern (C) void sl_fill(int dev, float* p, long n, float v)
{
    route!(cpu.fill, sl_cuda_fill)(dev, p, n, v);
}

export extern (C) void sl_binary(int dev, int op, const(float)* a, const(float)* b, float* o, int nd,
    const(long)* shape, const(long)* sa, const(long)* sb)
{
    route!(cpu.binary, sl_cuda_binary)(dev, op, a, b, o, nd, shape, sa, sb);
}

export extern (C) void sl_strided_copy(int dev, const(float)* src, long base, float* o, int nd,
    const(long)* shape, const(long)* ss)
{
    route!(cpu.stridedCopy, sl_cuda_strided_copy)(dev, src, base, o, nd, shape, ss);
}

export extern (C) int sl_reduce_sum(int dev, const(float)* src, float* dst, long ndst, int nd,
    const(long)* shape, const(long)* sd)
{
    long n = 1;
    foreach (d; 0 .. nd)
        n *= shape[d];
    auto s = Scratch(&pools[dev]);
    auto ones = s.floats(n / (ndst > 0 ? ndst : 1));
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.reduceSum(src, dst, ndst, nd, shape, sd);
    else if (dev == DEV_CUDA)
        sl_cuda_reduce_sum(src, dst, ndst, nd, shape, sd, ones);
    return 0;
}

export extern (C) void sl_axpy(int dev, const(float)* x, float* y, long n, float alpha)
{
    route!(cpu.axpy, sl_cuda_axpy)(dev, x, y, n, alpha);
}

export extern (C) void sl_scale(int dev, const(float)* x, float* y, long n, float alpha)
{
    route!(cpu.scale, sl_cuda_scale)(dev, x, y, n, alpha);
}

export extern (C) void sl_add_scalar(int dev, float* p, long n, float c)
{
    route!(cpu.addScalar, sl_cuda_add_scalar)(dev, p, n, c);
}

export extern (C) double sl_sumsq(int dev, const(float)* x, long n)
{
    return dev == DEV_CPU ? cpu.sumsq(x, n) : dev == DEV_CUDA ? sl_cuda_sumsq(x, n) : double.nan;
}

export extern (C) void sl_matmul(int dev, const(float)* a, const(float)* b, float* o, long bt, long n,
    long k, long m, int ta, int tb)
{
    route!(cpu.matmul, sl_cuda_matmul)(dev, a, b, o, bt, n, k, m, ta, tb);
}

export extern (C) int sl_softmax(int dev, const(float)* x, float* y, long outer, long dim)
{
    const chunks = chunksFor(dim);
    auto s = Scratch(&pools[dev]);
    auto cmax = s.floats(outer * chunks), rmax = s.floats(outer), rsum = s.floats(outer), ones = s.floats(dim);
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.softmax(x, y, outer, dim);
    else if (dev == DEV_CUDA)
        sl_cuda_softmax(x, y, outer, dim, chunks, cmax, rmax, rsum, ones);
    return 0;
}

export extern (C) int sl_softmax_bwd(int dev, const(float)* y, const(float)* dy, float* dx, long outer, long dim)
{
    auto s = Scratch(&pools[dev]);
    auto dot = s.floats(outer), ones = s.floats(dim);
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.softmaxBwd(y, dy, dx, outer, dim);
    else if (dev == DEV_CUDA)
        sl_cuda_softmax_bwd(y, dy, dx, outer, dim, dot, ones);
    return 0;
}

export extern (C) void sl_gelu(int dev, const(float)* x, float* y, long n)
{
    route!(cpu.gelu, sl_cuda_gelu)(dev, x, y, n);
}

export extern (C) void sl_gelu_bwd(int dev, const(float)* x, const(float)* dy, float* dx, long n)
{
    route!(cpu.geluBwd, sl_cuda_gelu_bwd)(dev, x, dy, dx, n);
}

export extern (C) int sl_layernorm(int dev, const(float)* x, const(float)* w, const(float)* b, float* y,
    float* mean, float* invstd, long outer, long c, float eps)
{
    auto s = Scratch(&pools[dev]);
    auto ones = s.floats(c);
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.layernorm(x, w, b, y, mean, invstd, outer, c, eps);
    else if (dev == DEV_CUDA)
        sl_cuda_layernorm(x, w, b, y, mean, invstd, outer, c, eps, ones);
    return 0;
}

export extern (C) int sl_layernorm_bwd(int dev, const(float)* dy, const(float)* x, const(float)* w,
    const(float)* mean, const(float)* invstd, float* dx, float* dw, float* db, long outer, long c)
{
    auto s = Scratch(&pools[dev]);
    auto tmp = s.floats(outer * c), sg = s.floats(outer), sgx = s.floats(outer);
    auto onesC = s.floats(c), onesRows = s.floats(outer);
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.layernormBwd(dy, x, w, mean, invstd, dx, dw, db, outer, c);
    else if (dev == DEV_CUDA)
        sl_cuda_layernorm_bwd(dy, x, w, mean, invstd, dx, dw, db, outer, c, tmp, sg, sgx, onesC, onesRows);
    return 0;
}

export extern (C) void sl_embedding(int dev, const(float)* w, const(int)* idx, float* o, long n, long c)
{
    route!(cpu.embedding, sl_cuda_embedding)(dev, w, idx, o, n, c);
}

export extern (C) int sl_embedding_bwd(int dev, const(float)* dy, const(int)* idx, float* dw, long vocab, long n, long c)
{
    // CUDA sorts indices so each vocab row is summed by exactly one thread.
    const tb = sortTempBytes(n);
    auto s = Scratch(&pools[dev]);
    auto keys = cast(int*) s.bytes(n * int.sizeof), perm = cast(int*) s.bytes(n * int.sizeof);
    auto permOut = cast(int*) s.bytes(n * int.sizeof);
    void* temp = s.bytes(tb);
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.embeddingBwd(dy, idx, dw, vocab, n, c);
    else if (dev == DEV_CUDA)
        sl_cuda_embedding_bwd(dy, idx, dw, vocab, n, c, keys, perm, permOut, temp, tb);
    return 0;
}

/// Mean loss over n rows (NaN on meta). Sets *err to OOM if scratch allocation failed.
export extern (C) double sl_cross_entropy(int dev, const(float)* logits, const(int)* t, long n, long v, int* err)
{
    auto s = Scratch(&pools[dev]);
    auto rowloss = s.floats(n);
    *err = s.ok ? 0 : OOM;
    if (!s.ok)
        return double.nan;
    return dev == DEV_CPU ? cpu.crossEntropy(logits, t, n, v)
        : dev == DEV_CUDA ? sl_cuda_cross_entropy(logits, t, n, v, rowloss) : double.nan;
}

export extern (C) int sl_cross_entropy_bwd(int dev, const(float)* logits, const(int)* t, float* dl, long n,
    long v, float dloss)
{
    // The fused kernel needs no scratch: one block per row keeps its running max and sum in shared memory.
    route!(cpu.crossEntropyBwd, sl_cuda_cross_entropy_bwd)(dev, logits, t, dl, n, v, dloss);
    return 0;
}

/// Fused causal attention over packed qkv (B,T,3,H,hs) -> out (B,T,H,hs), saving logsumexp L (B,H,T).
export extern (C) int sl_attention(int dev, const(float)* qkv, float* o, float* L, long B, long T, long H,
    long hs, float scale)
{
    const tile = sl_cuda_attn_tile(T);
    const long bh = B * H, rows = bh * tile;
    auto s = Scratch(&pools[dev]);
    auto sc = s.floats(rows * tile), acc = s.floats(rows * hs);
    auto m = s.floats(rows), l = s.floats(rows), c = s.floats(rows);
    auto rowbuf = s.floats(rows * (chunksFor(tile) + 1)), ones = s.floats(tile > hs ? tile : hs);
    auto ptrs = s.bytes(cast(size_t)(5 * bh * (void*).sizeof));
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.attention(qkv, o, L, B, T, H, hs, scale);
    else if (dev == DEV_CUDA)
        sl_cuda_attention(qkv, o, L, B, T, H, hs, scale, tile, sc, acc, m, l, c, rowbuf, ones, ptrs);
    return 0;
}

export extern (C) int sl_attention_bwd(int dev, const(float)* qkv, const(float)* o, const(float)* L,
    const(float)* dout, float* dqkv, long B, long T, long H, long hs, float scale)
{
    const tile = sl_cuda_attn_tile(T);
    const long bh = B * H, rows = bh * tile;
    auto s = Scratch(&pools[dev]);
    auto sc = s.floats(rows * tile), dp = s.floats(rows * tile);
    auto doc = s.floats(rows * hs), oc = s.floats(rows * hs);
    auto lrow = s.floats(rows), dd = s.floats(rows), rowbuf = s.floats(rows * (chunksFor(tile) + 1));
    auto ones = s.floats(tile > hs ? tile : hs);
    auto ptrs = s.bytes(cast(size_t)(9 * bh * (void*).sizeof));
    if (!s.ok)
        return OOM;
    if (dev == DEV_CPU)
        cpu.attentionBwd(qkv, o, L, dout, dqkv, B, T, H, hs, scale);
    else if (dev == DEV_CUDA)
        sl_cuda_attention_bwd(qkv, o, L, dout, dqkv, B, T, H, hs, scale, tile, sc, dp, doc, oc, lrow, dd, rowbuf,
            ones, ptrs);
    return 0;
}

export extern (C) void sl_adamw(int dev, float* p, const(float)* g, float* m, float* v, long n, float lr,
    float b1, float b2, float eps, float wd, int step)
{
    route!(cpu.adamw, sl_cuda_adamw)(dev, p, g, m, v, n, lr, b1, b2, eps, wd, step);
}

// ---- runtime-compiled fused kernels ----

/// Compile generated CUDA source. Returns 0 and a handle, or a negative code (see sl_jit_log for why).
export extern (C) int sl_jit(int dev, const(char)* src, const(char)* name, void** handle)
{
    return dev == DEV_CUDA ? sl_cuda_jit(src, name, handle) : 0;
}

export extern (C) void sl_jit_launch(int dev, void* handle, long n, void** args)
{
    if (dev == DEV_CUDA)
        sl_cuda_jit_launch(handle, n, args);
}

export extern (C) void sl_jit_free(int dev, void* handle)
{
    if (dev == DEV_CUDA)
        sl_cuda_jit_free(handle);
}

export extern (C) const(char)* sl_jit_log()
{
    return sl_cuda_jit_log();
}

// ---- data parallel: CUDA only; on meta collectives are no-ops that allocate nothing, like on the device ----

export extern (C) int sl_dist_unique_id(char* o)
{
    return sl_cuda_nccl_unique_id(o);
}

export extern (C) int sl_dist_init(int rank, int world, const(char)* id)
{
    return sl_cuda_nccl_init(rank, world, id);
}

export extern (C) int sl_dist_group(int dev, int start)
{
    return dev == DEV_CUDA ? sl_cuda_nccl_group(start) : 0;
}

export extern (C) int sl_dist_allreduce_mean(int dev, float* p, long n)
{
    return dev == DEV_CUDA ? sl_cuda_nccl_allreduce_mean(p, n) : 0;
}

export extern (C) void sl_dist_destroy()
{
    sl_cuda_nccl_destroy();
}
