// CUDA kernels behind soliton.api. Nothing here allocates except cudaMalloc on the pool's behalf: scratch space
// arrives as arguments, so every byte the kernels touch is visible to (and predictable by) the D allocator.
//
// Performance rule: long serial loops inside GPU threads run ~50x slower per element than one-thread-per-element
// kernels. So row sums go through cuBLAS gemv against a vector of ones, row maxima through short chunked loops,
// and everything else is flat.
#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <nccl.h>
#include <nvrtc.h>
#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

static cublasHandle_t H = nullptr;
static int g_err = 0;

static cublasHandle_t handle() {
    if (!H && cublasCreate(&H) != CUBLAS_STATUS_SUCCESS) g_err = 1;
    return H;
}

static inline unsigned blocks(long n) { return (unsigned)((n + 255) / 256); }
#define IDX long i = (long)blockIdx.x * blockDim.x + threadIdx.x
#define LAUNCH(kernel, n, ...)                                          \
    do {                                                                \
        if ((n) > 0) kernel<<<blocks(n), 256>>>((long)(n), __VA_ARGS__); \
        if (cudaPeekAtLastError()) g_err = 2;                           \
    } while (0)

// One block per row, threads cooperating through a shared-memory reduction: the right shape for layernorm and
// cross-entropy, which are bandwidth-bound and were costing several passes over memory each.
__device__ inline float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ inline float block_sum(float v, float* sh) {
    __syncthreads();  // the caller may still be reading sh[31] from a previous reduction
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0f;
    v = warp_sum(v);
    if (threadIdx.x == 0) sh[31] = v;
    __syncthreads();
    return sh[31];
}
__device__ inline float block_max(float v, float* sh) {
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : -INFINITY;
    v = warp_max(v);
    if (threadIdx.x == 0) sh[31] = v;
    __syncthreads();
    return sh[31];
}

struct Strided {
    int nd;
    long shape[8], sa[8], sb[8];
};

static Strided strided(int nd, const long* shape, const long* sa, const long* sb) {
    Strided s{};
    s.nd = nd;
    for (int d = 0; d < nd; d++) s.shape[d] = shape[d], s.sa[d] = sa[d], s.sb[d] = sb[d];
    return s;
}

static inline __host__ __device__ void offsets(const Strided& s, long i, long* oa, long* ob) {
    long a = 0, b = 0;
    for (int d = s.nd - 1; d >= 0; d--) {
        long k = i % s.shape[d];
        i /= s.shape[d];
        a += k * s.sa[d];
        b += k * s.sb[d];
    }
    *oa = a, *ob = b;
}

// ---- device & memory ----

int sl_cuda_set_device(int ordinal) { return cudaSetDevice(ordinal) == 0 && handle() ? 0 : -1; }
int sl_cuda_device_count() {
    int n = 0;
    return cudaGetDeviceCount(&n) == 0 ? n : 0;
}
int sl_cuda_check() {
    int e = g_err;
    if (!e && cudaPeekAtLastError()) e = 2;
    return e;
}
void sl_cuda_sync() {
    if (cudaDeviceSynchronize()) g_err = 3;
}
void sl_cuda_mem_info(size_t* free, size_t* total) { cudaMemGetInfo(free, total); }
void* sl_cuda_malloc(size_t n) {
    void* p = nullptr;
    return cudaMalloc(&p, n) == 0 ? p : nullptr;
}
void sl_cuda_free(void* p) { cudaFree(p); }
void sl_cuda_set_workspace(void* p, size_t n) { cublasSetWorkspace(handle(), p, n); }
// TF32 tensor cores: ~3x the fp32 gemm rate on Ampere, with ~10 bits of mantissa instead of 24.
void sl_cuda_set_tf32(int on) {
    if (cublasSetMathMode(handle(), on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
        g_err = 9;
}
void sl_cuda_h2d(void* d, const void* s, size_t n) { cudaMemcpy(d, s, n, cudaMemcpyHostToDevice); }
void sl_cuda_d2h(void* d, const void* s, size_t n) { cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost); }
void sl_cuda_d2d(void* d, const void* s, size_t n) { cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice); }

// ---- elementwise ----

static __global__ void k_fill(long n, float* p, float v) {
    IDX;
    if (i < n) p[i] = v;
}
void sl_cuda_fill(float* p, long n, float v) { LAUNCH(k_fill, n, p, v); }

static __global__ void k_arange(long n, int* p) {
    IDX;
    if (i < n) p[i] = (int)i;
}
void sl_cuda_arange_i32(int* p, long n) { LAUNCH(k_arange, n, p); }

static __global__ void k_binary(long n, int op, const float* a, const float* b, float* o, Strided s) {
    IDX;
    if (i >= n) return;
    long oa, ob;
    offsets(s, i, &oa, &ob);
    float x = a[oa], y = b[ob];
    o[i] = op == 0 ? x + y : op == 1 ? x - y : op == 2 ? x * y : x / y;
}
static __global__ void k_binary_suffix(long n, int op, const float* a, const float* b, float* o, long na, long nb) {
    IDX;
    if (i >= n) return;
    float x = a[i % na], y = b[i % nb];
    o[i] = op == 0 ? x + y : op == 1 ? x - y : op == 2 ? x * y : x / y;
}

// numel of the input if it is the output's trailing dims laid out contiguously (same shape, bias, mask, scalar),
// else -1.
static long suffix_numel(int nd, const long* shape, const long* s) {
    long st = 1, n = 1;
    bool leading = false;
    for (int d = nd - 1; d >= 0; d--) {
        if (!leading && s[d] == st) {
            n *= shape[d];
            st *= shape[d];
            continue;
        }
        leading = true;
        if (s[d] != 0 && shape[d] != 1) return -1;
    }
    return n;
}

void sl_cuda_binary(int op, const float* a, const float* b, float* o, int nd, const long* shape, const long* sa,
                    const long* sb) {
    long n = 1;
    for (int d = 0; d < nd; d++) n *= shape[d];
    const long na = suffix_numel(nd, shape, sa), nb = suffix_numel(nd, shape, sb);
    if (na > 0 && nb > 0) {  // residual, bias and mask adds: no per-element index decomposition
        LAUNCH(k_binary_suffix, n, op, a, b, o, na, nb);
        return;
    }
    LAUNCH(k_binary, n, op, a, b, o, strided(nd, shape, sa, sb));
}

static __global__ void k_strided_copy(long n, const float* src, long base, float* o, Strided s) {
    IDX;
    if (i >= n) return;
    long oa, ob;
    offsets(s, i, &oa, &ob);
    o[i] = src[base + oa];
}
void sl_cuda_strided_copy(const float* src, long base, float* o, int nd, const long* shape, const long* ss) {
    long n = 1;
    for (int d = 0; d < nd; d++) n *= shape[d];
    LAUNCH(k_strided_copy, n, src, base, o, strided(nd, shape, ss, ss));
}

static __global__ void k_axpy(long n, const float* x, float* y, float a) {
    IDX;
    if (i < n) y[i] += a * x[i];
}
void sl_cuda_axpy(const float* x, float* y, long n, float a) { LAUNCH(k_axpy, n, x, y, a); }

static __global__ void k_add_scalar(long n, float* p, float c) {
    IDX;
    if (i < n) p[i] += c;
}
void sl_cuda_add_scalar(float* p, long n, float c) { LAUNCH(k_add_scalar, n, p, c); }

static __global__ void k_scale(long n, const float* x, float* y, float a) {
    IDX;
    if (i < n) y[i] = a * x[i];
}
void sl_cuda_scale(const float* x, float* y, long n, float a) { LAUNCH(k_scale, n, x, y, a); }

static __global__ void k_mul(long n, const float* a, const float* b, float* o) {
    IDX;
    if (i < n) o[i] = a[i] * b[i];
}

double sl_cuda_sumsq(const float* x, long n) {
    // ponytail: float32 dot; plenty for grad-norm clipping. Use a double reduction if exact norms matter.
    float r = 0;
    if (cublasSdot_64(handle(), n, x, 1, x, 1, &r) != CUBLAS_STATUS_SUCCESS) g_err = 4;
    return r;
}

static __global__ void k_gelu(long n, const float* x, float* y) {
    IDX;
    if (i < n) y[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * 0.70710678f));
}
void sl_cuda_gelu(const float* x, float* y, long n) { LAUNCH(k_gelu, n, x, y); }

static __global__ void k_gelu_bwd(long n, const float* x, const float* dy, float* dx) {
    IDX;
    if (i >= n) return;
    const float xi = x[i];
    const float cdf = 0.5f * (1.0f + std::erf(xi * 0.70710678f));
    const float pdf = std::exp(-0.5f * xi * xi) * 0.39894228f;
    dx[i] = dy[i] * (cdf + xi * pdf);
}
void sl_cuda_gelu_bwd(const float* x, const float* dy, float* dx, long n) { LAUNCH(k_gelu_bwd, n, x, dy, dx); }

static __global__ void k_adamw(long n, float* p, const float* g, float* m, float* v, float lr, float b1, float b2,
                             float eps, float wd, double bc1, double bc2) {
    IDX;
    if (i >= n) return;
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    double mh = m[i] / bc1, vh = v[i] / bc2;
    p[i] = (float)(p[i] - lr * (mh / (sqrt(vh) + eps) + wd * p[i]));
}
void sl_cuda_adamw(float* p, const float* g, float* m, float* v, long n, float lr, float b1, float b2, float eps,
                   float wd, int step) {
    LAUNCH(k_adamw, n, p, g, m, v, lr, b1, b2, eps, wd, 1.0 - pow(b1, step), 1.0 - pow(b2, step));
}

// ---- reductions ----

// out[r] = alpha * sum_j a[r*k + j]  (row-major (rows,k) is column-major (k,rows), so this is A^T * ones)
static void row_sums(const float* a, long rows, long k, float* out, const float* ones_k, float alpha) {
    const float zero = 0;
    if (cublasSgemv_64(handle(), CUBLAS_OP_T, k, rows, &alpha, a, k, ones_k, 1, &zero, out, 1)) g_err = 7;
}

// out[j] = sum_r a[r*k + j]  (A * ones)
static void col_sums(const float* a, long rows, long k, float* out, const float* ones_rows) {
    const float one = 1, zero = 0;
    if (cublasSgemv_64(handle(), CUBLAS_OP_N, k, rows, &one, a, k, ones_rows, 1, &zero, out, 1)) g_err = 7;
}

static __global__ void k_chunk_max(long n, const float* x, float* o, long dim, long chunks, long len) {
    IDX;
    if (i >= n) return;
    const long row = i / chunks, lo = row * dim + (i % chunks) * len, end = (row + 1) * dim;
    const long hi = lo + len < end ? lo + len : end;
    float m = -INFINITY;
    for (long j = lo; j < hi; j++) m = x[j] > m ? x[j] : m;
    o[i] = m;
}

static long chunks_for(long dim) {
    long c = 1;
    while (c * c < dim) c++;
    return c;
}

// Row maxima of (outer, dim) using loops of ~sqrt(dim): chunk maxima first, then the max over chunks.
static void row_max(const float* x, long outer, long dim, long chunks, float* cmax, float* rmax) {
    LAUNCH(k_chunk_max, outer * chunks, x, cmax, dim, chunks, (dim + chunks - 1) / chunks);
    LAUNCH(k_chunk_max, outer, cmax, rmax, chunks, 1L, chunks);
}

// General case: dst element i sums its reduced region. s.shape = src shape, s.sa = dst strides (0 on reduced
// dims), s.sb = contiguous src strides.
static __global__ void k_reduce_sum(long n, const float* src, float* dst, Strided s, long nred) {
    IDX;
    if (i >= n) return;
    long base = 0, rem = i;
    for (int d = 0; d < s.nd; d++) {
        if (s.sa[d] == 0) continue;
        long k = rem / s.sa[d];
        rem %= s.sa[d];
        base += k * s.sb[d];
    }
    double acc = 0;
    for (long r = 0; r < nred; r++) {
        long off = base, q = r;
        for (int d = s.nd - 1; d >= 0; d--) {
            if (s.sa[d] != 0) continue;
            off += (q % s.shape[d]) * s.sb[d];
            q /= s.shape[d];
        }
        acc += src[off];
    }
    dst[i] = (float)acc;
}
void sl_cuda_reduce_sum(const float* src, float* dst, long ndst, int nd, const long* shape, const long* sd,
                        float* ones) {
    int p = 0;
    while (p < nd && (sd[p] == 0 || shape[p] == 1)) p++;
    bool prefix = true;
    long rows = 1;
    for (int d = 0; d < nd; d++) {
        if (d < p) rows *= shape[d];
        else if (sd[d] == 0 && shape[d] != 1) prefix = false;
    }
    if (prefix) {  // bias / positional grads: sum over leading dims
        sl_cuda_fill(ones, rows, 1);
        col_sums(src, rows, ndst, dst, ones);
        return;
    }
    long ss[8], nred = 1, st = 1;
    for (int d = nd - 1; d >= 0; d--) {
        ss[d] = st;
        st *= shape[d];
        if (sd[d] == 0) nred *= shape[d];
    }
    LAUNCH(k_reduce_sum, ndst, src, dst, strided(nd, shape, sd, ss), nred);
}

// ---- matmul (row-major on top of column-major cuBLAS) ----

void sl_cuda_matmul(const float* a, const float* b, float* o, long bt, long n, long k, long m, int ta, int tb) {
    // Row-major O = op(A) op(B) is column-major O^T = op(B)^T op(A)^T: swap operands, keep flags.
    const float one = 1, zero = 0;
    auto st = cublasSgemmStridedBatched(handle(), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                                        (int)m, (int)n, (int)k, &one, b, tb ? (int)k : (int)m, k * m, a,
                                        ta ? (int)n : (int)k, n * k, &zero, o, (int)m, n * m, (int)bt);
    if (st != CUBLAS_STATUS_SUCCESS) g_err = 5;
}

// ---- softmax ----

static __global__ void k_exp_shift(long n, const float* x, const float* m, float* y, long dim) {
    IDX;
    if (i < n) y[i] = std::exp(x[i] - m[i / dim]);
}
static __global__ void k_div_row(long n, float* y, const float* s, long dim) {
    IDX;
    if (i < n) y[i] /= s[i / dim];
}
void sl_cuda_softmax(const float* x, float* y, long outer, long dim, long chunks, float* cmax, float* rmax,
                     float* rsum, float* ones) {
    row_max(x, outer, dim, chunks, cmax, rmax);
    LAUNCH(k_exp_shift, outer * dim, x, rmax, y, dim);
    sl_cuda_fill(ones, dim, 1);
    row_sums(y, outer, dim, rsum, ones, 1);
    LAUNCH(k_div_row, outer * dim, y, rsum, dim);
}

static __global__ void k_sub_scaled_row(long n, float* o, const float* y, const float* r, long dim) {
    IDX;
    if (i < n) o[i] -= y[i] * r[i / dim];
}
void sl_cuda_softmax_bwd(const float* y, const float* dy, float* dx, long outer, long dim, float* dot, float* ones) {
    LAUNCH(k_mul, outer * dim, y, dy, dx);  // dx = y*dy, then dx -= y * sum_row(y*dy)
    sl_cuda_fill(ones, dim, 1);
    row_sums(dx, outer, dim, dot, ones, 1);
    LAUNCH(k_sub_scaled_row, outer * dim, dx, y, dot, dim);
}

// ---- layernorm ----

// Reads the row once for mean and variance, then writes it once. `ones` is unused now; kept so the pool's
// allocation pattern (and therefore the memory plan) stays the same shape as the CPU path.
static __global__ void k_ln_fwd(const float* x, const float* w, const float* b, float* y, float* mean,
                              float* invstd, long c, float eps) {
    const long r = blockIdx.x;
    const float* X = x + r * c;
    float* Y = y + r * c;
    __shared__ float sh[32];
    __shared__ float m_s, is_s;
    float s = 0, sq = 0;
    for (long j = threadIdx.x; j < c; j += blockDim.x) {
        const float v = X[j];
        s += v;
        sq += v * v;
    }
    s = block_sum(s, sh);
    sq = block_sum(sq, sh);
    if (threadIdx.x == 0) {
        m_s = s / c;
        is_s = rsqrtf(fmaxf(sq / c - m_s * m_s, 0.0f) + eps);
        mean[r] = m_s;
        invstd[r] = is_s;
    }
    __syncthreads();
    for (long j = threadIdx.x; j < c; j += blockDim.x) Y[j] = (X[j] - m_s) * is_s * w[j] + b[j];
}
void sl_cuda_layernorm(const float* x, const float* w, const float* b, float* y, float* mean, float* invstd,
                       long outer, long c, float eps, float* ones) {
    (void)ones;
    if (outer <= 0) return;
    k_ln_fwd<<<(unsigned)outer, 256>>>(x, w, b, y, mean, invstd, c, eps);
    if (cudaPeekAtLastError()) g_err = 2;
}

static __global__ void k_mul_col(long n, const float* a, const float* w, float* o, long c) {
    IDX;
    if (i < n) o[i] = a[i] * w[i % c];
}
static __global__ void k_mul_xhat(long n, const float* a, const float* x, const float* mean, const float* invstd,
                                float* o, long c) {
    IDX;
    if (i >= n) return;
    const long r = i / c;
    o[i] = a[i] * (x[i] - mean[r]) * invstd[r];
}
static __global__ void k_ln_dx(long n, const float* g, const float* x, const float* mean, const float* invstd,
                             const float* sg, const float* sgx, float* dx, long c) {
    IDX;
    if (i >= n) return;
    const long r = i / c;
    const float xh = (x[i] - mean[r]) * invstd[r];
    dx[i] = invstd[r] * (g[i] - sg[r] - xh * sgx[r]);
}
void sl_cuda_layernorm_bwd(const float* dy, const float* x, const float* w, const float* mean, const float* invstd,
                           float* dx, float* dw, float* db, long outer, long c, float* tmp, float* sg, float* sgx,
                           float* ones_c, float* ones_rows) {
    sl_cuda_fill(ones_c, c, 1);
    sl_cuda_fill(ones_rows, outer, 1);
    LAUNCH(k_mul_col, outer * c, dy, w, tmp, c);                 // g = dy * w
    row_sums(tmp, outer, c, sg, ones_c, 1.0f / c);               // mean_row(g)
    LAUNCH(k_mul_xhat, outer * c, tmp, x, mean, invstd, dx, c);  // dx as scratch: g * xhat
    row_sums(dx, outer, c, sgx, ones_c, 1.0f / c);               // mean_row(g * xhat)
    LAUNCH(k_ln_dx, outer * c, tmp, x, mean, invstd, sg, sgx, dx, c);
    LAUNCH(k_mul_xhat, outer * c, dy, x, mean, invstd, tmp, c);  // dy * xhat
    col_sums(tmp, outer, c, dw, ones_rows);
    col_sums(dy, outer, c, db, ones_rows);
}

// ---- embedding ----

static __global__ void k_embedding(long n, const float* w, const int* idx, float* o, long c) {
    IDX;
    if (i < n) o[i] = w[(long)idx[i / c] * c + i % c];
}
void sl_cuda_embedding(const float* w, const int* idx, float* o, long n, long c) {
    LAUNCH(k_embedding, n * c, w, idx, o, c);
}

size_t sl_cuda_sort_temp_bytes(long n) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr,
                                    (int)n);
    return bytes;
}

// After sorting tokens by id, the thread at the start of each run of equal ids sums the whole run.
static __global__ void k_embedding_bwd(long n, const float* dy, const int* keys, const int* perm, float* dw, long c) {
    IDX;
    if (i >= n || (i > 0 && keys[i] == keys[i - 1])) return;
    const long row = (long)keys[i] * c;
    for (long j = i; j < n && keys[j] == keys[i]; j++)
        for (long q = 0; q < c; q++) dw[row + q] += dy[(long)perm[j] * c + q];
}
void sl_cuda_embedding_bwd(const float* dy, const int* idx, float* dw, long vocab, long n, long c, int* keys,
                           int* perm, int* permOut, void* temp, size_t tempBytes) {
    if (sl_cuda_sort_temp_bytes(n) > tempBytes) {  // the pool's fixed bound must cover CUB's real need
        g_err = 8;
        return;
    }
    sl_cuda_fill(dw, vocab * c, 0);
    sl_cuda_arange_i32(perm, n);
    size_t tb = tempBytes;
    if (cub::DeviceRadixSort::SortPairs(temp, tb, idx, keys, perm, permOut, (int)n)) g_err = 6;
    LAUNCH(k_embedding_bwd, n, dy, keys, permOut, dw, c);
}

// ---- cross entropy (fused softmax + NLL) ----

// One block per row: two passes over the row (max, then sum of exp) instead of several over all of memory.
static __global__ void k_xent(const float* logits, const int* t, float* rowloss, long v) {
    const long r = blockIdx.x;
    const float* L = logits + r * v;
    __shared__ float sh[32];
    float mx = -INFINITY;
    for (long j = threadIdx.x; j < v; j += blockDim.x) mx = fmaxf(mx, L[j]);
    mx = block_max(mx, sh);
    float z = 0;
    for (long j = threadIdx.x; j < v; j += blockDim.x) z += std::exp(L[j] - mx);
    z = block_sum(z, sh);
    if (threadIdx.x == 0) rowloss[r] = logf(z) + mx - L[t[r]];
}
double sl_cuda_cross_entropy(const float* logits, const int* t, long n, long v, float* rowloss) {
    if (n > 0) k_xent<<<(unsigned)n, 256>>>(logits, t, rowloss, v);
    if (cudaPeekAtLastError()) g_err = 2;
    std::vector<float> host(n);
    cudaMemcpy(host.data(), rowloss, n * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0;
    for (float x : host) s += x;
    return s / n;
}

// One block per row: read the logits twice, write the gradient once.
static __global__ void k_xent_bwd(const float* logits, const int* t, float* dl, long v, float g) {
    const long r = blockIdx.x;
    const float* L = logits + r * v;
    float* D = dl + r * v;
    __shared__ float sh[32];
    float mx = -INFINITY;
    for (long j = threadIdx.x; j < v; j += blockDim.x) mx = fmaxf(mx, L[j]);
    mx = block_max(mx, sh);
    float z = 0;
    for (long j = threadIdx.x; j < v; j += blockDim.x) z += std::exp(L[j] - mx);
    z = block_sum(z, sh);
    const float scale = g / z;
    for (long j = threadIdx.x; j < v; j += blockDim.x) D[j] = std::exp(L[j] - mx) * scale;
    __syncthreads();  // the target element is written by whichever thread owns it; subtract only after that
    if (threadIdx.x == 0) D[t[r]] -= g;
}
void sl_cuda_cross_entropy_bwd(const float* logits, const int* t, float* dl, long n, long v, float dloss) {
    if (n > 0) k_xent_bwd<<<(unsigned)n, 256>>>(logits, t, dl, v, dloss / n);
    if (cudaPeekAtLastError()) g_err = 2;
}

// ---- fused causal attention ----
//
// Flash-style: query tiles x key tiles with an online softmax, so the (B,H,T,T) score matrix never exists and
// causal tiles above the diagonal are skipped entirely. qkv arrives packed as (B,T,3,H,hs) straight from one
// projection gemm, so no transpose copies are needed; the per-(b,h) matrices are addressed with pointer arrays.

static __global__ void k_ptrs(long n, const float** out, const float* base, long stride_b, long stride_h, long heads,
                            long extra) {
    IDX;
    if (i < n) out[i] = base + (i / heads) * stride_b + (i % heads) * stride_h + extra;
}

static void fill_ptrs(const float** arr, const float* base, long B, long H, long sb, long sh, long extra) {
    LAUNCH(k_ptrs, B * H, arr, base, sb, sh, H, extra);
}

// row-major C(n,m) = alpha * op(A)(n,k) @ op(B)(k,m) + beta*C, batched over pointer arrays
static void rm_gemm_b(int ta, int tb, long n, long k, long m, float alpha, const float** A, long lda,
                      const float** B, long ldb, float beta, float** C, long ldc, int batch) {
    auto st = cublasSgemmBatched(handle(), tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, (int)m,
                                 (int)n, (int)k, &alpha, B, (int)ldb, A, (int)lda, &beta, C, (int)ldc, batch);
    if (st != CUBLAS_STATUS_SUCCESS) g_err = 10;
}

static __global__ void k_causal_mask(long n, float* s, long tq, long tk, long q0, long k0) {
    IDX;
    if (i >= n) return;
    const long q = (i / tk) % tq, kk = i % tk;
    if (k0 + kk > q0 + q) s[i] = -INFINITY;
}
static __global__ void k_online_m(long n, float* m, const float* rowmax, float* c) {
    IDX;
    if (i >= n) return;
    const float mo = m[i], mn = fmaxf(mo, rowmax[i]);
    c[i] = mo == -INFINITY ? 0.0f : std::exp(mo - mn);
    m[i] = mn;
}
static __global__ void k_l_update(long n, float* l, const float* c, const float* rowsum) {
    IDX;
    if (i < n) l[i] = l[i] * c[i] + rowsum[i];
}
static __global__ void k_scale_rows(long n, float* o, const float* c, long d) {
    IDX;
    if (i < n) o[i] *= c[i / d];
}
// acc (B,H,tq,hs) -> out (B,T,H,hs), dividing by the softmax denominator
static __global__ void k_attn_write(long n, const float* acc, const float* l, float* out, long tq, long hs, long q0,
                                  long T, long H) {
    IDX;
    if (i >= n) return;
    const long d = i % hs, q = (i / hs) % tq, h = (i / (hs * tq)) % H, b = i / (hs * tq * H);
    out[((b * T + q0 + q) * H + h) * hs + d] = acc[i] / l[(b * H + h) * tq + q];
}
static __global__ void k_attn_gather(long n, const float* src, float* dst, long tq, long hs, long q0, long T, long H) {
    IDX;
    if (i >= n) return;
    const long d = i % hs, q = (i / hs) % tq, h = (i / (hs * tq)) % H, b = i / (hs * tq * H);
    dst[i] = src[((b * T + q0 + q) * H + h) * hs + d];
}
static __global__ void k_L_write(long n, const float* m, const float* l, float* L, long tq, long q0, long T) {
    IDX;
    if (i >= n) return;
    L[(i / tq) * T + q0 + (i % tq)] = m[i] + std::log(l[i]);
}
static __global__ void k_L_read(long n, const float* L, float* rows, long tq, long q0, long T) {
    IDX;
    if (i < n) rows[i] = L[(i / tq) * T + q0 + (i % tq)];
}
static __global__ void k_ds(long n, float* dp, const float* p, const float* D, long tk, float scale) {
    IDX;
    if (i < n) dp[i] = p[i] * (dp[i] - D[i / tk]) * scale;
}

// 256 when it divides the sequence, else one tile (keeps every tile full; no partial-tile paths).
// ponytail: measured 512 too -- 1.5% slower and +19MB, because a larger diagonal tile wastes more work
// under the causal mask. 256 stays.
long sl_cuda_attn_tile(long T) { return T % 256 == 0 ? 256 : T; }

void sl_cuda_attention(const float* qkv, float* out, float* L, long B, long T, long H, long hs, float scale,
                       long tile, float* s, float* acc, float* m, float* l, float* c, float* rowbuf, float* ones,
                       void* ptrs) {
    const long bh = B * H, ld = 3 * H * hs, sb = T * ld, rows = bh * tile;
    const float** pQ = (const float**)ptrs;
    const float** pK = pQ + bh;
    const float** pV = pK + bh;
    const float** pS = pV + bh;
    const float** pAcc = pS + bh;
    const long chunks = chunks_for(tile);
    float* rmax = rowbuf + rows * chunks;  // row_max fills rows*chunks chunk maxima, then the row maxima
    sl_cuda_fill(ones, tile > hs ? tile : hs, 1);
    fill_ptrs(pS, s, B, H, (long)H * tile * tile, tile * tile, 0);      // s is contiguous (B,H,tile,tile)
    fill_ptrs(pAcc, acc, B, H, (long)H * tile * hs, tile * hs, 0);      // acc is contiguous (B,H,tile,hs)
    for (long q0 = 0; q0 < T; q0 += tile) {
        sl_cuda_fill(m, rows, -INFINITY);
        sl_cuda_fill(l, rows, 0);
        sl_cuda_fill(acc, rows * hs, 0);
        fill_ptrs(pQ, qkv, B, H, sb, hs, q0 * ld);
        for (long k0 = 0; k0 <= q0; k0 += tile) {
            fill_ptrs(pK, qkv, B, H, sb, hs, k0 * ld + H * hs);
            fill_ptrs(pV, qkv, B, H, sb, hs, k0 * ld + 2 * H * hs);
            rm_gemm_b(0, 1, tile, hs, tile, scale, pQ, ld, pK, ld, 0, (float**)pS, tile, (int)bh);
            if (k0 == q0) LAUNCH(k_causal_mask, rows * tile, s, tile, tile, q0, k0);
            row_max(s, rows, tile, chunks, rowbuf, rmax);
            LAUNCH(k_online_m, rows, m, rmax, c);
            LAUNCH(k_exp_shift, rows * tile, s, m, s, tile);  // s becomes P in place
            row_sums(s, rows, tile, rowbuf, ones, 1);
            LAUNCH(k_l_update, rows, l, c, rowbuf);
            LAUNCH(k_scale_rows, rows * hs, acc, c, hs);
            rm_gemm_b(0, 0, tile, tile, hs, 1, pS, tile, pV, ld, 1, (float**)pAcc, hs, (int)bh);
        }
        LAUNCH(k_attn_write, rows * hs, acc, l, out, tile, hs, q0, T, H);
        LAUNCH(k_L_write, rows, m, l, L, tile, q0, T);
    }
}

void sl_cuda_attention_bwd(const float* qkv, const float* out, const float* L, const float* dout, float* dqkv,
                           long B, long T, long H, long hs, float scale, long tile, float* s, float* dp, float* doc,
                           float* oc, float* lrow, float* D, float* rowbuf, float* ones, void* ptrs) {
    const long bh = B * H, ld = 3 * H * hs, sb = T * ld, rows = bh * tile;
    const float** pQ = (const float**)ptrs;
    const float** pK = pQ + bh;
    const float** pV = pK + bh;
    const float** pS = pV + bh;
    const float** pdOc = pS + bh;
    const float** pdQ = pdOc + bh;
    const float** pdK = pdQ + bh;
    const float** pdV = pdK + bh;
    const float** pdP = pdV + bh;
    sl_cuda_fill(dqkv, B * T * 3 * H * hs, 0);
    sl_cuda_fill(ones, tile > hs ? tile : hs, 1);
    fill_ptrs(pS, s, B, H, (long)H * tile * tile, tile * tile, 0);
    fill_ptrs(pdP, dp, B, H, (long)H * tile * tile, tile * tile, 0);
    fill_ptrs(pdOc, doc, B, H, (long)H * tile * hs, tile * hs, 0);
    for (long q0 = 0; q0 < T; q0 += tile) {
        LAUNCH(k_attn_gather, rows * hs, dout, doc, tile, hs, q0, T, H);
        LAUNCH(k_attn_gather, rows * hs, out, oc, tile, hs, q0, T, H);
        LAUNCH(k_L_read, rows, L, lrow, tile, q0, T);
        LAUNCH(k_mul, rows * hs, doc, oc, oc);  // oc = dO*O, then row sums give D
        row_sums(oc, rows, hs, D, ones, 1);
        fill_ptrs(pQ, qkv, B, H, sb, hs, q0 * ld);
        fill_ptrs(pdQ, dqkv, B, H, sb, hs, q0 * ld);
        for (long k0 = 0; k0 <= q0; k0 += tile) {
            fill_ptrs(pK, qkv, B, H, sb, hs, k0 * ld + H * hs);
            fill_ptrs(pV, qkv, B, H, sb, hs, k0 * ld + 2 * H * hs);
            fill_ptrs(pdK, dqkv, B, H, sb, hs, k0 * ld + H * hs);
            fill_ptrs(pdV, dqkv, B, H, sb, hs, k0 * ld + 2 * H * hs);
            rm_gemm_b(0, 1, tile, hs, tile, scale, pQ, ld, pK, ld, 0, (float**)pS, tile, (int)bh);
            if (k0 == q0) LAUNCH(k_causal_mask, rows * tile, s, tile, tile, q0, k0);
            LAUNCH(k_exp_shift, rows * tile, s, lrow, s, tile);  // s = P = exp(S - L)
            rm_gemm_b(1, 0, tile, tile, hs, 1, pS, tile, pdOc, hs, 1, (float**)pdV, ld, (int)bh);
            rm_gemm_b(0, 1, tile, hs, tile, 1, pdOc, hs, pV, ld, 0, (float**)pdP, tile, (int)bh);
            LAUNCH(k_ds, rows * tile, dp, s, D, tile, scale);
            rm_gemm_b(0, 0, tile, tile, hs, 1, pdP, tile, pK, ld, 1, (float**)pdQ, ld, (int)bh);
            rm_gemm_b(1, 0, tile, tile, hs, 1, pdP, tile, pQ, ld, 1, (float**)pdK, ld, (int)bh);
        }
    }
}

// ---- runtime kernel compilation ----
//
// Fused elementwise chains are generated as CUDA source and compiled here by NVRTC, which ships inside the
// CUDA driver. No LLVM, no Triton, no Python in the loop. A compiled kernel is cached by its source, and the
// generated kernels take the element count as an argument, so changing shapes never triggers a recompile.

struct Jit {
    CUmodule mod;
    CUfunction fn;
};
static std::string g_jit_log;

const char* sl_cuda_jit_log() { return g_jit_log.c_str(); }

int sl_cuda_jit(const char* src, const char* name, void** out) {
    int major = 8, minor = 6, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", major, minor);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, name, 0, nullptr, nullptr) != NVRTC_SUCCESS) return -1;
    const char* opts[] = {arch, "--std=c++14"};
    const bool ok = nvrtcCompileProgram(prog, 2, opts) == NVRTC_SUCCESS;
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    g_jit_log.assign(logSize, '\0');
    if (logSize) nvrtcGetProgramLog(prog, &g_jit_log[0]);
    if (!ok) {
        nvrtcDestroyProgram(&prog);
        return -2;
    }
    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    Jit j{};
    if (cuModuleLoadData(&j.mod, ptx.data()) != CUDA_SUCCESS) return -3;
    if (cuModuleGetFunction(&j.fn, j.mod, name) != CUDA_SUCCESS) return -4;
    *out = new Jit(j);
    return 0;
}

void sl_cuda_jit_launch(void* handle, long n, void** args) {
    if (n <= 0) return;
    auto* j = (Jit*)handle;
    const unsigned block = 256, grid = (unsigned)((n + block - 1) / block);
    if (cuLaunchKernel(j->fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) g_err = 11;
}

void sl_cuda_jit_free(void* handle) {
    auto* j = (Jit*)handle;
    if (j) {
        cuModuleUnload(j->mod);
        delete j;
    }
}

// ---- data parallel (NCCL) ----
// NCCL keeps its own GPU buffers outside the pool; they show up as fixed per-process overhead, not in plans.

static ncclComm_t g_comm = nullptr;

int sl_cuda_nccl_unique_id(char* out) {
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) return -1;
    memcpy(out, id.internal, NCCL_UNIQUE_ID_BYTES);
    return 0;
}
int sl_cuda_nccl_init(int rank, int world, const char* id_bytes) {
    ncclUniqueId id;
    memcpy(id.internal, id_bytes, NCCL_UNIQUE_ID_BYTES);
    return ncclCommInitRank(&g_comm, world, id, rank) == ncclSuccess ? 0 : -1;
}
int sl_cuda_nccl_group(int start) { return (start ? ncclGroupStart() : ncclGroupEnd()) == ncclSuccess ? 0 : -1; }
int sl_cuda_nccl_allreduce_mean(float* p, long n) {
    return ncclAllReduce(p, p, (size_t)n, ncclFloat32, ncclAvg, g_comm, nullptr) == ncclSuccess ? 0 : -1;
}
void sl_cuda_nccl_destroy() {
    if (g_comm) ncclCommDestroy(g_comm);
    g_comm = nullptr;
}

}  // extern "C"
