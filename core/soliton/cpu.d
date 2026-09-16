/// Reference CPU kernels. float32 data, int32 indices, contiguous outputs.
/// Correctness first: the CUDA kernels are tested against these.
module soliton.cpu;

import core.stdc.math : exp, log, sqrt, erf, tanh, pow;
import core.stdc.string : memset, memcpy;

nothrow @nogc:

/// Walks a contiguous output of `shape` while tracking offsets into up to two strided inputs.
private mixin template StridedLoop(string body_)
{
    void run()
    {
        long n = 1;
        foreach (d; 0 .. nd)
            n *= shape[d];
        long[8] idx = 0;
        long oa = 0, ob = 0;
        for (long i = 0; i < n; i++)
        {
            mixin(body_);
            for (int d = nd - 1; d >= 0; d--)
            {
                idx[d]++;
                oa += sa[d];
                ob += sb[d];
                if (idx[d] < shape[d])
                    break;
                oa -= sa[d] * shape[d];
                ob -= sb[d] * shape[d];
                idx[d] = 0;
            }
        }
    }
}

private void binaryT(string s)(const(float)* a, const(float)* b, float* o, int nd,
    const(long)* shape, const(long)* sa, const(long)* sb)
{
    mixin StridedLoop!("o[i] = a[oa] " ~ s ~ " b[ob];");
    run();
}

/// op: 0 add, 1 sub, 2 mul, 3 div. Output contiguous with `shape`; sa/sb are 0 on broadcast dims.
void binary(int op, const(float)* a, const(float)* b, float* o, int nd, const(long)* shape,
    const(long)* sa, const(long)* sb)
{
    switch (op)
    {
    case 0:
        return binaryT!"+"(a, b, o, nd, shape, sa, sb);
    case 1:
        return binaryT!"-"(a, b, o, nd, shape, sa, sb);
    case 2:
        return binaryT!"*"(a, b, o, nd, shape, sa, sb);
    default:
        return binaryT!"/"(a, b, o, nd, shape, sa, sb);
    }
}

/// dst[i] = src[base + sum(idx*ss)] -- permute, expand and narrow are all this.
void stridedCopy(const(float)* src, long base, float* o, int nd, const(long)* shape, const(long)* ss)
{
    const(long)* sa = ss, sb = ss;
    mixin StridedLoop!("o[i] = src[base + oa];");
    run();
}

/// dst[strided offset] += src[i]; dst strides are 0 on reduced dims.
void reduceSum(const(float)* src, float* dst, long ndst, int nd, const(long)* shape, const(long)* sd)
{
    memset(dst, 0, ndst * float.sizeof);
    const(long)* sa = sd, sb = sd;
    mixin StridedLoop!("dst[oa] += src[i];");
    run();
}

void fill(float* p, long n, float v)
{
    foreach (i; 0 .. n)
        p[i] = v;
}

void axpy(const(float)* x, float* y, long n, float alpha)
{
    foreach (i; 0 .. n)
        y[i] += alpha * x[i];
}

void addScalar(float* p, long n, float c)
{
    foreach (i; 0 .. n)
        p[i] += c;
}

void scale(const(float)* x, float* y, long n, float alpha)
{
    foreach (i; 0 .. n)
        y[i] = alpha * x[i];
}

double sumsq(const(float)* x, long n)
{
    double s = 0;
    foreach (i; 0 .. n)
        s += cast(double) x[i] * x[i];
    return s;
}

/// out = op(A) @ op(B), batched with equal batch counts. A:(bt,n,k) B:(bt,k,m) after transposes.
void matmul(const(float)* a, const(float)* b, float* o, long bt, long n, long k, long m, int ta, int tb)
{
    memset(o, 0, bt * n * m * float.sizeof);
    // stride of A over (row i, col p) and B over (row p, col j), honoring transposes.
    long ar = ta ? 1 : k, ac = ta ? n : 1;
    long br = tb ? 1 : m, bc = tb ? k : 1;
    foreach (t; 0 .. bt)
    {
        const(float)* A = a + t * n * k, B = b + t * k * m;
        float* O = o + t * n * m;
        // ponytail: naive ikj loop; the CPU path is a test oracle. Link a BLAS if CPU training matters.
        foreach (i; 0 .. n)
            foreach (p; 0 .. k)
            {
                const float av = A[i * ar + p * ac];
                foreach (j; 0 .. m)
                    O[i * m + j] += av * B[p * br + j * bc];
            }
    }
}

void softmax(const(float)* x, float* y, long outer, long dim)
{
    foreach (r; 0 .. outer)
    {
        const(float)* X = x + r * dim;
        float* Y = y + r * dim;
        float mx = -float.infinity;
        foreach (i; 0 .. dim)
            if (X[i] > mx)
                mx = X[i];
        double z = 0;
        foreach (i; 0 .. dim)
            z += exp(cast(double)(X[i] - mx));
        foreach (i; 0 .. dim)
            Y[i] = cast(float)(exp(cast(double)(X[i] - mx)) / z);
    }
}

void softmaxBwd(const(float)* y, const(float)* dy, float* dx, long outer, long dim)
{
    foreach (r; 0 .. outer)
    {
        const(float)* Y = y + r * dim, DY = dy + r * dim;
        float* DX = dx + r * dim;
        double dot = 0;
        foreach (i; 0 .. dim)
            dot += cast(double) Y[i] * DY[i];
        foreach (i; 0 .. dim)
            DX[i] = cast(float)(Y[i] * (DY[i] - dot));
    }
}

// Exact GELU (erf form), same as GPT-2 reference implementations in float32.
void gelu(const(float)* x, float* y, long n)
{
    foreach (i; 0 .. n)
        y[i] = 0.5f * x[i] * (1.0f + cast(float) erf(x[i] / sqrt(2.0)));
}

void geluBwd(const(float)* x, const(float)* dy, float* dx, long n)
{
    enum inv = 0.3989422804014327; // 1/sqrt(2*pi)
    foreach (i; 0 .. n)
    {
        const double cdf = 0.5 * (1.0 + erf(x[i] / sqrt(2.0)));
        const double pdf = inv * exp(-0.5 * x[i] * x[i]);
        dx[i] = cast(float)(dy[i] * (cdf + x[i] * pdf));
    }
}

void layernorm(const(float)* x, const(float)* w, const(float)* b, float* y, float* mean,
    float* invstd, long outer, long c, float eps)
{
    foreach (r; 0 .. outer)
    {
        const(float)* X = x + r * c;
        float* Y = y + r * c;
        double m = 0, v = 0;
        foreach (i; 0 .. c)
            m += X[i];
        m /= c;
        foreach (i; 0 .. c)
            v += (X[i] - m) * (X[i] - m);
        v /= c;
        const double s = 1.0 / sqrt(v + eps);
        foreach (i; 0 .. c)
            Y[i] = cast(float)((X[i] - m) * s * w[i] + b[i]);
        mean[r] = cast(float) m;
        invstd[r] = cast(float) s;
    }
}

void layernormBwd(const(float)* dy, const(float)* x, const(float)* w, const(float)* mean,
    const(float)* invstd, float* dx, float* dw, float* db, long outer, long c)
{
    memset(dw, 0, c * float.sizeof);
    memset(db, 0, c * float.sizeof);
    foreach (r; 0 .. outer)
    {
        const(float)* DY = dy + r * c, X = x + r * c;
        float* DX = dx + r * c;
        const double s = invstd[r], m = mean[r];
        double sg = 0, sgx = 0; // sum(dxhat), sum(dxhat * xhat)
        foreach (i; 0 .. c)
        {
            const double xh = (X[i] - m) * s, g = DY[i] * w[i];
            dw[i] += cast(float)(DY[i] * xh);
            db[i] += DY[i];
            sg += g;
            sgx += g * xh;
        }
        foreach (i; 0 .. c)
        {
            const double xh = (X[i] - m) * s;
            DX[i] = cast(float)(s * (DY[i] * w[i] - sg / c - xh * sgx / c));
        }
    }
}

void embedding(const(float)* w, const(int)* idx, float* o, long n, long c)
{
    foreach (i; 0 .. n)
        memcpy(o + i * c, w + idx[i] * c, c * float.sizeof);
}

void embeddingBwd(const(float)* dy, const(int)* idx, float* dw, long vocab, long n, long c)
{
    memset(dw, 0, vocab * c * float.sizeof);
    foreach (i; 0 .. n)
        foreach (j; 0 .. c)
            dw[idx[i] * c + j] += dy[i * c + j];
}

/// Mean cross entropy over n rows; returns the loss.
double crossEntropy(const(float)* logits, const(int)* t, long n, long v)
{
    double total = 0;
    foreach (r; 0 .. n)
    {
        const(float)* L = logits + r * v;
        float mx = -float.infinity;
        foreach (i; 0 .. v)
            if (L[i] > mx)
                mx = L[i];
        double z = 0;
        foreach (i; 0 .. v)
            z += exp(cast(double)(L[i] - mx));
        total += log(z) + mx - L[t[r]];
    }
    return total / n;
}

void crossEntropyBwd(const(float)* logits, const(int)* t, float* dl, long n, long v, float dloss)
{
    softmax(logits, dl, n, v);
    const float g = dloss / n;
    foreach (r; 0 .. n)
    {
        foreach (i; 0 .. v)
            dl[r * v + i] *= g;
        dl[r * v + t[r]] -= g;
    }
}

/// Reference causal attention. qkv is packed (B,T,3,H,hs), out is (B,T,H,hs), L is (B,H,T) logsumexp.
/// ponytail: plain O(T^2) loops -- this is the oracle the CUDA version is tested against, not a fast path.
void attention(const(float)* qkv, float* o, float* L, long B, long T, long H, long hs, float scale)
{
    const long ld = 3 * H * hs;
    foreach (b; 0 .. B)
        foreach (h; 0 .. H)
            foreach (t; 0 .. T)
            {
                const(float)* q = qkv + (b * T + t) * ld + h * hs;
                double mx = -double.infinity;
                foreach (k; 0 .. t + 1)
                {
                    const(float)* kk = qkv + (b * T + k) * ld + H * hs + h * hs;
                    double s = 0;
                    foreach (d; 0 .. hs)
                        s += q[d] * kk[d];
                    s *= scale;
                    if (s > mx)
                        mx = s;
                }
                double z = 0;
                float* dst = o + ((b * T + t) * H + h) * hs;
                foreach (d; 0 .. hs)
                    dst[d] = 0;
                foreach (k; 0 .. t + 1)
                {
                    const(float)* kk = qkv + (b * T + k) * ld + H * hs + h * hs;
                    const(float)* vv = qkv + (b * T + k) * ld + 2 * H * hs + h * hs;
                    double s = 0;
                    foreach (d; 0 .. hs)
                        s += q[d] * kk[d];
                    const double p = exp(s * scale - mx);
                    z += p;
                    foreach (d; 0 .. hs)
                        dst[d] += cast(float)(p * vv[d]);
                }
                foreach (d; 0 .. hs)
                    dst[d] = cast(float)(dst[d] / z);
                L[(b * H + h) * T + t] = cast(float)(mx + log(z));
            }
}

void attentionBwd(const(float)* qkv, const(float)* o, const(float)* L, const(float)* dout, float* dqkv,
    long B, long T, long H, long hs, float scale)
{
    const long ld = 3 * H * hs;
    memset(dqkv, 0, B * T * ld * float.sizeof);
    foreach (b; 0 .. B)
        foreach (h; 0 .. H)
            foreach (t; 0 .. T)
            {
                const(float)* q = qkv + (b * T + t) * ld + h * hs;
                const(float)* dO = dout + ((b * T + t) * H + h) * hs;
                const(float)* oo = o + ((b * T + t) * H + h) * hs;
                float* dq = dqkv + (b * T + t) * ld + h * hs;
                double D = 0;
                foreach (d; 0 .. hs)
                    D += dO[d] * oo[d];
                const double lse = L[(b * H + h) * T + t];
                foreach (k; 0 .. t + 1)
                {
                    const(float)* kk = qkv + (b * T + k) * ld + H * hs + h * hs;
                    const(float)* vv = qkv + (b * T + k) * ld + 2 * H * hs + h * hs;
                    float* dk = dqkv + (b * T + k) * ld + H * hs + h * hs;
                    float* dv = dqkv + (b * T + k) * ld + 2 * H * hs + h * hs;
                    double s = 0, dp = 0;
                    foreach (d; 0 .. hs)
                        s += q[d] * kk[d];
                    const double p = exp(s * scale - lse);
                    foreach (d; 0 .. hs)
                    {
                        dv[d] += cast(float)(p * dO[d]);
                        dp += dO[d] * vv[d];
                    }
                    const double ds = p * (dp - D) * scale;
                    foreach (d; 0 .. hs)
                    {
                        dq[d] += cast(float)(ds * kk[d]);
                        dk[d] += cast(float)(ds * q[d]);
                    }
                }
            }
}

void adamw(float* p, const(float)* g, float* m, float* v, long n, float lr, float b1, float b2,
    float eps, float wd, int step)
{
    const double bc1 = 1.0 - pow(b1, step), bc2 = 1.0 - pow(b2, step);
    foreach (i; 0 .. n)
    {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        const double mh = m[i] / bc1, vh = v[i] / bc2;
        p[i] = cast(float)(p[i] - lr * (mh / (sqrt(vh) + eps) + wd * p[i]));
    }
}
