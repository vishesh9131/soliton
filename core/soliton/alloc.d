/// Deterministic caching allocator, one pool per device kind.
///
/// This is the heart of Soliton's memory prediction: the meta device runs the
/// exact same pool policy with fake addresses, so a dry run on meta reports the
/// same reserved/peak bytes the real device will see.
module soliton.alloc;

import core.stdc.stdlib : free;
import core.stdc.stdlib : realloc;
import core.sys.posix.stdlib : posix_memalign;

enum DEV_CPU = 0, DEV_META = 1, DEV_CUDA = 2;

extern (C) nothrow @nogc
{
    void* sl_cuda_malloc(size_t n);
    void sl_cuda_free(void* p);
}

struct Block
{
    size_t size;
    void* ptr;
}

struct Pool
{
    int dev;
    Block* cache;
    size_t ncache, capcache;
    long allocated, reserved, peakAllocated, peakReserved, nAlloc, nRaw;
    long limit; // 0 = unlimited; reserving past it fails like a real OOM
    ulong traceHash = 1469598103934665603UL; // FNV-1a over (kind, size) events
    size_t metaNext = 4096;

    // Trace recording (on meta): every allocation becomes a distinct block with a lifetime, so a planner
    // can assign fixed offsets. The block cache is bypassed while recording, or lifetimes would merge.
    bool recording;
    long* recSize, recStart, recEnd;
    size_t recCount, recCap;
    long[64] marks;
    size_t nmarks;

    // Arena replay: hand out base+offset from one allocation, in the recorded order.
    void* arena;
    long arenaBytes;
    const(long)* planOffset, planSize;
    size_t planCount, planCursor, planBody;
    bool arenaMode, diverged;

nothrow @nogc:

    // Small requests share 512-byte classes, large ones 2 MiB classes, so exact-size reuse hits often.
    static size_t roundSize(size_t n)
    {
        enum small = 512, large = 2 << 20;
        if (n == 0)
            n = 1;
        return n < (1 << 20) ? (n + small - 1) / small * small : (n + large - 1) / large * large;
    }

    void mix(ulong kind, size_t n)
    {
        traceHash = (traceHash ^ kind) * 1099511628211UL;
        traceHash = (traceHash ^ n) * 1099511628211UL;
    }

    void* rawAlloc(size_t n)
    {
        if (limit && reserved + cast(long) n > limit)
            return null;
        final switch (dev)
        {
        case DEV_CPU:
            void* p;
            return posix_memalign(&p, 64, n) == 0 ? p : null;
        case DEV_META:
            void* p = cast(void*) metaNext; // never dereferenced
            metaNext += n;
            return p;
        case DEV_CUDA:
            return sl_cuda_malloc(n);
        }
    }

    void rawFree(void* p)
    {
        if (dev == DEV_CPU)
            free(p);
        else if (dev == DEV_CUDA)
            sl_cuda_free(p);
    }

    // ponytail: exact-size block cache, linear scan, no splitting. O(cached blocks) per alloc and
    // wastes memory on odd shapes; add best-fit splitting when fragmentation shows up in real runs.
    /// Replay: the k-th allocation of the run gets the k-th planned offset. A size mismatch means the run
    /// diverged from the plan, which must fail loudly rather than hand back the wrong buffer.
    void* arenaAlloc(size_t n)
    {
        if (planCursor >= planCount || planSize[planCursor] != cast(long) n)
        {
            diverged = true;
            return null;
        }
        auto p = cast(void*)(cast(ubyte*) arena + planOffset[planCursor++]);
        allocated += n;
        nAlloc++;
        if (allocated > peakAllocated)
            peakAllocated = allocated;
        return p;
    }

    void record(size_t n, size_t index)
    {
        if (recCount == recCap)
        {
            recCap = recCap ? recCap * 2 : 4096;
            recSize = cast(long*) realloc(recSize, recCap * long.sizeof);
            recStart = cast(long*) realloc(recStart, recCap * long.sizeof);
            recEnd = cast(long*) realloc(recEnd, recCap * long.sizeof);
        }
        recSize[recCount] = cast(long) n;
        recStart[recCount] = cast(long) index;
        recEnd[recCount] = -1; // still live at the end of the trace
        recCount++;
    }

    void* alloc(size_t n)
    {
        n = roundSize(n);
        mix(1, n);
        if (arenaMode)
            return arenaAlloc(n);
        if (recording)
        {
            // Encode the record index in the (never dereferenced) pointer so free() can find it.
            record(n, recCount);
            allocated += n;
            nAlloc++;
            if (allocated > peakAllocated)
                peakAllocated = allocated;
            return cast(void*)(4096 + (recCount - 1) * 64);
        }
        void* p;
        foreach (i; 0 .. ncache)
            if (cache[i].size == n)
            {
                p = cache[i].ptr;
                cache[i] = cache[--ncache];
                break;
            }
        if (!p)
        {
            p = rawAlloc(n);
            if (!p)
            {
                emptyCache();
                p = rawAlloc(n);
            }
            if (!p)
                return null;
            reserved += n;
            nRaw++;
            if (reserved > peakReserved)
                peakReserved = reserved;
        }
        allocated += n;
        nAlloc++;
        if (allocated > peakAllocated)
            peakAllocated = allocated;
        return p;
    }

    void release(void* p, size_t n)
    {
        n = roundSize(n);
        mix(2, n);
        allocated -= n;
        if (arena && p >= arena && p < cast(void*)(cast(ubyte*) arena + arenaBytes))
        {
            // An arena-backed block must never enter the free cache: the cache outlives the arena, and
            // handing one of these pointers out afterwards is a use-after-free.
            if (!arenaMode && allocated == 0)
                releaseArena();
            return;
        }
        if (recording)
        {
            const idx = (cast(size_t) p - 4096) / 64;
            if (idx < recCount)
                recEnd[idx] = cast(long) recCount;
            return;
        }
        if (ncache == capcache)
        {
            capcache = capcache ? capcache * 2 : 256;
            cache = cast(Block*) realloc(cache, capcache * Block.sizeof);
        }
        cache[ncache++] = Block(n, p);
    }

    /// ponytail: frees the arena only when nothing at all is live. A pool block outliving the arena keeps it
    /// alive a little longer than needed; track arena blocks separately if that ever matters.
    void releaseArena()
    {
        if (!arena)
            return;
        rawFree(arena);
        reserved -= arenaBytes;
        arena = null;
        arenaBytes = 0;
    }

    void emptyCache()
    {
        foreach (i; 0 .. ncache)
        {
            rawFree(cache[i].ptr);
            reserved -= cache[i].size;
        }
        ncache = 0;
    }
}

__gshared Pool[3] pools = [Pool(DEV_CPU), Pool(DEV_META), Pool(DEV_CUDA)];
