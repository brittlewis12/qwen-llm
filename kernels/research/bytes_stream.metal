#include <metal_stdlib>
using namespace metal;

// Output-preserving same-byte stream for whole-token pricing probes. Reads the
// captured weight binding with a grid-stride uint4 scan and folds it into a
// per-simdgroup checksum so the reads cannot be eliminated. Research only.
struct BytesStreamArgs { ulong bytes; uint read, words; };

kernel void kernel_research_bytes_stream(
        constant BytesStreamArgs & a [[buffer(0)]],
        device const uint4 * src [[buffer(1)]],
        device uint * checksum [[buffer(2)]],
        uint3 g [[threadgroup_position_in_grid]],
        uint3 grid [[threadgroups_per_grid]],
        uint3 ts [[threads_per_threadgroup]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint nth = ts.x * ts.y * ts.z;
    const ulong n = (a.bytes + 15) >> 4;
    uint acc = tid + g.x;
    if (a.read) {
        for (ulong i = (ulong)g.x * nth + tid; i < n; i += (ulong)grid.x * nth) {
            const uint4 v = src[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    acc = simd_sum(acc);
    const uint nsimd = nth / 32;
    if (lane == 0) {
        checksum[((g.z * grid.y + g.y) * grid.x + g.x) * nsimd + sg] = acc;
    }
}
