#include <metal_stdlib>
using namespace metal;

// Pure same-byte read over selected expert rows, parameterized by rows per
// threadgroup and threads per threadgroup. Research only: it exists to measure
// the memory ceiling of the real expert row set, with no dequant or reduction.
struct ExpertReadArgs {
    ulong row_bytes;
    uint n_out;
    uint experts;
    uint slots;
    uint banks;
    uint rows_per_group;
    uint pad;
};

kernel void kernel_research_expert_read(
        constant ExpertReadArgs & a [[buffer(0)]],
        device const uchar * bank0 [[buffer(1)]],
        device const uchar * bank1 [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device uint * checksum [[buffer(4)]],
        uint3 g [[threadgroup_position_in_grid]],
        uint3 grid [[threadgroups_per_grid]],
        uint3 ts [[threads_per_threadgroup]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint nth = ts.x * ts.y * ts.z;
    const uint slot = g.y;
    const uint first = g.x * a.rows_per_group;
    uint acc = tid + g.x * 7u + slot;
    const int expert = slot < a.slots ? ids[slot] : -1;
    const uint count = (a.n_out > first) ? min(a.rows_per_group, a.n_out - first) : 0u;
    if (expert >= 0 && uint(expert) < a.experts && count > 0u) {
        for (uint b = 0; b < a.banks; ++b) {
            device const uchar * bank = (b == 0) ? bank0 : bank1;
            for (uint r = 0; r < count; ++r) {
                device const uchar * row =
                    bank + (ulong)(uint(expert) * a.n_out + first + r) * a.row_bytes;
                const uint nv = uint(a.row_bytes / 16u);
                for (uint v = tid; v < nv; v += nth) {
                    const uint4 q = ((device const uint4 *)row)[v];
                    acc ^= q.x ^ q.y ^ q.z ^ q.w;
                }
                for (uint off = nv * 16u + tid; off < uint(a.row_bytes); off += nth) {
                    acc ^= uint(row[off]);
                }
            }
        }
    }
    acc = simd_sum(acc);
    const uint nsimd = nth / 32u;
    if (lane == 0) {
        checksum[((g.y * grid.x + g.x) * nsimd) + sg] = acc;
    }
}
