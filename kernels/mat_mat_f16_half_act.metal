// F16 mat-mat using simdgroup half tiles and half-staged activations.
//
// This matches the prompt-time 64x32x32 shape used by the quantized mat-mat
// kernels. It is intentionally gated separately from the scalar exact F32-act
// fallback because activations are rounded to half before accumulation.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

struct mat_mat_f16_half_act_args {
    uint n_in;
    uint n_out;
    uint n_query;
};

kernel void kernel_mat_mat_f16_half_act_f32(
        constant mat_mat_f16_half_act_args & args [[buffer(0)]],
        device const half * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;

    const short nr0 = ((int)args.n_out - r0 < 64) ? (short)((int)args.n_out - r0) : 64;
    const short nr1 = ((int)args.n_query - r1 < 32) ? (short)((int)args.n_query - r1) : 32;

    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;
    const short il0 = tiitg % 2;
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;
    const short iy = 8 * (tiitg % 4);

    device const half * a_ptr = weight + (ulong)args.n_in * (r0 + lr0) + 16u * (ulong)il0;
    device const float * b_ptr = x + (ulong)args.n_in * (r1 + lr1) + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.n_in; loop_k += 32) {
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = a_ptr[i];
            }
        }

        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)b_ptr));
        }

        a_ptr += 32;
        b_ptr += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                        64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += 32) {
            device float * D = y + (ulong)(r1 + j) * args.n_out + r0;
            threadgroup float * C = temp_str + j * 64;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}
