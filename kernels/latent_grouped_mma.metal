// Native compressed grouped F32 MMA, patterned after the unchanged Q8
// grouped r2c16k64 kernel. One threadgroup owns [16 outputs, 128 tokens] for
// one group. The host admits complete tiles only: K % 64, M % 16, rows % 128,
// and K aligned to the native storage block (256 for the Q6_K instantiation).
// Bindings/ABI match GroupedProjectionArgs in metal/latent.rs. Strides are
// F32 elements; row_bytes is the compressed weight row stride in bytes.
//
// Scratch is exactly 16*64 floats (4096 bytes) of transient threadgroup
// memory. Weights remain compressed in device memory; neither weights nor
// activations pass through half operands. The shared native F32 decoder is
// identical to the scalar baseline. F32 MMA changes reduction order only:
// Exact retains F32 operand precision, not bitwise scalar reduction results.
// Each K64 block starts a fresh F32 MMA partial; its two elements per lane
// are added to the running F32 total after the block. This bounds each MMA
// accumulation chain to eight K8 steps instead of carrying it across all K,
// reducing cancellation/rounding exposure without changing the tile width,
// operand precision, decoder, or threadgroup scratch. The extra partials use
// 16 F32 register values per lane; this is not compensated or F64 summation.
#include <metal_stdlib>
using namespace metal;
#include "native_dequant_f32.h"

struct latent_grouped_mma_args {
    uint k;
    uint m;
    uint groups;
    uint rows;
    uint row_bytes;
    uint input_stride;
    uint output_stride;
};
static_assert(sizeof(latent_grouped_mma_args) == 28);

template <ushort BYTES, ushort NL,
          void (*DEQ)(device const uchar *, short, thread float4x4 &)>
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_latent_grouped_native_f32_r2c16k64(
        constant latent_grouped_mma_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        threadgroup float * shmem [[threadgroup(0)]],
        uint3 tile [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint group = tile.z;
    if (group >= args.groups) return; // Uniform across the entire threadgroup.
    const uint m0 = tile.y * 16u;
    const uint n0 = tile.x * 128u + uint(simd) * 32u;
    device const uchar * group_weights = weights
        + ulong(group) * args.m * args.row_bytes;

    simdgroup_float8x8 acc[2][4];
    QT_FOR_UNROLL (short mt = 0; mt < 2; ++mt) {
        QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
            acc[mt][nt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < args.k; k0 += 64u) {
        if (simd == 0) {
            QT_FOR_UNROLL (short chunk = 0; chunk < 2; ++chunk) {
                const uint cell = uint(lane) + 32u * uint(chunk);
                const uint row = cell / 4u;
                const uint kchunk = cell % 4u;
                const uint kbase = k0 + 16u * kchunk;
                device const uchar * block = group_weights
                    + ulong(m0 + row) * args.row_bytes
                    + ulong(kbase / (16u * NL)) * BYTES;
                float4x4 decoded;
                DEQ(block, short((kbase / 16u) % NL), decoded);
                QT_FOR_UNROLL (short i = 0; i < 16; ++i) {
                    shmem[row * 64u + kchunk * 16u + uint(i)] = decoded[i / 4][i % 4];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 partial[2][4];
        QT_FOR_UNROLL (short mt = 0; mt < 2; ++mt) {
            QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
                partial[mt][nt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
        }
        QT_FOR_UNROLL (short kt = 0; kt < 8; ++kt) {
            simdgroup_float8x8 activation[4];
            QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
                simdgroup_load(
                    activation[nt],
                    input + ulong(n0 + uint(nt) * 8u) * args.input_stride
                        + ulong(group) * args.k + k0 + uint(kt) * 8u,
                    args.input_stride, ulong2(0, 0), true);
            }
            QT_FOR_UNROLL (short mt = 0; mt < 2; ++mt) {
                simdgroup_float8x8 weight;
                simdgroup_load(weight, shmem + uint(mt) * 8u * 64u + uint(kt) * 8u, 64);
                QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
                    simdgroup_multiply_accumulate(partial[mt][nt], weight, activation[nt], partial[mt][nt]);
                }
            }
        }
        QT_FOR_UNROLL (short mt = 0; mt < 2; ++mt) {
            QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
                QT_FOR_UNROLL (short element = 0; element < 2; ++element) {
                    acc[mt][nt].thread_elements()[element] += partial[mt][nt].thread_elements()[element];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    QT_FOR_UNROLL (short mt = 0; mt < 2; ++mt) {
        QT_FOR_UNROLL (short nt = 0; nt < 4; ++nt) {
            simdgroup_store(
                acc[mt][nt],
                output + ulong(n0 + uint(nt) * 8u) * args.output_stride
                    + ulong(group) * args.m + m0 + uint(mt) * 8u,
                args.output_stride, ulong2(0, 0), true);
        }
    }
}

// Reusable storage-block/decode template; Q6_K is the first instantiated target.
// Other native formats retain their existing scalar dispatches.
typedef decltype(kernel_latent_grouped_native_f32_r2c16k64<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>) latent_grouped_mma_t;
template [[host_name("kernel_latent_grouped_q6_K_f32_r2c16k64")]] kernel latent_grouped_mma_t kernel_latent_grouped_native_f32_r2c16k64<QT_Q6_K_BYTES, QT_Q6_K_NL, latent_dequantize_q6_K>;
