// Chunked gated delta rule. F32 operands, accumulators, inverse and state.
// Q is unscaled; output uses the post-update state, as in the serial kernel.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct gdn_chunk_args { uint tokens, hv, hk, unused; };

// Share the Gram matrices across interleaved value heads. Store decay-weighted
// QK and (I + beta * decay * KK_lower)^-1 for each value head and chunk.
template<uint C>
[[max_total_threads_per_threadgroup(128)]]
kernel void gdn_chunk_precompute(
    constant gdn_chunk_args &a [[buffer(0)]],
    device float *q [[buffer(1)]], device float *k [[buffer(2)]],
    device const float *decay [[buffer(3)]], device const float *beta [[buffer(4)]],
    device float *p [[buffer(5)]], device float *inv [[buffer(6)]],
    device float *scales [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint NB = C / 16;
    threadgroup float lower[C * C];
    threadgroup float diag[NB * 16 * 16];
    threadgroup float logs[C];
    threadgroup uint zeros[C];
    const uint base = group.x * C, hk = group.y;
    constexpr auto d = tensor_ops::matmul2d_descriptor(C, C, 128, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d, execution_simdgroups<4>> mm;
    auto qt = tensor(q + base * a.hk * 128 + hk * 128, extents<int, 128, C>(), array<int, 2>{1, int(a.hk * 128)});
    auto kt = tensor(k + base * a.hk * 128 + hk * 128, extents<int, 128, C>(), array<int, 2>{1, int(a.hk * 128)});
    auto kk = mm.template get_destination_cooperative_tensor<decltype(kt), decltype(kt), float>();
    auto qk = mm.template get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    mm.run(kt, kt, kk);
    mm.run(qt, kt, qk);
    for (uint h = hk; h < a.hv; h += a.hk) {
        if (tid < C) {
            float x = decay[(base + tid) * a.hv + h];
            logs[tid] = simd_prefix_inclusive_sum(log(x > 0 ? x : 1.0f));
            zeros[tid] = simd_prefix_inclusive_sum(uint(x == 0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= 32 && tid < C) {
            logs[tid] += logs[31];
            zeros[tid] += zeros[31];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint off = (group.x * a.hv + h) * C * C;
        if (tid < C) {
            // Zero decays reset state. Counting zeros avoids log(0) and 0/0,
            // while preserving interactions entirely after the last reset.
            scales[(group.x * a.hv + h) * (2 * C) + tid] = zeros[tid] ? 0 : exp(logs[tid]);
            scales[(group.x * a.hv + h) * (2 * C) + C + tid] = zeros[C - 1] != zeros[tid] ? 0 : exp(logs[C - 1] - logs[tid]);
        }
        for (uint e = 0; e < kk.get_capacity(); ++e) if (kk.is_valid_element(e)) {
            auto ij = kk.get_multidimensional_index(e);
            const uint r = ij[1], c = ij[0];
            const float w = c <= r && zeros[r] == zeros[c] ? exp(logs[r] - logs[c]) : 0;
            lower[r * C + c] = c < r ? kk[e] * w * beta[(base + r) * a.hv + h] : 0;
            p[off + r * C + c] = qk[e] * w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint sg = tid / 32, lane = tid % 32;
        float col[16];
        if (sg < NB && lane < 16) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 16; ++r) {
                float value = float(lane == r);
                if (r > lane) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < r; ++j) value -= lower[(sg * 16 + r) * C + sg * 16 + j] * col[j];
                }
                col[r] = value;
            }
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 16; ++r) diag[sg * 256 + r * 16 + lane] = col[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One simdgroup per 16-column slice. Completed block rows remain F32
        // in registers. Inactive simdgroups at C=32 perform no memory accesses.
        simdgroup_float8x8 blocks[NB][2][2];
        #pragma clang loop unroll(full)
        for (uint row = 0; row < NB; ++row) {
            simdgroup_float8x8 rhs[2][2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; ++j) {
                    rhs[i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    blocks[row][i][j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }
            }
            if (row == sg) {
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 2; ++j) simdgroup_load(blocks[row][i][j], diag + row * 256 + i * 8 * 16 + j * 8, 16);
                }
            } else if (row > sg) {
                #pragma clang loop unroll(full)
                for (uint m = 0; m < row; ++m) if (m >= sg) {
                    #pragma clang loop unroll(full)
                    for (uint i = 0; i < 2; ++i) {
                        #pragma clang loop unroll(full)
                        for (uint kk = 0; kk < 2; ++kk) {
                            simdgroup_float8x8 aa;
                            simdgroup_load(aa, lower + (row * 16 + i * 8) * C + m * 16 + kk * 8, C);
                            aa.thread_elements()[0] = -aa.thread_elements()[0];
                            aa.thread_elements()[1] = -aa.thread_elements()[1];
                            #pragma clang loop unroll(full)
                            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(rhs[i][j], aa, blocks[m][kk][j], rhs[i][j]);
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    #pragma clang loop unroll(full)
                    for (uint kk = 0; kk < 2; ++kk) {
                        simdgroup_float8x8 dd;
                        simdgroup_load(dd, diag + row * 256 + i * 8 * 16 + kk * 8, 16);
                        #pragma clang loop unroll(full)
                        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(blocks[row][i][j], dd, rhs[kk][j], blocks[row][i][j]);
                    }
                }
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; ++j) if (sg < NB) simdgroup_store(blocks[row][i][j], inv + off + (row * 16 + i * 8) * C + sg * 16 + j * 8, C);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Each group owns a 32-value slice of one head. The transposed F32 state
// stays in threadgroup memory across internal chunks; only the final state
// is written to device memory. Neither input operands nor carry are narrowed.
template<uint C>
[[max_total_threads_per_threadgroup(128)]]
kernel void gdn_chunk_scan(
    constant gdn_chunk_args &a [[buffer(0)]],
    device float *q [[buffer(1)]], device float *k [[buffer(2)]],
    device float *v [[buffer(3)]], device const float *beta [[buffer(4)]],
    device float *p [[buffer(5)]], device float *inv [[buffer(6)]],
    device const float *scales [[buffer(7)]], device float *state [[buffer(8)]],
    device float *out [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float st[128 * 32];
    threadgroup float rv[C * 32];
    const uint h = group.x, vs = group.y * 32, hk = h % a.hk;
    for (uint e = tid; e < 4096; e += 128) st[(e % 128) * 32 + e / 128] = state[(h * 128 + vs + e / 128) * 128 + e % 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto proj_desc = tensor_ops::matmul2d_descriptor(C, 32, 128, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto solve_desc = tensor_ops::matmul2d_descriptor(C, 32, C, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto out_desc = tensor_ops::matmul2d_descriptor(C, 32, C, false, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto update_desc = tensor_ops::matmul2d_descriptor(128, 32, C, true, false, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<proj_desc, execution_simdgroups<4>> project;
    tensor_ops::matmul2d<solve_desc, execution_simdgroups<4>> solve;
    tensor_ops::matmul2d<out_desc, execution_simdgroups<4>> output;
    tensor_ops::matmul2d<update_desc, execution_simdgroups<4>> update;
    auto stt = tensor(st, extents<int, 32, 128>());
    auto rvt = tensor(rv, extents<int, 32, C>());
    for (uint base = 0; base < a.tokens; base += C) {
        const uint off = ((base / C) * a.hv + h) * C * C;
        device const float *s = scales + ((base / C) * a.hv + h) * (2 * C);
        auto qt = tensor(q + base * a.hk * 128 + hk * 128, extents<int, 128, C>(), array<int, 2>{1, int(a.hk * 128)});
        auto kt = tensor(k + base * a.hk * 128 + hk * 128, extents<int, 128, C>(), array<int, 2>{1, int(a.hk * 128)});
        auto resid = project.template get_destination_cooperative_tensor<decltype(kt), decltype(stt), float>();
        auto y = project.template get_destination_cooperative_tensor<decltype(qt), decltype(stt), float>();
        project.run(kt, stt, resid);
        project.run(qt, stt, y);
        for (uint e = 0; e < resid.get_capacity(); ++e) if (resid.is_valid_element(e)) {
            auto ij = resid.get_multidimensional_index(e);
            const uint t = ij[1], val = ij[0];
            resid[e] = beta[(base + t) * a.hv + h] * (v[((base + t) * a.hv + h) * 128 + vs + val] - s[t] * resid[e]);
        }
        resid.store(rvt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tt = tensor(inv + off, extents<int, C, C>());
        auto solved = solve.template get_destination_cooperative_tensor<decltype(tt), decltype(rvt), float>();
        solve.run(tt, rvt, solved);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        solved.store(rvt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < y.get_capacity(); ++e) if (y.is_valid_element(e)) y[e] *= s[y.get_multidimensional_index(e)[1]];
        auto pt = tensor(p + off, extents<int, C, C>());
        output.run(pt, rvt, y);
        auto ot = tensor(out + (base * a.hv + h) * 128 + vs, extents<int, 32, C>(), array<int, 2>{1, int(a.hv * 128)});
        y.store(ot);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < C * 32; e += 128) rv[e] *= s[C + e / 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto su = update.template get_destination_cooperative_tensor<decltype(kt), decltype(rvt), float>();
        update.run(kt, rvt, su);
        for (uint e = 0; e < su.get_capacity(); ++e) if (su.is_valid_element(e)) {
            auto ij = su.get_multidimensional_index(e);
            su[e] += s[C - 1] * st[ij[1] * 32 + ij[0]];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        su.store(stt);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = tid; e < 4096; e += 128) state[(h * 128 + vs + e / 128) * 128 + e % 128] = st[(e % 128) * 32 + e / 128];
}

template [[host_name("kernel_gdn_chunk_precompute32_f32")]] kernel decltype(gdn_chunk_precompute<32>) gdn_chunk_precompute<32>;
template [[host_name("kernel_gdn_chunk_precompute64_f32")]] kernel decltype(gdn_chunk_precompute<64>) gdn_chunk_precompute<64>;
template [[host_name("kernel_gdn_chunk_scan32_f32")]] kernel decltype(gdn_chunk_scan<32>) gdn_chunk_scan<32>;
template [[host_name("kernel_gdn_chunk_scan64_f32")]] kernel decltype(gdn_chunk_scan<64>) gdn_chunk_scan<64>;
