#include <metal_stdlib>
using namespace metal;

struct qwen4exp_gdn_controls_args { uint n_in; uint heads; };

// Independent beta and alpha projections share one dispatch. Keep the
// incumbent beta SIMD-row and alpha cooperative-two-row reduction orders.
// The alpha group also publishes its decay, removing the dependent leaf.
kernel void kernel_qwen4exp_gdn_controls_f32(
        constant qwen4exp_gdn_controls_args & args [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device const float * bw      [[buffer(2)]],
        device const float * aw      [[buffer(3)]],
        device const float * dt      [[buffer(4)]],
        device const float * a_log   [[buffer(5)]],
        device float * beta         [[buffer(6)]],
        device float * alpha        [[buffer(7)]],
        device float * decay        [[buffer(8)]],
        threadgroup float * shmem   [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint alpha_groups = args.heads / 2;
    const uint nv = args.n_in / 4;
    device const float4 * xv = (device const float4 *)x;
    if (group >= alpha_groups) {
        const uint row = (group - alpha_groups) * 4 + sg;
        if (row >= args.heads) return;
        device const float4 * wv = (device const float4 *)(bw + row * args.n_in);
        float sum = 0.0f;
        for (uint i = lane; i < nv; i += 32) {
            const float4 a = wv[i], b = xv[i];
            sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
        sum = simd_sum(sum);
        if (lane == 0) beta[row] = 1.0f / (1.0f + exp(-sum));
        return;
    }
    const uint first = group * 2;
    float sum[2] = {0.0f, 0.0f};
    for (uint i = uint(sg) * 32 + lane; i < nv; i += 128) {
        const float4 b = xv[i];
        for (ushort row = 0; row < 2; ++row) {
            device const float4 * wv = (device const float4 *)(aw + (first + row) * args.n_in);
            const float4 a = wv[i];
            sum[row] += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
    }
    for (ushort row = 0; row < 2; ++row) {
        if (sg == 0) shmem[row * 32 + lane] = 0.0f;
        sum[row] = simd_sum(sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2; ++row) {
        if (lane == 0) shmem[row * 32 + sg] = sum[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort row = 0; row < 2; ++row) {
        const float total = simd_sum(shmem[row * 32 + lane]);
        if (sg == 0 && lane == 0) {
            const uint head = first + row;
            alpha[head] = total;
            const float v = total + dt[head];
            const float sp = v > 20.0f ? v : (v < -20.0f ? exp(v) : log(1.0f + exp(v)));
            decay[head] = exp(sp * a_log[head]);
        }
    }
}
