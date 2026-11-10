#include <metal_stdlib>
using namespace metal;

struct qwen4exp_gdn_batch_args { uint n_in; uint heads; float eps; };

// Member-axis GDN controls: the same beta/alpha projections and decay epilogue
// as kernel_qwen4exp_gdn_controls_f32, with one threadgroup row per independent
// sequence. Shared weights are bound once; the member axis selects the private
// input and output vectors.
kernel void kernel_qwen4exp_gdn_controls_members_f32(
        constant qwen4exp_gdn_batch_args & args [[buffer(0)]],
        device const float * x0        [[buffer(1)]],
        device const float * x1        [[buffer(2)]],
        device const float * x2        [[buffer(3)]],
        device const float * x3        [[buffer(4)]],
        device const float * bw        [[buffer(5)]],
        device const float * aw        [[buffer(6)]],
        device const float * dt        [[buffer(7)]],
        device const float * a_log     [[buffer(8)]],
        device float * beta0           [[buffer(9)]],
        device float * beta1           [[buffer(10)]],
        device float * beta2           [[buffer(11)]],
        device float * beta3           [[buffer(12)]],
        device float * alpha0          [[buffer(13)]],
        device float * alpha1          [[buffer(14)]],
        device float * alpha2          [[buffer(15)]],
        device float * alpha3          [[buffer(16)]],
        device float * decay0          [[buffer(17)]],
        device float * decay1          [[buffer(18)]],
        device float * decay2          [[buffer(19)]],
        device float * decay3          [[buffer(20)]],
        threadgroup float * shmem      [[threadgroup(0)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint member = tg.z;
    const uint group = tg.x;
    device const float * x = member == 0 ? x0 : member == 1 ? x1 : member == 2 ? x2 : x3;
    device float * beta = member == 0 ? beta0 : member == 1 ? beta1 : member == 2 ? beta2 : beta3;
    device float * alpha = member == 0 ? alpha0 : member == 1 ? alpha1 : member == 2 ? alpha2 : alpha3;
    device float * decay = member == 0 ? decay0 : member == 1 ? decay1 : member == 2 ? decay2 : decay3;

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

// Member-axis causal-conv/SiLU and Q/K L2 normalization: the prep half of
// kernel_qwen4exp_gdn_controls_prep_f32, with one threadgroup row per member.
kernel void kernel_qwen4exp_gdn_prep_members_f32(
        constant qwen4exp_gdn_batch_args & args [[buffer(0)]],
        device const float * qkv0       [[buffer(1)]],
        device const float * qkv1       [[buffer(2)]],
        device const float * qkv2       [[buffer(3)]],
        device const float * qkv3       [[buffer(4)]],
        device float * conv_state0      [[buffer(5)]],
        device float * conv_state1      [[buffer(6)]],
        device float * conv_state2      [[buffer(7)]],
        device float * conv_state3      [[buffer(8)]],
        device const float * conv_w     [[buffer(9)]],
        device float * convolved0       [[buffer(10)]],
        device float * convolved1       [[buffer(11)]],
        device float * convolved2       [[buffer(12)]],
        device float * convolved3       [[buffer(13)]],
        device float * qnorm0           [[buffer(14)]],
        device float * qnorm1           [[buffer(15)]],
        device float * qnorm2           [[buffer(16)]],
        device float * qnorm3           [[buffer(17)]],
        device float * knorm0           [[buffer(18)]],
        device float * knorm1           [[buffer(19)]],
        device float * knorm2           [[buffer(20)]],
        device float * knorm3           [[buffer(21)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint member = tg.z;
    const uint group = tg.x;
    device const float * qkv = member == 0 ? qkv0 : member == 1 ? qkv1 : member == 2 ? qkv2 : qkv3;
    device float * conv_state = member == 0 ? conv_state0 : member == 1 ? conv_state1 : member == 2 ? conv_state2 : conv_state3;
    device float * convolved = member == 0 ? convolved0 : member == 1 ? convolved1 : member == 2 ? convolved2 : convolved3;
    device float * qnorm = member == 0 ? qnorm0 : member == 1 ? qnorm1 : member == 2 ? qnorm2 : qnorm3;
    device float * knorm = member == 0 ? knorm0 : member == 1 ? knorm1 : member == 2 ? knorm2 : knorm3;

    const uint head = group * 4 + sg;
    const uint width = 10240;
    float values[4];
    float ss = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        const uint c = head * 128 + lane + 32 * j;
        const float t[4] = {conv_state[c], conv_state[width+c], conv_state[2*width+c], qkv[c]};
        float sum = 0.0f;
        for (uint k = 0; k < 4; ++k) sum += conv_w[c*4+k] * t[k];
        const float v = sum / (1.0f + exp(-sum));
        convolved[c] = v;
        values[j] = v;
        ss += v * v;
        for (uint k = 0; k < 3; ++k) conv_state[k*width+c] = t[k+1];
    }
    if (head < 32) {
        ss = simd_sum(ss);
        const float scale = 1.0f / max(sqrt(ss), args.eps);
        device float * out = head < 16 ? qnorm + head * 128 : knorm + (head-16) * 128;
        for (uint j = 0; j < 4; ++j) out[lane+32*j] = values[j] * scale;
    }
}
