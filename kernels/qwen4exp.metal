// Qwen3.8-Flash-Next gated-residual elementwise kernels.

#include <metal_stdlib>
using namespace metal;

struct hc_norm_args {
    uint branch_count;
    uint hidden_size;
    float eps;
};

struct hc_low_args {
    uint count;
    float inverse_branches;
};

struct hc_branch_args {
    uint branch_count;
    uint hidden_size;
};

kernel void kernel_qwen4exp_hc_rms_norm_f32(
        constant hc_norm_args & args [[buffer(0)]],
        device const float * input [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * normalized [[buffer(3)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint branch [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        uint thread_count [[threads_per_threadgroup]]) {
    if (branch >= args.branch_count) return;
    const ulong base = (ulong)branch * args.hidden_size;
    float sum_square = 0.0f;
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const float value = input[base + hidden];
        sum_square += value * value;
    }
    sum_square = simd_sum(sum_square);
    if (lane == 0) partial[simdgroup] = sum_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_square = lane < (thread_count + 31u) / 32u ? partial[lane] : 0.0f;
    sum_square = simd_sum(sum_square);
    const float scale = rsqrt(sum_square / float(args.hidden_size) + args.eps);
    for (uint hidden = tid; hidden < args.hidden_size; hidden += thread_count) {
        const ulong index = base + hidden;
        normalized[index] = input[index] * scale * weight[index];
    }
}

kernel void kernel_qwen4exp_hc_low_silu_f32(
        constant hc_low_args & args [[buffer(0)]],
        device float * values [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
    if (index >= args.count) return;
    const float value = values[index] * args.inverse_branches;
    values[index] = value / (1.0f + exp(-value));
}

kernel void kernel_qwen4exp_hc_gated_mean_f32(
        constant hc_branch_args & args [[buffer(0)]],
        device const float * normalized [[buffer(1)]],
        device const float * raw_gate [[buffer(2)]],
        device float * mixed [[buffer(3)]],
        uint hidden [[thread_position_in_grid]]) {
    if (hidden >= args.hidden_size) return;
    float sum = 0.0f;
    for (uint branch = 0; branch < args.branch_count; ++branch) {
        const ulong index = (ulong)branch * args.hidden_size + hidden;
        const float gate = 1.0f / (1.0f + exp(-raw_gate[index]));
        sum += gate * normalized[index];
    }
    mixed[hidden] = sum / float(args.branch_count);
}

kernel void kernel_qwen4exp_hc_inject_f32(
        constant hc_branch_args & args [[buffer(0)]],
        device const float * block_output [[buffer(1)]],
        device const float * raw_injection [[buffer(2)]],
        device float * residual [[buffer(3)]],
        uint2 index [[thread_position_in_grid]]) {
    const uint hidden = index.x;
    const uint branch = index.y;
    if (hidden >= args.hidden_size || branch >= args.branch_count) return;
    const float value = raw_injection[branch] / float(args.branch_count);
    const float injection = 2.0f / (1.0f + exp(-value));
    residual[(ulong)branch * args.hidden_size + hidden] += block_output[hidden] * injection;
}
