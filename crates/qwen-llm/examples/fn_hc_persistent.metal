#include <metal_stdlib>
using namespace metal;

struct fn_hc_persistent_args {
    uint task_count;
    uint branch_count;
    uint hidden_size;
    uint low_rank;
    float eps;
    uint prefetch_up;
};

inline void fn_hc_device_fence() {
    atomic_thread_fence(
        mem_flags::mem_device,
        memory_order_seq_cst,
        thread_scope::thread_scope_device);
}

inline void fn_hc_wait_flag(
        coherent(device) device atomic_uint *ready,
        uint index) {
    while (atomic_load_explicit(&ready[index], memory_order_relaxed) == 0u) {
    }
    fn_hc_device_fence();
}

inline void fn_hc_publish_flag(
        coherent(device) device atomic_uint *ready,
        uint index) {
    fn_hc_device_fence();
    atomic_store_explicit(&ready[index], 1u, memory_order_relaxed);
}

inline void fn_hc_wait_flags_without_fence(
        coherent(device) device atomic_uint *ready,
        uint begin,
        uint end) {
    for (uint index = begin; index < end; ++index) {
        while (atomic_load_explicit(&ready[index], memory_order_relaxed) == 0u) {
        }
    }
    fn_hc_device_fence();
}

__attribute__((noinline)) inline void fn_hc_norm_1024_order(
        coherent(device) device const float *input,
        device const float *weight,
        coherent(device) device float *normalized,
        threadgroup float *partial,
        uint branch,
        ushort simdgroup,
        ushort lane,
        uint tid,
        uint hidden_size,
        float eps) {
    // Reproduce the incumbent 1024-lane reduction with a 128-lane group.
    // Each physical SIMD group executes eight incumbent SIMD-group reductions
    // in order, preserving their original sum tree and final 32-way reduction.
    const ulong base = ulong(branch) * hidden_size;
    for (uint virtual_group = 0; virtual_group < 8; ++virtual_group) {
        const uint original_simdgroup = uint(simdgroup) * 8u + virtual_group;
        const uint original_tid = original_simdgroup * 32u + uint(lane);
        float sum_square = 0.0f;
        for (uint h = original_tid; h < hidden_size; h += 1024u) {
            const float value = input[base + h];
            sum_square += value * value;
        }
        sum_square = simd_sum(sum_square);
        if (lane == 0) partial[original_simdgroup] = sum_square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_square = lane < 32 ? partial[lane] : 0.0f;
    sum_square = simd_sum(sum_square);
    const float scale = rsqrt(sum_square / float(hidden_size) + eps);
    for (uint h = tid; h < hidden_size; h += 128u) {
        const ulong index = base + h;
        normalized[index] = input[index] * scale * weight[index];
    }
}

inline float fn_hc_dot8_persistent(uint4 u, float4 x0, float4 x1) {
    const float4 lo = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                             as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
    const float4 hi = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                             as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
    return dot(lo, x0) + dot(hi, x1);
}

__attribute__((noinline)) inline void fn_hc_down_silu_persistent(
        coherent(device) device const uint4 *weights,
        coherent(device) device const float4 *normalized,
        coherent(device) device float *low,
        threadgroup float *partial,
        uint row,
        ushort tid,
        ushort simdgroup,
        ushort lane) {
    coherent(device) device const uint4 *w = weights + ulong(row) * 1280u;
    float s0 = 0.0f;
    float s1 = 0.0f;
    for (uint c = tid; c < 1280u; c += 256u) {
        const uint d = c + 128u;
        s0 += fn_hc_dot8_persistent(w[c], normalized[2u * c], normalized[2u * c + 1u]);
        s1 += fn_hc_dot8_persistent(w[d], normalized[2u * d], normalized[2u * d + 1u]);
    }
    const float sum = simd_sum(s0 + s1);
    if (lane == 0) partial[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        volatile float raw = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        const float value = raw * 0.25f;
        low[row] = value / (1.0f + exp(-value));
    }
}

__attribute__((noinline)) inline void fn_hc_injection_persistent(
        coherent(device) device const float *inject_weights,
        coherent(device) device const float4 *normalized,
        coherent(device) device float *injection,
        threadgroup float *partial,
        uint branch,
        ushort tid,
        ushort simdgroup,
        ushort lane) {
    coherent(device) device const float *w = inject_weights + ulong(branch) * 10240u;
    coherent(device) device const float *x = (coherent(device) device const float *)normalized;
    float sums[8] = {0.0f};
#pragma clang loop unroll(disable)
    for (uint i = 0; i < 10; ++i) {
#pragma clang loop unroll(full)
        for (uint v = 0; v < 8; ++v) {
            const uint k = uint(tid) + v * 128u + i * 1024u;
            sums[v] += w[k] * x[k];
        }
    }
#pragma clang loop unroll(full)
    for (uint v = 0; v < 8; ++v) {
        const float sum = simd_sum(sums[v]);
        if (lane == 0) partial[v * 4u + simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        const float sum = simd_sum(partial[lane]);
        if (lane == 0) injection[branch] = sum;
    }
}

__attribute__((noinline)) inline void fn_hc_up_mix_bf16_global(
        coherent(device) device const ushort4 *weight,
        coherent(device) device const float4 *low,
        coherent(device) device const float *normalized,
        coherent(device) device float *raw_gate,
        coherent(device) device float *mixed,
        uint group,
        ushort simdgroup,
        ushort lane) {
    const uint hidden = group * 4u + simdgroup;
    const uint branch = lane / 8u;
    const uint sublane = lane % 8u;
    const uint row = branch * 2560u + hidden;
    coherent(device) device const ushort4 *w = weight + ulong(row) * 80u;
    float acc = 0.0f;
    for (uint i = sublane; i < 80u; i += 8u) {
        const ushort4 u = w[i];
        const float4 a = float4(as_type<float>(uint(u.x) << 16), as_type<float>(uint(u.y) << 16),
                                as_type<float>(uint(u.z) << 16), as_type<float>(uint(u.w) << 16));
        const float4 b = low[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    if (sublane == 0) raw_gate[row] = acc;
    const float value = (1.0f / (1.0f + exp(-acc))) * normalized[row];
    float sum = 0.0f;
    for (ushort b = 0; b < 4; ++b) {
        sum += simd_broadcast(value, ushort(b * 8));
    }
    if (lane == 0) mixed[hidden] = sum / 4.0f;
}

kernel void kernel_fn_hc_persistent(
        coherent(device) device atomic_uint *next_task [[buffer(0)]],
        coherent(device) device atomic_uint *ready [[buffer(1)]],
        coherent(device) device const float *hyper_input [[buffer(2)]],
        device const float *norm_weight [[buffer(3)]],
        coherent(device) device float *normalized [[buffer(4)]],
        coherent(device) device const uint4 *down_weight [[buffer(5)]],
        coherent(device) device float *low [[buffer(6)]],
        coherent(device) device const float *inject_weight [[buffer(7)]],
        coherent(device) device float *injection [[buffer(8)]],
        coherent(device) device const ushort4 *up_weight [[buffer(9)]],
        coherent(device) device float *raw_gate [[buffer(10)]],
        coherent(device) device float *mixed [[buffer(11)]],
        constant fn_hc_persistent_args &args [[buffer(12)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint task_shared;
    threadgroup float partial[32];
    coherent(device) device atomic_uint *down_completed = next_task + 1;
    if (tid == 0) {
        task_shared = atomic_fetch_add_explicit(next_task, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint task = task_shared;
    if (task >= args.task_count) return;

    if (task < 4u) {
        fn_hc_norm_1024_order(
            hyper_input, norm_weight, normalized, partial, task,
            simdgroup, lane, tid, args.hidden_size, args.eps);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) fn_hc_publish_flag(ready, task);
        return;
    }
    if (task == 4u) {
        if (tid == 0) {
            fn_hc_wait_flags_without_fence(ready, 0u, 4u);
            fn_hc_publish_flag(ready, 4u);
        }
        return;
    }
    if (task < 325u) {
        if (tid == 0) fn_hc_wait_flag(ready, 4u);
        threadgroup_barrier(mem_flags::mem_device);
        coherent(device) device const float4 *normalized4 =
            reinterpret_cast<coherent(device) device const float4 *>(normalized);
        fn_hc_down_silu_persistent(
            down_weight, normalized4, low, partial, task - 5u,
            ushort(tid), simdgroup, lane);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            fn_hc_device_fence();
            const uint completed = atomic_fetch_add_explicit(
                down_completed, 1u, memory_order_relaxed);
            fn_hc_device_fence();
            if (completed + 1u == args.low_rank) {
                fn_hc_publish_flag(ready, 329u);
            }
        }
        return;
    }
    if (task < 329u) {
        if (tid == 0) fn_hc_wait_flag(ready, 4u);
        threadgroup_barrier(mem_flags::mem_device);
        coherent(device) device const float4 *normalized4 =
            reinterpret_cast<coherent(device) device const float4 *>(normalized);
        fn_hc_injection_persistent(
            inject_weight, normalized4, injection, partial, task - 325u,
            ushort(tid), simdgroup, lane);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) fn_hc_publish_flag(ready, task);
        return;
    }
    const uint worker = task - 329u;
    coherent(device) device const float4 *low4 =
        reinterpret_cast<coherent(device) device const float4 *>(low);
    if (args.prefetch_up != 0u) {
        const uint branch = uint(lane) / 8u;
        const uint sublane = uint(lane) % 8u;
        const uint row = branch * 2560u + worker * 4u + uint(simdgroup);
        coherent(device) device const ushort4 *row_weight = up_weight + ulong(row) * 80u;
        thread volatile ushort4 prefetched[10];
        volatile thread uint prefetch_checksum = 0u;
        for (uint block = 0; block < 10u; ++block) {
            prefetched[block] = row_weight[sublane + block * 8u];
            prefetch_checksum ^= uint(prefetched[block].x)
                | (uint(prefetched[block].y) << 16);
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (tid == 0) fn_hc_wait_flag(ready, 329u);
        threadgroup_barrier(mem_flags::mem_device);
        if (prefetch_checksum == 0xffffffffu) {
            atomic_fetch_add_explicit(&ready[args.task_count - 1u], 0u, memory_order_relaxed);
        }
        fn_hc_up_mix_bf16_global(
            up_weight, low4, normalized, raw_gate, mixed, worker, simdgroup, lane);
        for (uint group = worker + (args.task_count - 329u); group < 640u; group += (args.task_count - 329u)) {
            fn_hc_up_mix_bf16_global(
                up_weight, low4, normalized, raw_gate, mixed, group, simdgroup, lane);
        }
    } else {
        if (tid == 0) fn_hc_wait_flag(ready, 329u);
        threadgroup_barrier(mem_flags::mem_device);
        for (uint group = worker; group < 640u; group += (args.task_count - 329u)) {
            fn_hc_up_mix_bf16_global(
                up_weight, low4, normalized, raw_gate, mixed, group, simdgroup, lane);
        }
    }
}
