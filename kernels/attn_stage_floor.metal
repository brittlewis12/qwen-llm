#include <metal_stdlib>
using namespace metal;

constant constexpr uint ASF_HD = 256;
constant constexpr uint ASF_C = 32;
constant constexpr uint ASF_GROUPS = 16;
constant constexpr uint ASF_ROW_BYTES = 288;

struct attn_stage_floor_args {
    uint n_pos;
    uint n_kv_heads;
    uint n_partitions;
    uint rows_per_partition;
};

inline void stage_g16_tile(
        device const uchar *cache,
        constant attn_stage_floor_args &args,
        threadgroup half *tile,
        uint kvh,
        uint row_start,
        ushort row_count,
        ushort tid) {
    const uint group_count = uint(row_count) * ASF_GROUPS;
    for (uint index = tid; index < group_count; index += 256) {
        const uint row = row_start + index / ASF_GROUPS;
        const uint group = index % ASF_GROUPS;
        device const uchar *src =
            cache + ((ulong)row * args.n_kv_heads + kvh) * ASF_ROW_BYTES;
        const float scale = float(((device const half *)src)[group]);
        device const char *payload = (device const char *)(src + 32 + group * 16);
        threadgroup half *dst = tile + (index / ASF_GROUPS) * ASF_HD + group * 16;
        for (ushort value = 0; value < 16; ++value) {
            dst[value] = half(scale * float(payload[value]));
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_attn_stage_floor_g16(
        constant attn_stage_floor_args &args [[buffer(0)]],
        device const uchar *k_cache [[buffer(1)]],
        device const uchar *v_cache [[buffer(2)]],
        device float *checksum [[buffer(3)]],
        threadgroup half *tile [[threadgroup(0)]],
        uint3 tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint kvh = tg.x;
    const uint partition = tg.z;
    if (kvh >= args.n_kv_heads || partition >= args.n_partitions) return;

    const uint begin = partition * args.rows_per_partition;
    const uint end = min(begin + args.rows_per_partition, args.n_pos);
    float sum = 0.0f;
    for (uint row = begin; row < end; row += ASF_C) {
        const ushort count = ushort(min(ASF_C, end - row));
        stage_g16_tile(k_cache, args, tile, kvh, row, count, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort local = 0; local < count; ++local) {
            sum += float(tile[uint(local) * ASF_HD + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        stage_g16_tile(v_cache, args, tile, kvh, row, count, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort local = 0; local < count; ++local) {
            sum += float(tile[uint(local) * ASF_HD + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong output = ((ulong)kvh * args.n_partitions + partition) * ASF_HD + tid;
    checksum[output] = sum;
}
