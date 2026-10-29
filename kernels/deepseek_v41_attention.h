// Row geometry helpers shared by the attention kernels in
// deepseek_v41_attention.metal and the Metal 4 tensor-ops sink tile in
// tensor/deepseek_v41_attention_tensor.metal. Includers must
// `#include <metal_stdlib>` and `using namespace metal;` first.
//
//   geometry a: x = head_dim, y = window, z = index_topk, w = score stride
//   history  b: x = position of the tile's first query, y = ring base,
//               z = ring rows, w = owner ratio (0: no compressed rows)
#pragma once

inline uint4 attention_row_raw(uint4 geometry, uint4 history, uint query) {
    uint end = history.x + query + 1, count = min(end, geometry.y);
    return uint4(geometry.x, count, end - count - history.y, history.z);
}
inline uint attention_row_length(uint4 history, uint query) {
    return history.w ? (history.x + query + 1) / history.w : 0;
}

// Key i of a query row: its raw ring prefix (rr.y rows, ring offset rr.z,
// ring rows rr.w) followed by its selected compressed rows.
inline device const float *attention_row_key(uint4 raw, device const float *history,
    device const float *compressed, device const uint *ids, uint i) {
    return i < raw.y ? history + ((raw.z + i) % raw.w) * raw.x
        : compressed + ids[i - raw.y] * raw.x;
}

// Same rows and offsets from a pre-converted half mirror of the latent.
inline device const half *attention_row_key_half(uint4 raw, device const half *history,
    device const half *compressed, device const uint *ids, uint i) {
    return i < raw.y ? history + ((raw.z + i) % raw.w) * raw.x
        : compressed + ids[i - raw.y] * raw.x;
}
