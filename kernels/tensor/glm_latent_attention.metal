#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
// Shared-latent attention: 64 heads, width 512, selected K=V cache rows.
// a = (queries, slots, score stride, query offset), b = (cache capacity,64,0,0).
// Cache is native F16; Q/P stage as Half. Accumulation and softmax stay F32.
// QK: one group per 32-key panel/query, all heads sharing the gathered keys.
kernel void kernel_glm_latent_score_tensor(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant float &scale [[buffer(2)]], device const float *q [[buffer(3)]],
    device const half *cache [[buffer(4)]], device const int *ids [[buffer(5)]],
    device const int *counts [[buffer(6)]], device const int *visible [[buffer(7)]],
    device float *scores [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half qb[64 * 128];
    threadgroup half kb[32 * 128];
    threadgroup float sb[64 * 32];
    const uint r = group.y, k0 = group.x * 32;
    if (r >= a.x || k0 >= a.y)
        return;
    const uint count = min(uint(max(counts[r], 0)), a.y), vis = min(uint(max(visible[r], 0)), b.x);
    constexpr auto d0 = tensor_ops::matmul2d_descriptor(
        64, 32, 128, false, true, false, tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto d1 =
        tensor_ops::matmul2d_descriptor(64, 32, 128, false, true, false,
                                        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d0, execution_simdgroups<4>> op0;
    tensor_ops::matmul2d<d1, execution_simdgroups<4>> op1;
    using Q = tensor<threadgroup half, extents<int, 128, 64>, tensor_inline>;
    using K = tensor<threadgroup half, extents<int, 128, 32>, tensor_inline>;
    using S = tensor<threadgroup float, extents<int, 32, 64>, tensor_inline>;
    Q qt(qb, extents<int, 128, 64>());
    K kt(kb, extents<int, 128, 32>());
    S st(sb, extents<int, 32, 64>());
    auto acc = op0.get_destination_cooperative_tensor<Q, K, float>();
    for (uint dc = 0; dc < 4; ++dc) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 64 * 128 / 4; e += 128) {
            uint h = e / 32, c = e % 32;
            ((threadgroup half4 *)qb)[e] = half4(
                ((device const float4 *)(q + ulong(a.w + r) * 64 * 512 + h * 512 + dc * 128))[c]);
        }
        for (uint e = tid; e < 32 * 128 / 4; e += 128) {
            uint key = e / 32, c = e % 32, i = k0 + key;
            int id = i < count ? ids[ulong(r) * a.y + i] : -1;
            half4 v = half4(0);
            if (id >= 0 && uint(id) < vis)
                v = ((device const half4 *)(cache + ulong(id) * 512 + dc * 128))[c];
            ((threadgroup half4 *)kb)[e] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (dc == 0)
            op0.run(qt, kt, acc);
        else
            op1.run(qt, kt, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc.store(st);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64 * 32; e += 128) {
        uint h = e / 32, i = k0 + e % 32;
        if (i < a.y) {
            int id = i < count ? ids[ulong(r) * a.y + i] : -1;
            scores[(ulong(r) * 64 + h) * a.z + i] =
                (id >= 0 && uint(id) < vis) ? sb[e] * scale : -1.0e30f;
        }
    }
}
kernel void kernel_glm_latent_softmax(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const int *ids [[buffer(5)]], device const int *counts [[buffer(6)]],
    device const int *visible [[buffer(7)]], device float *scores [[buffer(8)]],
    device const float *sinks [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, r = group.y;
    device float *s = scores + (ulong(r) * 64 + h) * a.z;
    uint count = min(uint(max(counts[r], 0)), a.y), vis = min(uint(max(visible[r], 0)), b.x);
    float m = sinks[h];
    for (uint i = lane; i < count; i += 32) {
        int id = ids[ulong(r) * a.y + i];
        if (id >= 0 && uint(id) < vis)
            m = max(m, s[i]);
    }
    m = simd_max(m);
    float sum = lane == 0 ? exp(sinks[h] - m) : 0.0f;
    for (uint i = lane; i < a.y; i += 32) {
        int id = i < count ? ids[ulong(r) * a.y + i] : -1;
        float p = id >= 0 && uint(id) < vis ? exp(s[i] - m) : 0.0f;
        s[i] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    for (uint i = lane; i < a.y; i += 32)
        s[i] /= sum;
}
kernel void kernel_glm_latent_pv_tensor(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const half *cache [[buffer(4)]], device const int *ids [[buffer(5)]],
    device const int *counts [[buffer(6)]], device const int *visible [[buffer(7)]],
    device const float *scores [[buffer(8)]], device float *out [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half pb[64 * 32];
    threadgroup half vb[32 * 64];
    uint dc = group.x, r = group.y;
    uint count = min(uint(max(counts[r], 0)), a.y), vis = min(uint(max(visible[r], 0)), b.x);
    constexpr auto d =
        tensor_ops::matmul2d_descriptor(64, 64, 32, false, false, false,
                                        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d, execution_simdgroups<4>> op;
    using P = tensor<threadgroup half, extents<int, 32, 64>, tensor_inline>;
    using V = tensor<threadgroup half, extents<int, 64, 32>, tensor_inline>;
    P pt(pb, extents<int, 32, 64>());
    V vt(vb, extents<int, 64, 32>());
    auto acc = op.get_destination_cooperative_tensor<P, V, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i)
        acc[i] = 0;
    for (uint k0 = 0; k0 < count; k0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 32 * 64 / 4; e += 128) {
            uint key = e / 16, c = e % 16, i = k0 + key;
            int id = i < count ? ids[ulong(r) * a.y + i] : -1;
            half4 v = half4(0);
            if (id >= 0 && uint(id) < vis)
                v = ((device const half4 *)(cache + ulong(id) * 512 + dc * 64))[c];
            ((threadgroup half4 *)vb)[e] = v;
        }
        for (uint e = tid; e < 64 * 32; e += 128) {
            uint h = e / 32, i = k0 + e % 32;
            pb[e] = half(i < count ? scores[(ulong(r) * 64 + h) * a.z + i] : 0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(pt, vt, acc);
    }
    tensor<device float, extents<int, 64, 64>, tensor_inline> ot(
        out + ulong(a.w + r) * 64 * 512 + dc * 64, extents<int, 64, 64>(), array<int, 2>{1, 512});
    acc.store(ot);
}

// Device F32 queries remove their staging copy; gathered cache stays exact F32.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_glm_latent_score_tensor_f32_direct(
        constant uint4 &a [[buffer(0)]],constant uint4 &b [[buffer(1)]],constant float &scale [[buffer(2)]],
        device const float *q [[buffer(3)]],device const half *cache [[buffer(4)]],device const int *ids [[buffer(5)]],
        device const int *counts [[buffer(6)]],device const int *visible [[buffer(7)]],device float *scores [[buffer(8)]],
        uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
#pragma METAL fp math_mode(safe)
    const uint r=group.y,k0=group.x*32,count=min(uint(max(counts[r],0)),a.y),vis=min(uint(max(visible[r],0)),b.x);
    threadgroup float kb[32*128],sb[64*32];
    constexpr auto d0=tensor_ops::matmul2d_descriptor(64,32,128,false,true,false,tensor_ops::matmul2d_descriptor::mode::multiply);
    constexpr auto d1=tensor_ops::matmul2d_descriptor(64,32,128,false,true,false,tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d0,execution_simdgroups<4>> op0;
    tensor_ops::matmul2d<d1,execution_simdgroups<4>> op1;
    using Q=tensor<device float,extents<int,128,64>,tensor_inline>;
    using K=tensor<threadgroup float,extents<int,128,32>,tensor_inline>;
    K kt(kb,extents<int,128,32>());
    auto acc=op0.get_destination_cooperative_tensor<Q,K,float>();
    for(uint dc=0;dc<4;++dc){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint e=tid;e<32*128/4;e+=128){uint i=k0+e/32,c=e%32;int id=i<count?ids[ulong(r)*a.y+i]:-1;
            float4 v=0;if(id>=0&&uint(id)<vis)v=float4(((device const half4*)(cache+ulong(id)*512+dc*128))[c]);
            ((threadgroup float4*)kb)[e]=v;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Q qt((device float*)q+ulong(a.w+r)*64*512+dc*128,extents<int,128,64>(),array<int,2>{1,512});
        if(dc==0)op0.run(qt,kt,acc);else op1.run(qt,kt,acc);
    }
    tensor<threadgroup float,extents<int,32,64>,tensor_inline> st(sb,extents<int,32,64>());
    threadgroup_barrier(mem_flags::mem_threadgroup);acc.store(st);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint e=tid;e<64*32;e+=128){uint h=e/32,i=k0+e%32;if(i<a.y){int id=i<count?ids[ulong(r)*a.y+i]:-1;
        scores[(ulong(r)*64+h)*a.z+i]=(id>=0&&uint(id)<vis)?sb[e]*scale:-1.0e30f;}}
}

template <int V,int R>
inline void glm_pv_f32_direct(constant uint4 &a,constant uint4 &b,device const half *cache,
        device const int *ids,device const int *counts,device const int *visible,
        device const float *scores,device float *out,uint2 group,uint tid,threadgroup float *vb){
#pragma METAL fp math_mode(safe)
    const uint dc=group.x,r=group.y,count=min(uint(max(counts[r],0)),a.y),vis=min(uint(max(visible[r],0)),b.x);
    constexpr auto d=tensor_ops::matmul2d_descriptor(64,V,R,false,false,false,tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> op;
    using P=tensor<device float,extents<int,R,64>,tensor_inline>;
    using K=tensor<threadgroup float,extents<int,V,R>,tensor_inline>;
    K vt(vb,extents<int,V,R>());
    auto acc=op.template get_destination_cooperative_tensor<P,K,float>();
    for(ushort i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint k0=0;k0<count;k0+=R){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint e=tid;e<R*V/4;e+=128){uint i=k0+e/(V/4),c=e%(V/4);int id=i<count?ids[ulong(r)*a.y+i]:-1;
            float4 v=0;if(id>=0&&uint(id)<vis)v=float4(((device const half4*)(cache+ulong(id)*512+dc*V))[c]);
            ((threadgroup float4*)vb)[e]=v;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        P pt((device float*)scores+(ulong(r)*64)*a.z+k0,extents<int,R,64>(),array<int,2>{1,int(a.z)});
        op.run(pt,vt,acc);
    }
    tensor<device float,extents<int,V,64>,tensor_inline> ot(out+ulong(a.w+r)*64*512+dc*V,extents<int,V,64>(),array<int,2>{1,512});
    acc.store(ot);
}
#define GLM_PV_F32_DIRECT(NAME,V,R) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME(constant uint4 &a [[buffer(0)]],constant uint4 &b [[buffer(1)]],device const half *cache [[buffer(4)]], \
        device const int *ids [[buffer(5)]],device const int *counts [[buffer(6)]],device const int *visible [[buffer(7)]], \
        device const float *scores [[buffer(8)]],device float *out [[buffer(9)]],uint2 group [[threadgroup_position_in_grid]], \
        uint tid [[thread_index_in_threadgroup]]){threadgroup float vb[R*V]; \
    glm_pv_f32_direct<V,R>(a,b,cache,ids,counts,visible,scores,out,group,tid,vb);}
GLM_PV_F32_DIRECT(kernel_glm_latent_pv_tensor_f32_direct_64,64,128)
#undef GLM_PV_F32_DIRECT

kernel void kernel_glm_latent_softmax_f32(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    device const int *ids [[buffer(5)]], device const int *counts [[buffer(6)]],
    device const int *visible [[buffer(7)]], device float *scores [[buffer(8)]],
    device const float *sinks [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, r = group.y;
    device float *s = scores + (ulong(r) * 64 + h) * a.z;
    uint count = min(uint(max(counts[r], 0)), a.y), vis = min(uint(max(visible[r], 0)), b.x);
    float m = sinks[h];
    for (uint i = lane; i < count; i += 32) {
        int id = ids[ulong(r) * a.y + i];
        if (id >= 0 && uint(id) < vis)
            m = max(m, s[i]);
    }
    m = simd_max(m);
    float sum = lane == 0 ? exp(sinks[h] - m) : 0.0f;
    for (uint i = lane; i < a.z; i += 32) {
        int id = i < count ? ids[ulong(r) * a.y + i] : -1;
        float p = id >= 0 && uint(id) < vis ? exp(s[i] - m) : 0.0f;
        s[i] = p;
        sum += p;
    }
    sum = simd_sum(sum);
    for (uint i = lane; i < a.z; i += 32)
        s[i] /= sum;
}
