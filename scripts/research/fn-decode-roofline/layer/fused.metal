#include <metal_stdlib>
using namespace metal;

// Each group computes a complete head's recurrence and normalization. A
// partition owns disjoint output/state rows; all groups read immutable old
// state. Repetition trades ALU/cache traffic for occupancy, without polling.
struct gdn_fused_args { uint partitions; float eps; };
kernel void layer_gdn_fused(
        constant gdn_fused_args &a [[buffer(0)]],
        device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]], device const float *decay [[buffer(4)]],
        device const float *beta [[buffer(5)]], device const float *old_state [[buffer(6)]],
        device float *new_state [[buffer(7)]], device const float *gate [[buffer(8)]],
        device const float *weight [[buffer(9)]], device float *out [[buffer(10)]],
        uint2 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float values[128];
    const uint h=group.y, hk=h%16, lo=group.x*(128/a.partitions), hi=lo+128/a.partitions;
    float kr[4],qr[4];
    for(uint j=0;j<4;++j){kr[j]=k[hk*128+lane*4+j];qr[j]=q[hk*128+lane*4+j];}
    for(uint row=sg;row<128;row+=4){
        const ulong base=ulong(h)*16384+row*128+lane*4;
        float sr[4];for(uint j=0;j<4;++j)sr[j]=old_state[base+j]*decay[h];
        float sk=0;for(uint j=0;j<4;++j)sk+=sr[j]*kr[j];sk=simd_sum(sk);
        const float delta=(v[h*128+row]-sk)*beta[h];
        for(uint j=0;j<4;++j)sr[j]+=delta*kr[j];
        float o=0;for(uint j=0;j<4;++j)o+=sr[j]*qr[j];o=simd_sum(o);
        if(lane==0)values[row]=o;
        if(row>=lo&&row<hi)for(uint j=0;j<4;++j)new_state[base+j]=sr[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss=0;for(uint j=lane;j<128;j+=32)ss+=values[j]*values[j];
    ss=simd_sum(ss);const float scale=1.f/sqrt(ss/128.f+a.eps);
    for(uint j=uint(sg)*32+lane;j<128;j+=128){
        if(j>=lo&&j<hi){const float z=values[j]*scale*weight[j];out[h*128+j]=z*(1.f/(1.f+exp(-gate[h*128+j])));}
    }
}

// Flash decode: QK, block softmax and PV share threadgroup memory. The only
// cross-group boundary is a following ordinary merge dispatch.
struct flash_args { uint count; uint capacity; uint splits; uint keys; float scale; };
kernel void layer_qsa_flash(
        constant flash_args &a [[buffer(0)]], device const float *q [[buffer(1)]],
        device const half *k [[buffer(2)]], device const half *v [[buffer(3)]],
        device const int *ids [[buffer(4)]], device float *partials [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float logits[4*256];
    const uint head=group.y*4+sg, kv=head/12, start=group.x*a.keys;
    const uint n=min(a.keys,a.count-min(a.count,start));
    float query[8];for(uint d=0;d<8;++d)query[d]=q[head*256+lane+d*32];
    for(uint j=0;j<n;++j){
        const int pos=ids[start+j];float dot=0;
        if(pos>=0&&uint(pos)<a.capacity){
            const ulong base=(ulong(pos)*2+kv)*256+lane;
            for(uint d=0;d<8;++d)dot+=query[d]*float(k[base+d*32]);
            dot=simd_sum(dot)*a.scale;
        }else{dot=-INFINITY;}
        if(lane==0)logits[sg*256+j]=dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY;for(uint j=lane;j<n;j+=32)m=max(m,logits[sg*256+j]);m=simd_max(m);
    float sum=0;for(uint j=lane;j<n;j+=32){float z=m==-INFINITY?0.f:exp(logits[sg*256+j]-m);logits[sg*256+j]=z;sum+=z;}
    sum=simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float num[8]={};
    for(uint j=0;j<n;++j){
        const int pos=ids[start+j];if(pos<0||uint(pos)>=a.capacity)continue;
        const ulong base=(ulong(pos)*2+kv)*256+lane;const float mass=logits[sg*256+j];
        for(uint d=0;d<8;++d)num[d]+=float(v[base+d*32])*mass;
    }
    const ulong base=(ulong(head)*a.splits+group.x)*258;
    if(lane==0){partials[base]=m;partials[base+1]=sum;}
    for(uint d=0;d<8;++d)partials[base+2+lane+d*32]=num[d];
}
kernel void layer_qsa_flash_merge(
        constant flash_args &a [[buffer(0)]], device const float *partials [[buffer(1)]],
        device const float *gate [[buffer(2)]], device float *out [[buffer(3)]],
        uint head [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    device const float *p=partials+ulong(head)*a.splits*258;
    float m=-INFINITY;for(uint i=0;i<a.splits;++i)m=max(m,p[i*258]);
    float denom=0,num[8]={};
    for(uint i=0;i<a.splits;++i){
        const float f=m==-INFINITY?0.f:exp(p[i*258]-m);denom+=f*p[i*258+1];
        for(uint d=0;d<8;++d)num[d]+=p[i*258+2+lane+d*32]*f;
    }
    for(uint d=0;d<8;++d){const uint j=head*256+lane+d*32;out[j]=denom>0?num[d]/denom*(1.f/(1.f+exp(-gate[j]))):0.f;}
}

// Routing, shared-scale accumulation and HC injection keep each scalar on
// chip. The expert and shared paths retain their dispatch completion boundary.
kernel void layer_ffn_epilogue(
        device const float *experts [[buffer(0)]], device const float *weights [[buffer(1)]],
        device const float *shared [[buffer(2)]], device const float *scale [[buffer(3)]],
        device const float *raw_inject [[buffer(4)]], device float *residual [[buffer(5)]],
        uint2 j [[thread_position_in_grid]]) {
    if(j.x>=2560||j.y>=4)return;
    float sum=0;for(uint e=0;e<10;++e)sum+=weights[e]*experts[e*2560+j.x];
    sum+=scale[0]*shared[j.x];
    const float injection=2.f/(1.f+exp(-raw_inject[j.y]/4.f));
    residual[j.y*2560+j.x]+=sum*injection;
}

struct mat_vec_args { uint n_in; uint n_out; };
struct block_iq4_nl_local { half d; uchar qs[16]; };
constant float iq4nl_values[16]={-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};
kernel void layer_shared_down(
        device const block_iq4_nl_local * weight [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const mat_vec_args args={640,2560};
    threadgroup float lut[32],x[640];
    const uint tid=uint(sgitg)*32+tiisg;
    for(uint i=tid;i<640;i+=64)x[i]=(gate[i]/(1.f+exp(-gate[i])))*up[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short NR0 = 2;
    const short NSG = 2;
    const uint first_row = (tgpig * NSG + uint(sgitg)) * NR0;
    if (first_row >= args.n_out) return;

    if(sgitg==0)lut[tiisg] = iq4nl_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nb = args.n_in / 32u;
    const short ix = tiisg / 2;
    const short it = tiisg % 2;
    threadgroup const float * yb = x + uint(ix) * 32u + uint(it) * 8u;

    float sumf[2] = {0.0f, 0.0f};
    uint aux32[2];
    thread const uchar * q8 = (thread const uchar *)aux32;

    for (uint ib = uint(ix); ib < nb; ib += 16u) {
        threadgroup const float4 * y4 = (threadgroup const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;
            device const block_iq4_nl_local & b = weight[out_row * nb + ib];
            device const ushort * q4 = (device const ushort *)(b.qs + 8 * it);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            aux32[0] = uint(q4[0]) | (uint(q4[1]) << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0fu;
            aux32[0] &= 0x0f0f0f0fu;
            const float4 qf10 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf20 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl0 * qf10;
            acc2 += yl1 * qf20;

            aux32[0] = uint(q4[2]) | (uint(q4[3]) << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0fu;
            aux32[0] &= 0x0f0f0f0fu;
            const float4 qf11 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf21 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl2 * qf11;
            acc2 += yl3 * qf21;
            acc1 += acc2;

            sumf[row] += float(b.d) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 16u * 32u;
    }

    for (short row = 0; row < NR0; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) y[out_row] = total;
    }
}

kernel void roof4_read(device const float4*x[[buffer(0)]],device float*y[[buffer(1)]],uint g[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]],ushort l[[thread_index_in_simdgroup]],ushort s[[simdgroup_index_in_threadgroup]]){float4 z=0.0f;for(uint i=0;i<16;++i)z+=x[ulong(g)*4096+t+i*256];float v=simd_sum(z.x+z.y+z.z+z.w);if(l==0)y[g*8+s]=v;}
