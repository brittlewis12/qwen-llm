// Bench-only Q6 F32 microbatch dataflow screen. No runtime selector.
#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
struct fewrow_gen_q6_K {
    enum { block_bytes = 210, block_weights = 256, nl = 16 };

    static inline void dequant(device const uchar * xb, short il, thread float4x4 & reg) {
        device const uchar * ql = xb;
        device const uchar * qh = xb + 128;
        device const char  * scales = (device const char *) (xb + 192);
        const float d_all = float(as_type<half>(*(device const ushort *) (xb + 208)));

        ql += 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
        qh += 32 * (il / 8) + 16 * (il & 1);
        float sc = float(scales[(il % 2) + 2 * (il / 2)]);
        const short il4 = (il / 2) & 3;

        const uint kmask1 = il4 > 1 ? (il4 > 2 ? 0xC0C0C0C0u : 0x30303030u)
                                    : (il4 > 0 ? 0x0C0C0C0Cu : 0x03030303u);
        const uint kmask2 = il4 > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
        const float ml = d_all * sc * 32.0f;
        const float dl0 = d_all * sc;
        const float dl1 = dl0 / 256.0f;
        const float dl2 = dl0 / (256.0f * 256.0f);
        const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
        const uint shr_h = il4 > 2 ? 2u : 0u;
        const uint shl_h = il4 > 1 ? 0u : (il4 > 0 ? 2u : 4u);
        const uint shr_l = il4 > 1 ? 4u : 0u;

        device const ushort * ql16 = (device const ushort *) ql;
        device const ushort * qh16 = (device const ushort *) qh;

        FOR_UNROLL (int i = 0; i < 4; ++i) {
            // Upstream reads two 16-bit words per 32-bit lane; the block
            // base is only 2-byte aligned, so 32-bit loads are not safe.
            const uint low = (uint(ql16[2 * i]) | (uint(ql16[2 * i + 1]) << 16)) & kmask2;
            const uint high = (uint(qh16[2 * i]) | (uint(qh16[2 * i + 1]) << 16)) & kmask1;
            const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
            reg[i][0] = dl0 * float(q & 0xFFu) - ml;
            reg[i][1] = dl1 * float(q & 0xFF00u) - ml;
            reg[i][2] = dl2 * float(q & 0xFF0000u) - ml;
            reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
        }
    }
};

struct glm_batch_q6_args { uint n_in, n_out, n_cols; };

// Decode each 16-weight fragment once and use it for all live members.
// Unlike the old exact two-row experiment this uses explicit F32 weight
// fragments and FMAs, with a freely reassociated reduction (Fast/Half only).
template<ushort B, ushort OUT, ushort NSG>
kernel void glm_batch_q6_vector(
 constant glm_batch_q6_args & a [[buffer(0)]],
 device const uchar * w [[buffer(1)]], device const float * x [[buffer(2)]],
 device float * y [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
 const uint first=(group*NSG+sg)*OUT;
 const ulong rb=ulong(a.n_in/256)*210;
 float sum[OUT][B];
 FOR_UNROLL(ushort o=0;o<OUT;++o) FOR_UNROLL(ushort b=0;b<B;++b) sum[o][b]=0.0f;
 for(uint k=uint(lane)*16;k<a.n_in;k+=512) {
  float4 xv[B][4];
  FOR_UNROLL(ushort b=0;b<B;++b) FOR_UNROLL(ushort j=0;j<4;++j)
   xv[b][j]=*((device const float4 *)(x+ulong(b)*a.n_in+k+4*j));
  FOR_UNROLL(ushort o=0;o<OUT;++o) {
   if(first+o<a.n_out) {
    float4x4 v;
    fewrow_gen_q6_K::dequant(w+ulong(first+o)*rb+ulong(k/256)*210,short((k/16)%16),v);
    FOR_UNROLL(ushort j=0;j<4;++j) FOR_UNROLL(ushort e=0;e<4;++e)
     FOR_UNROLL(ushort b=0;b<B;++b) sum[o][b]=fma(v[j][e],xv[b][j][e],sum[o][b]);
   }
  }
 }
 FOR_UNROLL(ushort o=0;o<OUT;++o) FOR_UNROLL(ushort b=0;b<B;++b) {
  const float v=simd_sum(sum[o][b]);
  if(lane==0 && first+o<a.n_out) y[ulong(b)*a.n_out+first+o]=v;
 }
}
typedef decltype(glm_batch_q6_vector<2,1,4>) glm_batch_q6_t;
#define V(B,O,S) template [[host_name("glm_batch_q6_vector_b" #B "_o" #O "_s" #S)]] kernel glm_batch_q6_t glm_batch_q6_vector<B,O,S>;
#define VS(B) V(B,1,2) V(B,1,4) V(B,2,2) V(B,2,4) V(B,4,2) V(B,4,4)
VS(1) VS(2) VS(3) VS(4)

// Eight consecutive quant values/lane. Apply the exact F32 block scale
// after each short dot, avoiding full scaled weight fragments in registers.
// Reduction order is E1; no Half weight/activation conversion occurs.
template<ushort B, ushort OUT, ushort NSG>
kernel void glm_batch_q6_scaled(
 constant glm_batch_q6_args & a [[buffer(0)]],
 device const uchar * w [[buffer(1)]], device const float * x [[buffer(2)]],
 device float * y [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
 const uint first=(group*NSG+sg)*OUT;
 const ulong rb=ulong(a.n_in/256)*210;
 const ushort offset=lane*8;
 const ushort hi=offset/128;
 const ushort quadrant=(offset%128)/32;
 const ushort within=offset%32;
 float sums[OUT][B];
 FOR_UNROLL(ushort o=0;o<OUT;++o) FOR_UNROLL(ushort b=0;b<B;++b) sums[o][b]=0;
 for(uint block=0;block<a.n_in/256;++block) {
  float xv[B][8];
  FOR_UNROLL(ushort b=0;b<B;++b) FOR_UNROLL(ushort j=0;j<8;++j)
   xv[b][j]=x[ulong(b)*a.n_in+ulong(block)*256+offset+j];
  FOR_UNROLL(ushort o=0;o<OUT;++o) if(first+o<a.n_out) {
   device const uchar * p=w+ulong(first+o)*rb+ulong(block)*210;
   const float scale=float(as_type<half>(*((device const ushort *)(p+208))))*float(((device const char *)(p+192))[offset/16]);
   float partial[B];FOR_UNROLL(ushort b=0;b<B;++b)partial[b]=0;
   FOR_UNROLL(ushort j=0;j<8;++j) {
    const uchar lo=p[64*hi+32*(quadrant%2)+within+j];
    const uchar high=p[128+32*hi+within+j];
    const int q=int(((lo>>(quadrant>=2?4:0))&15)|(((high>>(2*quadrant))&3)<<4))-32;
    FOR_UNROLL(ushort b=0;b<B;++b) partial[b]=fma(float(q),xv[b][j],partial[b]);
   }
   FOR_UNROLL(ushort b=0;b<B;++b)sums[o][b]=fma(scale,partial[b],sums[o][b]);
  }
 }
 FOR_UNROLL(ushort o=0;o<OUT;++o) FOR_UNROLL(ushort b=0;b<B;++b) {
  float total=simd_sum(sums[o][b]);if(lane==0&&first+o<a.n_out)y[ulong(b)*a.n_out+first+o]=total;
 }
}
typedef decltype(glm_batch_q6_scaled<2,1,4>) glm_batch_q6_scaled_t;
#define S(B,O,N) template [[host_name("glm_batch_q6_scaled_b" #B "_o" #O "_s" #N)]] kernel glm_batch_q6_scaled_t glm_batch_q6_scaled<B,O,N>;
#define SS(B) S(B,1,2) S(B,1,4) S(B,2,2) S(B,2,4) S(B,4,2) S(B,4,4)
SS(1) SS(2) SS(3) SS(4)

// Arithmetic of kernel_mat_vec_f32_f32_lcpp_r2, with weight loads and
// both barriers shared by independent member accumulators.
template<ushort B>
kernel void glm_batch_router(
 constant glm_batch_q6_args & a [[buffer(0)]],
 device const float * weight [[buffer(1)]], device const float * x [[buffer(2)]],
 device float * y [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float red[B][2][32];
 const uint first=group*2;
 if(first>=a.n_out)return;
 float sums[B][2];
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)sums[b][row]=0;
 for(uint i=uint(sg)*32+lane;i<a.n_in/4;i+=128) {
  float4 xv[B];FOR_UNROLL(ushort b=0;b<B;++b)xv[b]=((device const float4 *)(x+ulong(b)*a.n_in))[i];
  FOR_UNROLL(ushort row=0;row<2;++row)if(first+row<a.n_out){
   const float4 wv=((device const float4 *)(weight+ulong(first+row)*a.n_in))[i];
   FOR_UNROLL(ushort b=0;b<B;++b){
    const float4 v=xv[b];sums[b][row]+=wv.x*v.x+wv.y*v.y+wv.z*v.z+wv.w*v.w;
   }
  }
 }
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row){
  if(sg==0)red[b][row][lane]=0;
  sums[b][row]=simd_sum(sums[b][row]);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)if(lane==0)red[b][row][sg]=sums[b][row];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)if(sg==0&&first+row<a.n_out){
  const float total=simd_sum(red[b][row][lane]);if(lane==0)y[ulong(b)*a.n_out+first+row]=total;
 }
}
typedef decltype(glm_batch_router<2>) glm_batch_router_t;
#define R(B) template [[host_name("glm_batch_router_b" #B)]] kernel glm_batch_router_t glm_batch_router<B>;
R(2) R(3) R(4)
