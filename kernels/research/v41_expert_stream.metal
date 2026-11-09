#include <metal_stdlib>
using namespace metal;
struct V41StreamArgs { ulong row_bytes; uint n_out,experts,channels,slots,banks,read; };
kernel void kernel_research_v41_expert_stream(
 constant V41StreamArgs &a [[buffer(0)]], device const uchar *bank0 [[buffer(1)]],
 device const uchar *bank1 [[buffer(2)]],device const int *ids [[buffer(3)]],
 device const float *held [[buffer(4)]],device float *output [[buffer(5)]],
 device uint *checksum [[buffer(6)]],uint3 g [[threadgroup_position_in_grid]],
 uint3 grid [[threadgroups_per_grid]],uint3 ts [[threads_per_threadgroup]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 const uint nth=ts.x*ts.y*ts.z,first=g.x*a.channels;
 const uint count=min(a.channels,a.n_out-first);
 const ulong slot=ulong(g.z)*a.slots+g.y;
 const int expert=ids[slot];uint sum=tid+g.x;
 if (a.read && expert>=0 && uint(expert)<a.experts) {
  for(uint b=0;b<a.banks;++b) {
   device const uchar *bank=b==0?bank0:bank1;
   for(uint r=0;r<count;++r) {
    device const uchar *row=bank+(ulong(expert)*a.n_out+first+r)*a.row_bytes;
    const uint nv=uint(a.row_bytes/8);
    for(uint v=tid;v<nv;v+=nth) { const ushort4 q=((device const packed_ushort4 *)row)[v];sum+=uint(q.x)+q.y+q.z+q.w; }
    for(uint off=nv*8+2*tid;off<a.row_bytes;off+=2*nth)sum+=*((device const ushort *)(row+off));
   }
  }
 }
 for(uint r=tid;r<count;r+=nth) {const ulong i=slot*a.n_out+first+r;output[i]=held[i];}
 sum=simd_sum(sum);
 if(lane==0)checksum[(((g.z*grid.y+g.y)*grid.x+g.x)*(nth/32))+sg]=sum;
}
