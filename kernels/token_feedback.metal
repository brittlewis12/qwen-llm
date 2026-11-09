// Request-local GPU token selection. Total float order and ascending IDs break
// ties. Local barriers never depend on another threadgroup's progress.
#include <metal_stdlib>
using namespace metal;
struct token_pair { float value; uint id; };
struct token_sort_args { uint n; uint keep; uint first; uint run; };
uint token_key(float v) {
    uint b=as_type<uint>(v);
    if ((b&0x7fffffffu)>0x7f800000u) return UINT_MAX;
    return (b&0x80000000u) ? ~b : (b^0x80000000u);
}
bool token_before(token_pair a, token_pair b) {
    uint x=token_key(a.value), y=token_key(b.value);
    return x>y || (x==y && a.id<b.id);
}
kernel void kernel_token_sort_tiles(
    constant token_sort_args &a [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device const token_pair *input [[buffer(2)]],
    device token_pair *output [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup token_pair tile[256];
    uint i=group*256+tid;
    tile[tid]= i<a.n ? (a.first ? token_pair{logits[i],i}:input[i]) : token_pair{-INFINITY,i};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=2;k<=256;k*=2) for(uint j=k/2;j>0;j/=2) {
        uint other=tid^j;
        token_pair own=tile[tid], peer=tile[other];
        bool descending=(tid&k)==0;
        bool lower=(tid&j)==0;
        token_pair chosen=(descending==lower) ? (token_before(own,peer)?own:peer) : (token_before(own,peer)?peer:own);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile[tid]=chosen;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid<a.keep) output[ulong(group)*a.keep+tid]=tile[tid];
}
// Merge sorted runs by stable parallel rank. This supports unbounded top-k and
// nucleus-only configurations; bounded K<=256 uses the cheaper tile reduction.
kernel void kernel_token_merge_runs(
    constant token_sort_args &a [[buffer(0)]],
    device const token_pair *input [[buffer(1)]],
    device token_pair *output [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i>=a.n)return;
    uint pair_start=(i/(2*a.run))*(2*a.run);
    bool right=(i-pair_start)>=a.run;
    uint other_start=pair_start+(right?0:a.run);
    uint other_end=min(other_start+a.run,a.n);
    uint lo=min(other_start,a.n),hi=other_end;
    token_pair value=input[i];
    while(lo<hi){uint mid=lo+(hi-lo)/2;if(token_before(input[mid],value))lo=mid+1;else hi=mid;}
    uint local=i-(right?pair_start+a.run:pair_start);
    output[pair_start+local+(lo-min(other_start,a.n))]=value;
}
struct token_sample_args { uint n; float temperature; float min_p; float top_p; float uniform; };
kernel void kernel_token_sample_sorted(
    constant token_sample_args &a [[buffer(0)]],
    device const token_pair *sorted [[buffer(1)]],
    device int *token [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    if(tid!=0)return;
    float maximum=sorted[0].value;
    uint maximum_bits=as_type<uint>(maximum);
    if((maximum_bits&0x7fffffffu)>0x7f800000u){token[0]=-int(sorted[0].id)-1;return;}
    if(maximum_bits==0xff800000u){token[0]=INT_MIN;return;}
    bool positive_infinity=maximum_bits==0x7f800000u;
    // Three passes preserve the specified filter order and avoid scratch
    // allocation. The common Flash-Next preset retains only 20 candidates.
    uint count=0; float total=0.f, correction=0.f;
    for(uint i=0;i<a.n;++i){
        float v=sorted[i].value;
        if(a.min_p>0.f && exp(v-maximum)<a.min_p)break;
        float weight=positive_infinity ? (as_type<uint>(v)==0x7f800000u?1.f:0.f):exp((v-maximum)/a.temperature);
        float y=weight-correction,t=total+y;correction=(t-total)-y;total=t;++count;
    }
    float retained=0.f;uint limit=count;correction=0.f;
    if(a.top_p<1.f){for(uint i=0;i<count;++i){float v=sorted[i].value;float w=positive_infinity?(as_type<uint>(v)==0x7f800000u?1.f:0.f):exp((v-maximum)/a.temperature);float y=w-correction,t=retained+y;correction=(t-retained)-y;retained=t;if(retained>=a.top_p*total){limit=i+1;break;}}}
    else retained=total;
    float target=a.uniform*retained,cumulative=0.f;correction=0.f;
    for(uint i=0;i<limit;++i){float v=sorted[i].value;float w=positive_infinity?(as_type<uint>(v)==0x7f800000u?1.f:0.f):exp((v-maximum)/a.temperature);float y=w-correction,t=cumulative+y;correction=(t-cumulative)-y;cumulative=t;if(target<cumulative || i+1==limit){token[0]=int(sorted[i].id);return;}}
    token[0]=INT_MIN;
}
