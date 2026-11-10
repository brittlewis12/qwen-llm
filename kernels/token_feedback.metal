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

// Host-order radix path for unbounded top-k. `token_descending_bits` is the
// host's `descending_logit_bits`; a stable ascending sort of the full 64-bit
// key (bits, then token id) gives descending logit with ascending token ties.
uint token_descending_bits(float v) {
    uint b=as_type<uint>(v);
    if ((b&0x7fffffffu)>0x7f800000u) return 0u;
    uint ascending=(b&0x80000000u)?~b:(b|0x80000000u);
    return ~ascending;
}
struct token_radix_args {
    uint n; uint shift; uint buckets; uint mask; uint groups; uint pad0; uint pad1; uint pad2;
};
kernel void kernel_token_radix_build(
    device const float *logits [[buffer(0)]],
    device ulong *keys [[buffer(1)]],
    constant token_radix_args &a [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=a.n)return;
    keys[i]=(ulong(token_descending_bits(logits[i]))<<32)|ulong(i);
}
// One threadgroup per 256-element chunk: shared histogram, written to device.
kernel void kernel_token_radix_chunk_hist(
    device const ulong *keys [[buffer(0)]],
    device uint *chunk_hist [[buffer(1)]],
    constant token_radix_args &a [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[2048];
    for(uint b=tid;b<a.buckets;b+=256)atomic_store_explicit(&hist[b],0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i=group*256+tid;
    if(i<a.n){uint digit=uint((keys[i]>>a.shift)&ulong(a.mask));atomic_fetch_add_explicit(&hist[digit],1u,memory_order_relaxed);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint b=tid;b<a.buckets;b+=256)chunk_hist[group*a.buckets+b]=atomic_load_explicit(&hist[b],memory_order_relaxed);
}
// Per-bucket running base across chunks; totals feed the bucket prefix.
kernel void kernel_token_radix_chunk_base(
    device const uint *chunk_hist [[buffer(0)]],
    device uint *chunk_base [[buffer(1)]],
    device uint *bucket_totals [[buffer(2)]],
    constant token_radix_args &a [[buffer(3)]],
    uint b [[thread_position_in_grid]]) {
    if(b>=a.buckets)return;
    uint sum=0;
    for(uint g=0;g<a.groups;++g){chunk_base[g*a.buckets+b]=sum;sum+=chunk_hist[g*a.buckets+b];}
    bucket_totals[b]=sum;
}
kernel void kernel_token_radix_bucket_prefix(
    device const uint *bucket_totals [[buffer(0)]],
    device uint *bucket_offsets [[buffer(1)]],
    constant token_radix_args &a [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(tid!=0)return;
    uint sum=0;
    for(uint b=0;b<a.buckets;++b){bucket_offsets[b]=sum;sum+=bucket_totals[b];}
}
// Stable scatter: thread 0 assigns each chunk element its in-bucket rank in
// input order, so equal logits keep ascending token id across passes.
kernel void kernel_token_radix_stable_scatter(
    device const ulong *keys [[buffer(0)]],
    device ulong *out [[buffer(1)]],
    device const uint *chunk_base [[buffer(2)]],
    device const uint *bucket_offsets [[buffer(3)]],
    constant token_radix_args &a [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[2048];
    threadgroup uint rank[256];
    for(uint b=tid;b<a.buckets;b+=256)atomic_store_explicit(&hist[b],0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base=group*256;
    if(tid==0){
        uint len=min(256u,a.n-base);
        for(uint j=0;j<len;++j){uint digit=uint((keys[base+j]>>a.shift)&ulong(a.mask));rank[j]=atomic_fetch_add_explicit(&hist[digit],1u,memory_order_relaxed);}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint i=base+tid;
    if(i<a.n){uint digit=uint((keys[i]>>a.shift)&ulong(a.mask));uint pos=bucket_offsets[digit]+chunk_base[group*a.buckets+digit]+rank[tid];out[pos]=keys[i];}
}
kernel void kernel_token_radix_pairs(
    device const ulong *keys [[buffer(0)]],
    device token_pair *pairs [[buffer(1)]],
    constant token_radix_args &a [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=a.n)return;
    ulong k=keys[i];
    uint d=uint(k>>32),ascending=~d;
    uint bits=(ascending&0x80000000u)?(ascending&0x7fffffffu):~ascending;
    pairs[i]=token_pair{as_type<float>(bits),uint(k)};
}

struct token_weight_args { uint n; float temperature; float min_p; uint pad; };
// Parallel weight pass: the exp is the expensive part of sampler-v1's scan.
// `raw` is the pre-temperature weight for the min-p filter, matching the host.
kernel void kernel_token_weight_sorted(
    constant token_weight_args &a [[buffer(0)]],
    device const token_pair *sorted [[buffer(1)]],
    device float *weights [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=a.n)return;
    float v=sorted[i].value,maximum=sorted[0].value;
    uint mb=as_type<uint>(maximum);
    if((mb&0x7fffffffu)>0x7f800000u || mb==0xff800000u){weights[i]=0.f;return;}
    bool pi=mb==0x7f800000u;
    float raw=pi?(as_type<uint>(v)==0x7f800000u?1.f:0.f):exp(v-maximum);
    if(a.min_p>0.f && raw<a.min_p){weights[i]=0.f;return;}
    weights[i]=pi?raw:exp((v-maximum)/a.temperature);
}
// Same sequential compensated scan and tie order as kernel_token_sample_sorted,
// but over precomputed weights, so the single thread only adds.
kernel void kernel_token_sample_weighted(
    constant token_sample_args &a [[buffer(0)]],
    device const token_pair *sorted [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device int *token [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid!=0)return;
    float maximum=sorted[0].value;
    uint maximum_bits=as_type<uint>(maximum);
    if((maximum_bits&0x7fffffffu)>0x7f800000u){token[0]=-int(sorted[0].id)-1;return;}
    if(maximum_bits==0xff800000u){token[0]=INT_MIN;return;}
    uint n=a.n;
    float total=0.f,correction=0.f;
    for(uint i=0;i<n;++i){float y=weights[i]-correction,t=total+y;correction=(t-total)-y;total=t;}
    float retained=0.f;uint limit=n;correction=0.f;
    if(a.top_p<1.f){for(uint i=0;i<n;++i){float y=weights[i]-correction,t=retained+y;correction=(t-retained)-y;retained=t;if(retained>=a.top_p*total){limit=i+1;break;}}}
    else retained=total;
    float target=a.uniform*retained,cumulative=0.f;correction=0.f;
    for(uint i=0;i<limit;++i){float y=weights[i]-correction,t=cumulative+y;correction=(t-cumulative)-y;cumulative=t;if(target<cumulative || i+1==limit){token[0]=int(sorted[i].id);return;}}
    token[0]=INT_MIN;
}

struct token_block_args { uint n; uint block; uint blocks; uint pad; };
struct token_sample_block_args { uint n; float top_p; float uniform; uint block; uint blocks; uint pad; };
// One thread per block; Kahan within the block preserves the sequential order
// locally, and the block prefix below preserves it across blocks.
kernel void kernel_token_block_sums(
    constant token_block_args &a [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *block_sums [[buffer(2)]],
    uint block [[thread_position_in_grid]]) {
    if(block>=a.blocks)return;
    uint start=block*a.block,end=min(start+a.block,a.n);
    float sum=0.f,correction=0.f;
    for(uint i=start;i<end;++i){float y=weights[i]-correction,t=sum+y;correction=(t-sum)-y;sum=t;}
    block_sums[block]=sum;
}
kernel void kernel_token_block_prefix(
    constant token_block_args &a [[buffer(0)]],
    device const float *block_sums [[buffer(1)]],
    device float *block_prefix [[buffer(2)]],
    device float *total [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if(tid!=0)return;
    float sum=0.f,correction=0.f;
    for(uint b=0;b<a.blocks;++b){block_prefix[b]=sum;float y=block_sums[b]-correction,t=sum+y;correction=(t-sum)-y;sum=t;}
    total[0]=sum;
}
// Find the top-p limit and the categorical draw by binary-searching the block
// prefix and scanning only the selected block.
kernel void kernel_token_sample_blocks(
    constant token_sample_block_args &a [[buffer(0)]],
    device const token_pair *sorted [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device const float *block_sums [[buffer(3)]],
    device const float *block_prefix [[buffer(4)]],
    device const float *total [[buffer(5)]],
    device int *token [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid!=0)return;
    float maximum=sorted[0].value;
    uint maximum_bits=as_type<uint>(maximum);
    if((maximum_bits&0x7fffffffu)>0x7f800000u){token[0]=-int(sorted[0].id)-1;return;}
    if(maximum_bits==0xff800000u){token[0]=INT_MIN;return;}
    float tot=total[0];
    if(!(tot>0.f) || !isfinite(tot)){token[0]=INT_MIN;return;}
    float retained=tot;uint limit=a.n;
    if(a.top_p<1.f){
        float target=a.top_p*tot;
        uint lo=0,hi=a.blocks;
        while(lo+1<hi){uint mid=(lo+hi)/2;if(block_prefix[mid]<=target)lo=mid;else hi=mid;}
        uint start=lo*a.block,end=min(start+a.block,a.n);
        float cum=block_prefix[lo],correction=0.f;uint i=start;
        for(;i<end;++i){float y=weights[i]-correction,t=cum+y;correction=(t-cum)-y;cum=t;if(cum>=target)break;}
        limit=min(i+1,a.n);retained=cum;
    }
    float target=a.uniform*retained;
    uint lo=0,hi=a.blocks;
    while(lo+1<hi){uint mid=(lo+hi)/2;if(block_prefix[mid]<=target)lo=mid;else hi=mid;}
    uint start=lo*a.block,end=min(start+a.block,a.n);
    float cum=block_prefix[lo],correction=0.f;
    for(uint i=start;i<end;++i){float y=weights[i]-correction,t=cum+y;correction=(t-cum)-y;cum=t;if(target<cum || i+1==limit){token[0]=int(sorted[i].id);return;}}
    token[0]=INT_MIN;
}

// ---- GLM batched loop plumbing ---------------------------------------------
// Per-member token gather and sparse/index controls encoded at the start of the
// forward command, so a prepared next frame never writes a buffer the in-flight
// frame reads.
struct glm_token_rows_args { uint rows; uint pad0; uint pad1; uint pad2; };
kernel void kernel_glm_token_rows(
    device const int *t0 [[buffer(0)]],
    device const int *t1 [[buffer(1)]],
    device const int *t2 [[buffer(2)]],
    device const int *t3 [[buffer(3)]],
    device int *dst [[buffer(4)]],
    constant glm_token_rows_args &a [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=a.rows)return;
    dst[i]=(i==0)?t0[0]:(i==1)?t1[0]:(i==2)?t2[0]:t3[0];
}
struct glm_sparse_args { uint visible[4]; uint pool; uint rows; uint blocks; uint stride; };
kernel void kernel_glm_sparse_member(
    device int *pools [[buffer(0)]],
    device int *visible [[buffer(1)]],
    device int *status [[buffer(2)]],
    constant glm_sparse_args &a [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if(i==0){pools[0]=(int)(a.visible[0]/a.pool);visible[0]=(int)a.visible[0];}
    if(i<a.blocks)status[i]=-1;
}
kernel void kernel_glm_sparse_shared(
    device int *pools [[buffer(0)]],
    device int *visible [[buffer(1)]],
    device int *status [[buffer(2)]],
    constant glm_sparse_args &a [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if(i<a.rows){pools[i]=(int)(a.visible[i]/a.pool);visible[i]=(int)a.visible[i];}
    if(i<a.rows*a.blocks){uint mla=i/a.rows,row=i%a.rows;status[mla*a.stride+row]=-1;}
}
// Fast distribution-exact categorical draw for the unforced agent preset
// (temperature > 0, top_p = 1, top_k = 0, min_p = 0): Gumbel-max over the whole
// vocabulary. score_i = logit_i / T + gumbel(u_i), where u_i is a counter-based
// hash of (seed, member, i); the argmax (lowest id on ties) is an exact draw
// from softmax(logits / T). Unlike kernel_token_sample_*, no host uniform and no
// sort are used, and the result is not token-identical with the host sampler.
struct token_gumbel_args {
    ulong seed;
    uint vocab;
    uint members;
    uint blocks;
    uint block;
    float inv_temp;
    uint pad;
};
static inline float token_gumbel_uniform(ulong seed, uint member, uint i) {
    ulong z = seed ^ (ulong(member) << 40) ^ ((ulong(i) + 1ul) * 0x9E3779B97F4A7C15ul);
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ul;
    z ^= z >> 27; z *= 0x94D049BB133111EBul;
    z ^= z >> 31;
    // (0, 1), 24-bit resolution, never 0 or 1.
    return (float(z >> 40) + 1.0f) / 16777217.0f;
}
static inline token_pair token_pair_winner(token_pair a, token_pair b) {
    return token_before(a, b) ? a : b;
}
kernel void kernel_token_gumbel_block(
        constant token_gumbel_args & a [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device token_pair * partial [[buffer(2)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint member = group.y, b = group.x;
    const uint start = b * a.block, end = min(start + a.block, a.vocab);
    const uint stride = 256u;
    token_pair best = { -INFINITY, UINT_MAX };
    for (uint i = start + tid; i < end; i += stride) {
        const float u = token_gumbel_uniform(a.seed, member, i);
        const float score = logits[(ulong(member) * a.vocab) + i] * a.inv_temp
            - log(-log(u));
        token_pair c = { score, i };
        best = token_pair_winner(best, c);
    }
    threadgroup token_pair tg[256];
    tg[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) tg[tid] = token_pair_winner(tg[tid], tg[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partial[ulong(member) * a.blocks + b] = tg[0];
}
kernel void kernel_token_gumbel_final(
        constant token_gumbel_args & a [[buffer(0)]],
        device const token_pair * partial [[buffer(2)]],
        device int * token [[buffer(3)]],
        uint member [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    token_pair best = { -INFINITY, UINT_MAX };
    for (uint b = tid; b < a.blocks; b += 256u) {
        token_pair c = partial[ulong(member) * a.blocks + b];
        best = token_pair_winner(best, c);
    }
    threadgroup token_pair tg[256];
    tg[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) tg[tid] = token_pair_winner(tg[tid], tg[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) token[member] = best.id == UINT_MAX ? INT_MIN : int(best.id);
}
