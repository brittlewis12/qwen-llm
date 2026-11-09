#include <metal_stdlib>
using namespace metal;
struct fn_gpu_ple_banks { array<device const uchar *,32> bank [[id(0)]]; device const uchar *tail [[id(32)]]; };
struct fn_gpu_controls { array<device int *,96> scalar [[id(0)]]; };
struct fn_gpu_input_args {
    ulong table_offset;
    ulong tail_begin;
    uint row_bytes,ngram,heads,heads_per_ngram,eos,controls,position,pad;
};
kernel void kernel_qwen4exp_gpu_ple_rows(
    constant fn_gpu_ple_banks &banks [[buffer(0)]],
    device const ulong *metadata [[buffer(1)]],
    constant fn_gpu_input_args &a [[buffer(2)]],
    device const int *token [[buffer(3)]],device const int *history [[buffer(4)]],
    device uchar *packed [[buffer(5)]],uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint nt [[threads_per_threadgroup]]) {
    threadgroup ulong address;
    if(tid==0){
        uint ngram=2+head/a.heads_per_ngram;
        ulong mixed=ulong(uint(token[0]))*metadata[0];bool cut=false;
        for(uint shift=1;shift<ngram;++shift){
            uint prior= cut || shift>uint(history[0]) ? a.eos:uint(history[1+uint(history[0])-shift]);
            mixed^=ulong(prior)*metadata[shift];cut|=prior==a.eos;
        }
        long size=long(metadata[a.ngram+a.heads+head]);
        long remainder=as_type<long>(mixed)%size;if(remainder<0)remainder+=size;
        ulong row=metadata[a.ngram+head]+ulong(remainder);
        address=a.table_offset+row*a.row_bytes;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<a.row_bytes;j+=nt){ulong p=address+j;packed[ulong(head)*a.row_bytes+j]=p>=a.tail_begin?banks.tail[p-a.tail_begin]:banks.bank[uint(p>>30)][uint(p&0x3ffffffful)];}
}
// Separate dispatch: all heads finish reading the old history before its update.
kernel void kernel_qwen4exp_gpu_input_controls(
    constant fn_gpu_controls &controls [[buffer(0)]],device const uint *kind [[buffer(1)]],
    constant fn_gpu_input_args &a [[buffer(2)]],device const int *feedback [[buffer(3)]],
    device int *history [[buffer(4)]],device int *input [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid<a.controls)controls.scalar[tid][0]=kind[tid]==0?0:int((a.position+1)/kind[tid]);
    if(tid==0){
        uint count=uint(history[0]),keep=a.ngram-1;
        if(keep>0){if(count==keep){for(uint i=1;i<keep;++i)history[i]=history[i+1];}else ++count;
            history[count]=feedback[0];history[0]=int(count);}
        input[0]=feedback[0];
    }
}
