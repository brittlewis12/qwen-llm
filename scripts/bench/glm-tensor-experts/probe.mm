#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include "../register-operands/contracts.h"
#include <random>

struct Args {
    uint32_t M, N, K, nb01, stride_b, min_count, max_count, b_div, slot_limit;
};
static constexpr size_t max_fixture_arms = 14;
static constexpr size_t max_name_capacity = 192;
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void emit(NSDictionary *record) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    puts("");
    fflush(stdout);
}
static uint32_t rng(uint32_t &state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static probe_contracts::Domains domains(id<MTLDevice> device) {
    uint64_t physical = 0;
    size_t length = sizeof(physical);
    require(sysctlbyname("hw.memsize", &physical, &length, nullptr, 0) == 0
            && length == sizeof(physical) && physical > 0, "physical telemetry unavailable");
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t size = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    require(host != MACH_PORT_NULL, "host telemetry port unavailable");
    vm_size_t page = 0;
    const auto page_result = host_page_size(host, &page);
    const auto result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &size);
    const auto released = mach_port_deallocate(mach_task_self(), host);
    const auto needed = (offsetof(vm_statistics64_data_t, internal_page_count) + sizeof(vm.internal_page_count)) / sizeof(integer_t);
    require(result == KERN_SUCCESS && size >= needed && size <= HOST_VM_INFO64_COUNT
            && page_result == KERN_SUCCESS && page > 0 && !(page & (page - 1)) && released == KERN_SUCCESS,
            "CPU telemetry unavailable/incomplete");
    using probe_contracts::add;
    for (uint64_t pages : {uint64_t(vm.free_count), uint64_t(vm.inactive_count), uint64_t(vm.internal_page_count)})
        require(pages <= physical / page, "inconsistent CPU telemetry");
    require(add(vm.free_count, vm.inactive_count) <= physical / page, "inconsistent free/inactive telemetry");
    const uint64_t available_pages = add(vm.free_count, vm.inactive_count > vm.internal_page_count
            ? vm.inactive_count - vm.internal_page_count : 0);
    require(available_pages <= physical / page, "inconsistent CPU availability");
    return {physical, available_pages * page, device.recommendedMaxWorkingSetSize,
            device.currentAllocatedSize, device.maxBufferLength, page,
            vm.free_count, vm.inactive_count, vm.internal_page_count};
}
static id<MTLBuffer> alloc(id<MTLDevice> device, uint64_t bytes) {
    require(bytes <= device.maxBufferLength, "buffer exceeds maxBufferLength");
    id<MTLBuffer> result = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    require(result != nil, "ordinary shared-buffer allocation failed");
    return result;
}

struct Fixture {
    Args a;
    int experts, block;
    bool half;
    id<MTLBuffer> w,x,counts,ids,out,decoded,decoded_alt,sink;
    std::vector<int> c;
    Fixture(id<MTLDevice> d, int block_bytes, bool half_act, int m, int k, int e, int n): experts(e), block(block_bytes), half(half_act) {
        a={uint32_t(m),uint32_t(n),uint32_t(k),uint32_t(k/256*block),uint32_t(k),1,UINT32_MAX/2,1,uint32_t(n*8)};
        std::array<uint64_t,7> sizes={uint64_t(e)*m*a.nb01,uint64_t(a.slot_limit)*k*(half?2:4),uint64_t(e)*4,uint64_t(e)*n*4,uint64_t(a.slot_limit)*m*4,uint64_t(e)*m*k*2,uint64_t(e)*((m+63)/64)*((n+15)/16)*128*4};
        auto dm=domains(d); probe_contracts::Price p{sizes};
        for(auto z:sizes)p.metal+=probe_contracts::pages(z,dm.page);
        // Second decoded bank for the fast-decoder bitwise witness.
        p.metal+=probe_contracts::pages(sizes[5],dm.page);
        p.cpu_vectors=uint64_t(a.slot_limit)*m*4+uint64_t(e)*4; p.cpu_control=1<<20;
        const auto why=probe_contracts::admit(dm,p); require(!why,why?why:"");
        emit(@{@"event":@"admission",@"metal_bytes":@(p.metal),@"mwss":@(dm.mwss),@"physical":@(dm.physical),@"current_metal":@(dm.current_metal)});
        w=alloc(d,sizes[0]); x=alloc(d,sizes[1]);counts=alloc(d,sizes[2]);ids=alloc(d,sizes[3]);out=alloc(d,sizes[4]);decoded=alloc(d,sizes[5]);sink=alloc(d,sizes[6]);decoded_alt=alloc(d,sizes[5]);
        uint32_t state=11177; auto bytes=(uint8_t*)w.contents;
        for(uint64_t off=0;off<sizes[0];off+=block){
            bytes[off]=0;bytes[off+1]=0x20;
            for(int b=2;b<block;++b)bytes[off+b]=uint8_t(rng(state));
        }
        for(uint64_t i=0;i<uint64_t(a.slot_limit)*k;++i){
            float v=float(int(rng(state)%257)-128)/128;
            if(half)((_Float16*)x.contents)[i]=v;else((float*)x.contents)[i]=v;
        }
        memset(out.contents,0,sizes[4]);
    }
    void buckets(NSDictionary *hist){
        c.clear();
        for(NSString *key in hist)for(int j=0;j<[hist[key] intValue];++j)c.push_back(key.intValue);
        require(c.size()==size_t(experts),"histogram size");
        std::mt19937 random(17);std::shuffle(c.begin(),c.end(),random);
        memcpy(counts.contents,c.data(),c.size()*4);
        auto ix=(int*)ids.contents; std::fill(ix,ix+experts*a.N,-1);
        int slot=0;
        for(int e=0;e<experts;++e)for(int j=0;j<c[e];++j)ix[e*a.N+j]=slot++;
        require(slot==int(a.slot_limit),"route histogram must sum to rows*8");
    }
};
struct Probe {
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> q;id<MTLLibrary> lib;
    NSMutableDictionary *cache=[NSMutableDictionary new];
    Probe(const char *path, bool tensor=true){
        require(d!=nil,"Metal device");q=[d newCommandQueue];NSError *err=nil;
        if (tensor && ![d supportsFamily:MTLGPUFamilyApple10]) throw std::runtime_error("Apple10 tensor execution required; compile-only on M4");
        lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&err];require(lib!=nil,err.description.UTF8String);
        NSMutableArray *sets=[NSMutableArray new];for(id<MTLCounterSet> s in d.counterSets){
            NSMutableArray *names=[NSMutableArray new];for(id<MTLCounter> c in s.counters)[names addObject:c.name];
            [sets addObject:@{@"name":s.name,@"counters":names}];
        }
        emit(@{@"event":@"device",@"name":d.name,@"max_tgm":@(d.maxThreadgroupMemoryLength),@"counter_sets":sets});
    }
    id<MTLComputePipelineState> pipeline(NSString *name){
        id<MTLComputePipelineState> p=cache[name];if(p)return p;
        id<MTLFunction> f=[lib newFunctionWithName:name];require(f!=nil,name.UTF8String);NSError *err=nil;
        p=[d newComputePipelineStateWithFunction:f error:&err];require(p!=nil,err.description.UTF8String);cache[name]=p;
        emit(@{@"event":@"pipeline",@"name":name,@"static_tgm":@(p.staticThreadgroupMemoryLength),@"max_threads":@(p.maxTotalThreadsPerThreadgroup),@"simd":@(p.threadExecutionWidth)});return p;
    }
    NSString *unpack_name(NSString *dtype, bool fast, bool vec){
        NSString *suffix=vec?@"_vec":fast?@"_fast":@"";
        return [NSString stringWithFormat:@"unpack_%@%@",dtype,suffix];
    }
    void unpack(Fixture &f, NSString *dtype, bool fast, bool vec, id<MTLBuffer> dst){
        auto cmd=[q commandBuffer];auto enc=[cmd computeCommandEncoder];auto p=pipeline(unpack_name(dtype,fast,vec));
        [enc setComputePipelineState:p];[enc setBuffer:f.w offset:0 atIndex:0];[enc setBuffer:dst offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(uint64_t(f.experts)*f.a.M*f.a.K/16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,cmd.error.description.UTF8String);
    }
    double unpack_time(Fixture &f, NSString *dtype, bool fast, bool vec, int reps){
        auto cmd=[q commandBuffer];auto enc=[cmd computeCommandEncoder];auto p=pipeline(unpack_name(dtype,fast,vec));
        [enc setComputePipelineState:p];[enc setBuffer:f.w offset:0 atIndex:0];[enc setBuffer:f.decoded offset:0 atIndex:1];
        for(int rep=0;rep<reps;++rep)
            [enc dispatchThreads:MTLSizeMake(uint64_t(f.experts)*f.a.M*f.a.K/16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,cmd.error.description.UTF8String);
        return (cmd.GPUEndTime-cmd.GPUStartTime)*1000/reps;
    }
    double run(Fixture &f, NSString *dtype, int hold, bool baseline, int reps, int only_panel=0, bool fast=false, bool vec=false){
        auto cmd=[q commandBuffer];auto enc=[cmd computeCommandEncoder];float limit=10;int epi=0;
        std::vector<int> panels;
        if(only_panel)panels={only_panel};
        else if(f.half){panels={32};}
        else panels=f.a.N<128?std::vector<int>{32}:std::vector<int>{16,32,64};
        for(int rep=0;rep<reps;++rep)for(int n:panels){
            auto a=f.a;
            if(!only_panel && !f.half && f.a.N>=128){a.min_count=n==16?1:n==32?(f.half?1:17):33;a.max_count=n==16?16:n==32?32:UINT32_MAX/2;}
            NSString *name=vec ? [NSString stringWithFormat:@"probe_%@_n%d_vec",dtype,n]
                               : fast ? [NSString stringWithFormat:@"probe_%@_n%d_fast",dtype,n]
                                      : [NSString stringWithFormat:@"probe_%@_n%d_h%d",dtype,n,hold];
            if(baseline){
                if(f.half && n==64)name=@"kernel_moe_wide_probe_iq2_s_f16act";
                else if(!f.half && n==16)name=@"kernel_glm_iq3_down_n16";
                else if(!f.half && n==64)name=@"kernel_glm_iq3_down_n64";
                else name=[NSString stringWithFormat:@"kernel_moe_down_%@_%@_grouped_slots_tensor",dtype,f.half?@"f16act":@"f32"];
            }
            auto p=pipeline(name);[enc setComputePipelineState:p];[enc setBytes:&a length:sizeof(a) atIndex:0];
            std::array<id<MTLBuffer>,5> binds={f.w,f.x,f.counts,f.ids,f.out};
            for(int j=0;j<5;++j)[enc setBuffer:binds[j] offset:0 atIndex:j+1];
            [enc setBytes:&limit length:4 atIndex:6];[enc setBytes:&epi length:4 atIndex:7];
            [enc setBuffer:f.decoded offset:0 atIndex:8];[enc setBuffer:f.sink offset:0 atIndex:9];
            [enc setThreadgroupMemoryLength:n==16?5120:n==32?8192:16384 atIndex:0];
            int nx=only_panel?int((a.N+n-1)/n):(f.half || a.N<128)?int((a.N+31)/32):n==64?int((a.N+63)/64):1;
            [enc dispatchThreadgroups:MTLSizeMake(nx,(a.M+63)/64,f.experts) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];require(cmd.status==MTLCommandBufferStatusCompleted,cmd.error.description.UTF8String);
        return (cmd.GPUEndTime-cmd.GPUStartTime)*1000/reps;
    }
};
int main(int argc,const char **argv){@autoreleasepool{try{
    require(argc==4,"usage: probe library routes.jsonl smoke|holds|empty");require(getenv("QWEN_METAL_LEASE_FD"),"canonical lease required");
    bool decode=std::string(argv[3])=="decode";
    Probe p(argv[1],!decode); bool smoke=std::string(argv[3])=="smoke"; bool empty_only=std::string(argv[3])=="empty";
    if(decode){
        // M4-safe decoder screen: no tensor ops, only the shared 16-element
        // dequantizers. Bitwise comparison is the gate; timing is diagnostic.
        struct Dec { const char *name; int block; bool half; };
        const Dec formats[]={{"iq2_s",82,true},{"iq2_xs",74,true},{"iq2_xxs",66,true},
                             {"iq3_s",110,false},{"iq3_xxs",98,true}};
        for(const Dec &dec:formats){@autoreleasepool{
            NSString *dtype=@(dec.name);
            Fixture f(p.d,dec.block,dec.half,2048,4096,1,1);
            uint64_t elems=uint64_t(f.experts)*f.a.M*f.a.K;
            p.unpack(f,dtype,false,false,f.decoded);
            bool iq2=[dtype hasPrefix:@"iq2_"];
            for(int variant=0;variant<2;++variant){bool fast=variant==0, vec=variant==1;
                if(vec && !iq2)continue;
                p.unpack(f,dtype,fast,vec,f.decoded_alt);
                uint64_t bits=memcmp(f.decoded.contents,f.decoded_alt.contents,elems*2)!=0
                    ? [&](){ uint64_t n=0; auto a=(const _Float16*)f.decoded.contents, b=(const _Float16*)f.decoded_alt.contents;
                             for(uint64_t i=0;i<elems;++i)n+=memcmp(&a[i],&b[i],2)!=0; return n; }() : 0;
                emit(@{@"event":@"decode_witness",@"dtype":dtype,@"arm":vec?@"vec":@"fast",@"elements":@(elems),@"bit_mismatches":@(bits)});
                require(bits==0,"fast decoder must be bitwise equal");
                for(int pair=0;pair<3;++pair){
                    double slow=p.unpack_time(f,dtype,false,false,4);
                    double candidate=p.unpack_time(f,dtype,fast,vec,4);
                    emit(@{@"event":@"decode_timing",@"dtype":dtype,@"arm":vec?@"vec":@"fast",@"pair":@(pair),@"slow_ms":@(slow),@"fast_ms":@(candidate)});
                }
            }
        }}
        return 0;
    }
    bool screen=std::string(argv[3])=="screen";
    NSString *data=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil];require(data!=nil,"routes input");
    if(screen){
        for(NSString *line in [data componentsSeparatedByString:@"\n"]){
            NSDictionary *r=[NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
            if(![r[@"kind"] isEqual:@"occupancy"] || [r[@"depth"] intValue]!=81920)continue;
            int rows=[r[@"width"] intValue];if(rows!=32 && rows!=161 && rows!=512)continue;
            for(bool iq2:{true,false}){@autoreleasepool{
                NSString *dtype=iq2?@"iq2_s":@"iq3_s";
                Fixture f(p.d,iq2?82:110,iq2,iq2?2048:4096,iq2?4096:2048,288,rows);p.unpack(f,dtype,false,false,f.decoded);
                int layer_index=0;
                for(NSDictionary *layer in r[@"per_layer"]){
                    if(layer_index++>=6)break;
                    int block=[layer[@"block"] intValue];f.buckets(layer[@"distribution"][@"histogram_rows_per_expert"]);
                    // `code` is the hold index except for production/fast/vec.
                    struct Arm{int code;bool production;bool fast;bool vec;const char*label;};
                    std::vector<Arm> arms={{0,true,false,false,"production"},{0,false,false,false,"h0"},
                                           {3,false,false,false,"h3"},{4,false,false,false,"h4"},
                                           {0,false,true,false,"fast"}};
                    if(iq2)arms.push_back({0,false,false,true,"vec"});
                    for(int rep=0;rep<3;++rep)for(const Arm &arm:arms){
                        bool baseline=arm.production, fast=arm.fast, vec=arm.vec;
                        int h=baseline?0:(fast||vec?0:arm.code);
                        double ms=p.run(f,dtype,h,baseline,4,0,fast,vec);
                        emit(@{@"event":@"timing",@"rows":@(rows),@"block":@(block),@"dtype":dtype,@"arm":@(arm.label),@"rep":@(rep),@"ms":@(ms)});
                    }
                }
            }}
        }
        return 0;
    }
    for(NSString *line in [data componentsSeparatedByString:@"\n"]){
        NSDictionary *r=[NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
        if(![r[@"kind"] isEqual:@"occupancy"] || [r[@"depth"] intValue]!=81920)continue;
        int rows=[r[@"width"] intValue];if(rows!=32 && rows!=161 && rows!=256 && rows!=512)continue;
        for(bool iq2:{true,false}){@autoreleasepool{
            NSString *dtype=iq2?@"iq2_s":@"iq3_s";
            Fixture f(p.d,iq2?82:110,iq2,smoke?128:iq2?2048:4096,smoke?256:iq2?4096:2048,288,rows);p.unpack(f,dtype,false,false,f.decoded);
            for(NSDictionary *layer in r[@"per_layer"]){
                // Match only native dominant types; exceptions are retained by the model stage gate.
                int block=[layer[@"block"] intValue];f.buckets(layer[@"distribution"][@"histogram_rows_per_expert"]);
                p.run(f,dtype,0,true,1);std::vector<float> ref((float*)f.out.contents,(float*)f.out.contents+f.a.slot_limit*f.a.M);
                for(int h:{0,2}){
                    p.run(f,dtype,h,false,1);auto v=(float*)f.out.contents;double max=0,delta=0;uint64_t bits=0;
                    for(size_t j=0;j<ref.size();++j){max=std::max(max,double(fabs(ref[j])));delta=std::max(delta,double(fabs(ref[j]-v[j])));bits+=memcmp(&ref[j],&v[j],4)!=0;require(std::isfinite(v[j]),"nonfinite witness");}
                    emit(@{@"event":@"witness",@"rows":@(rows),@"block":@(block),@"dtype":dtype,@"hold":@(h),@"max_abs":@(delta),@"reference_max":@(max),@"bit_mismatches":@(bits)});
                    require(delta<=0.002*max+0.00001,"tensor witness tolerance");
                }
                p.run(f,dtype,0,false,1,0,true);auto fv=(float*)f.out.contents;uint64_t fbits=0;
                for(size_t j=0;j<ref.size();++j){fbits+=memcmp(&ref[j],&fv[j],4)!=0;require(std::isfinite(fv[j]),"nonfinite fast witness");}
                emit(@{@"event":@"witness",@"rows":@(rows),@"block":@(block),@"dtype":dtype,@"arm":@"fast",@"bit_mismatches":@(fbits)});
                require(fbits==0,"fast decoder must be bitwise equal on the tile");
                if(smoke)break;
                // Rotate controls across layers/reps. Four dispatch repetitions per timing sample.
                // Arm 0 production, 1..5 h0..h4, 6 fast decoder, 7 empty grid.
                for(int rep=0;rep<3;++rep)for(int j=0;j<8;++j){int arm=(j+rep*2+block)%8;bool empty=arm==7;if(empty_only && !empty)continue;bool baseline=arm==0 || empty;bool fast=arm==6;int h=baseline?0:(fast?0:arm-1);
                    if(empty)memset(f.counts.contents,0,f.experts*4);
                    double ms=p.run(f,dtype,h,baseline,4,0,fast);
                    if(empty)memcpy(f.counts.contents,f.c.data(),f.experts*4);
                    NSString *label=empty?@"empty":baseline?@"production":fast?@"fast":[NSString stringWithFormat:@"h%d",h];
                    emit(@{@"event":@"timing",@"rows":@(rows),@"block":@(block),@"dtype":dtype,@"arm":label,@"rep":@(rep),@"ms":@(ms)});
                }
            }
        }}
    }
    return 0;
}catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}}}
