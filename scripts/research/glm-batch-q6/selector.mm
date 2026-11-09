// Existing top-512 shader: independent single-query dispatches vs one member grid.
// Build: clang++ -O3 -fobjc-arc -std=c++17 -framework Foundation -framework Metal
//        scripts/research/glm-batch-q6/selector.mm -o /tmp/glm-selector
// Run on M5 under with-metal-lease, passing the normal build's kernels.metallib.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <vector>
struct Args {uint32_t rows, top, queries, ranked;};
int main(int argc,char**argv){@autoreleasepool{
 if(argc!=2||!getenv("QWEN_METAL_LEASE_FD")||fcntl(atoi(getenv("QWEN_METAL_LEASE_FD")),F_GETFD)<0)return 2;
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError*error=nil;
 id<MTLLibrary>lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
 if(!lib){fprintf(stderr,"%s\n",error.description.UTF8String);return 3;}
 id<MTLFunction>fn=[lib newFunctionWithName:@"kernel_glm53_select_top_k_ids_fast"];
 id<MTLComputePipelineState>pso=[dev newComputePipelineStateWithFunction:fn error:&error];
 if(!pso||pso.maxTotalThreadsPerThreadgroup<256)return 4;
 id<MTLCommandQueue>queue=[dev newCommandQueue];
 printf("{\"event\":\"header\",\"device\":\"%s\"}\n",dev.name.UTF8String);
 for(uint32_t n:{2048u,8192u,16384u})for(uint32_t b:{2u,3u,4u}){
  std::vector<float>scores(n*b);uint32_t rng=881;
  for(float&x:scores){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=float(rng>>8)*(2.f/16777216.f)-1.f;}
  std::vector<int32_t>visible(b);for(uint32_t q=0;q<b;++q)visible[q]=n-q*17;
  id<MTLBuffer>s=[dev newBufferWithBytes:scores.data() length:scores.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer>v=[dev newBufferWithBytes:visible.data() length:b*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer>ids=[dev newBufferWithLength:b*512*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer>counts=[dev newBufferWithLength:b*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer>status=[dev newBufferWithLength:b*4 options:MTLResourceStorageModeShared];
  std::vector<id<MTLBuffer>>oldids,oldcounts,oldstatus;
  for(uint32_t q=0;q<b;++q){oldids.push_back([dev newBufferWithLength:512*4 options:MTLResourceStorageModeShared]);oldcounts.push_back([dev newBufferWithLength:4 options:MTLResourceStorageModeShared]);oldstatus.push_back([dev newBufferWithLength:4 options:MTLResourceStorageModeShared]);}
  std::vector<double>times[2];
  for(int rep=-8;rep<32;++rep){
   for(int order=0;order<2;++order){const int route=(rep&1)?1-order:order;
    id<MTLCommandBuffer>cmd=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];[enc setThreadgroupMemoryLength:1024 atIndex:0];[enc setThreadgroupMemoryLength:1024 atIndex:1];
    Args args{n,512,route?b:1u,0};[enc setBytes:&args length:sizeof(args) atIndex:0];
    for(uint32_t q=0;q<(route?1u:b);++q){
     [enc setBuffer:s offset:route?0:q*n*4 atIndex:1];[enc setBuffer:v offset:route?0:q*4 atIndex:2];
     [enc setBuffer:route?ids:oldids[q] offset:0 atIndex:3];[enc setBuffer:route?counts:oldcounts[q] offset:0 atIndex:4];[enc setBuffer:route?status:oldstatus[q] offset:0 atIndex:5];
     [enc dispatchThreadgroups:MTLSizeMake(route?b:1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    }
    [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
    if(cmd.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",cmd.error.description.UTF8String);return 5;}
    if(rep>=0)times[route].push_back((cmd.GPUEndTime-cmd.GPUStartTime)*1e6);
   }
   for(uint32_t q=0;q<b;++q){
    if(memcmp(oldids[q].contents,(char*)ids.contents+q*512*4,512*4)||memcmp(oldcounts[q].contents,(char*)counts.contents+q*4,4)||memcmp(oldstatus[q].contents,(char*)status.contents+q*4,4))return 6;
   }
  }
  double mean[2];for(int a=0;a<2;++a)mean[a]=std::accumulate(times[a].begin(),times[a].end(),0.)/times[a].size();
  printf("{\"event\":\"selector\",\"pools\":%u,\"b\":%u,\"serial_us\":%.9f,\"members_us\":%.9f,\"predicted_11_layers_saved_ms\":%.9f,\"retained_per_route\":32,\"bitwise\":true}\n",n,b,mean[0],mean[1],11*(mean[0]-mean[1])/1000);
 }
 printf("{\"event\":\"complete\"}\n");return 0;
}}
