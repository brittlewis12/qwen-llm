// CPU build: clang++ -O3 -fobjc-arc -std=c++17 -framework Foundation -framework Metal kda.mm -o /tmp/glm-kda
// Under canonical lease: /tmp/glm-kda kernels/glm53_kda.metal kernels/research/glm_batch_kda.metalh
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <vector>
struct Args {uint32_t heads,rows;float lower,eps;};
int main(int argc,char**argv){@autoreleasepool {
 if(argc!=3||!getenv("QWEN_METAL_LEASE_FD")||fcntl(atoi(getenv("QWEN_METAL_LEASE_FD")),F_GETFD)<0)return 2;
 id<MTLDevice>dev=MTLCreateSystemDefaultDevice();NSError*err=nil;
 NSString*base=[NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
 NSString*extra=[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:&err];
 MTLCompileOptions*opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion3_1;opts.fastMathEnabled=YES;
 id<MTLLibrary>lib=[dev newLibraryWithSource:([base containsString:@"kernel void kernel_glm53_kda_decode_members("] ? base : [base stringByAppendingString:extra]) options:opts error:&err];
 if(!lib){fprintf(stderr,"%s\n",err.description.UTF8String);return 3;}
 id<MTLComputePipelineState>p[2];
 p[0]=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_glm53_kda_decode"] error:&err];
 p[1]=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_glm53_kda_decode_members"] error:&err];
 if(!p[0]||!p[1])return 4;
 id<MTLCommandQueue>queue=[dev newCommandQueue];
 printf("{\"event\":\"header\",\"device\":\"%s\"}\n",dev.name.UTF8String);
 const size_t H=64,D=128,W=H*D,C=9*W,S=H*D*D;
 auto buffer=[&](size_t n,float scale){std::vector<float>x(n);uint32_t seed=881;for(float&v:x){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;v=(float(seed>>8)*(2.f/16777216.f)-1.f)*scale;}return [dev newBufferWithBytes:x.data() length:n*4 options:MTLResourceStorageModeShared];};
 for(uint32_t b:{2u,3u,4u}){
  id<MTLBuffer>in[13];for(int j=1;j<=6;++j)in[j]=buffer(b*(j==5?H:W),.2f);
  for(int j=7;j<=9;++j)in[j]=buffer(W*4,.1f);
  in[10]=buffer(H,.5f);in[11]=buffer(W,.1f);in[12]=buffer(D,1.f);
  std::vector<id<MTLBuffer>>conv[2],state[2],oldout;
  for(uint32_t r=0;r<b;++r){for(int v=0;v<2;++v){conv[v].push_back(buffer(C,.1f));state[v].push_back(buffer(S,.01f));}oldout.push_back(buffer(W,0.f));}
  id<MTLBuffer>out=buffer(b*W,0.f);std::vector<double>times[2];
  for(int rep=-8;rep<32;++rep)for(int order=0;order<2;++order){int route=(rep&1)?1-order:order;
   id<MTLCommandBuffer>cmd=[queue commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];[enc setComputePipelineState:p[route]];
   Args args{uint32_t(H),route?b:1u,-5.f,1e-6f};[enc setBytes:&args length:sizeof(args) atIndex:0];
   [enc setThreadgroupMemoryLength:(5*D+16)*4 atIndex:0];
   for(uint32_t r=0;r<(route?1u:b);++r){
    for(int j=1;j<=12;++j)[enc setBuffer:in[j] offset:(!route&&j<=6)?r*(j==5?H:W)*4:0 atIndex:j];
    [enc setBuffer:conv[route][r] offset:0 atIndex:13];[enc setBuffer:state[route][r] offset:0 atIndex:14];[enc setBuffer:route?out:oldout[r] offset:0 atIndex:15];
    if(route)for(uint32_t member=1;member<4;++member){uint32_t m=member<b?member:0;[enc setBuffer:conv[1][m] offset:0 atIndex:14+2*member];[enc setBuffer:state[1][m] offset:0 atIndex:15+2*member];}
    [enc dispatchThreadgroups:MTLSizeMake(route?b:1,H,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
   }
   [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];if(cmd.status==MTLCommandBufferStatusError){fprintf(stderr,"%s\n",cmd.error.description.UTF8String);return 5;}
   if(rep>=0)times[route].push_back((cmd.GPUEndTime-cmd.GPUStartTime)*1e6);
  }
  for(uint32_t r=0;r<b;++r)if(memcmp(conv[0][r].contents,conv[1][r].contents,C*4)||memcmp(state[0][r].contents,state[1][r].contents,S*4)||memcmp(oldout[r].contents,(char*)out.contents+r*W*4,W*4))return 6;
  double mean[2];for(int r=0;r<2;++r)mean[r]=std::accumulate(times[r].begin(),times[r].end(),0.)/times[r].size();
  printf("{\"event\":\"kda\",\"b\":%u,\"serial_us\":%.9f,\"members_us\":%.9f,\"predicted_34_layers_saved_ms\":%.9f,\"state_conv_output_bitwise\":true}\n",b,mean[0],mean[1],34*(mean[0]-mean[1])/1000.);
 }
 printf("{\"event\":\"complete\"}\n");return 0;
}}
