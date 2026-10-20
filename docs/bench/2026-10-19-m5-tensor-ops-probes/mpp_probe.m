// Host harness for mpp_mm.metal. Compiles the MSL source at runtime with the
// requested language version, times each kernel over R back-to-back
// dispatches using GPU timestamps, and checks sampled outputs against a
// float64 CPU reference computed from the exact operand values.
//
// Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation mpp_probe.m -o mpp_probe
// Run:   ./mpp_probe mpp_mm.metal 41 [kernel-substring]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int M, N, K; } Dims;
typedef enum { T_F16, T_BF16, T_F32, T_I8, T_I32, T_MXFP4 } Ty;

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}
static double rnd_normal(void) {
  double u1 = (rnd32() + 1.0) / 4294967297.0, u2 = (rnd32() + 1.0) / 4294967297.0;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static uint16_t f_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
static float bf16_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static size_t esize(Ty t) { return t == T_F16 || t == T_BF16 ? 2 : t == T_F32 || t == T_I32 ? 4 : 1; }

// Fill a buffer with values and return the exact double value of element i.
static void fill(void *p, Ty t, size_t n, double scale) {
  for (size_t i = 0; i < n; i++) {
    double v = rnd_normal() * scale;
    switch (t) {
      case T_F16: ((__fp16 *)p)[i] = (__fp16)v; break;
      case T_BF16: ((uint16_t *)p)[i] = f_to_bf16((float)v); break;
      case T_F32: ((float *)p)[i] = (float)v; break;
      case T_I8: { int x = (int)lround(v * 40); x = x > 127 ? 127 : x < -127 ? -127 : x; ((int8_t *)p)[i] = (int8_t)x; break; }
      default: break;
    }
  }
}
static double val(const void *p, Ty t, size_t i) {
  switch (t) {
    case T_F16: return ((const __fp16 *)p)[i];
    case T_BF16: return bf16_to_f(((const uint16_t *)p)[i]);
    case T_F32: return ((const float *)p)[i];
    case T_I8: return ((const int8_t *)p)[i];
    case T_I32: return ((const int32_t *)p)[i];
    default: return 0;
  }
}
static const double E2M1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
static double fp4(uint8_t code) { double v = E2M1[code & 7]; return (code & 8) ? -v : v; }
// order 0: element 2j in low nibble; order 1: element 2j in high nibble.
static double mx_w(const uint8_t *q, const uint8_t *s, int K, int n, int k, int order) {
  uint8_t b = q[(size_t)n * (K / 2) + k / 2];
  uint8_t code = ((k & 1) ^ order) ? (b >> 4) : (b & 15);
  return fp4(code) * ldexp(1.0, (int)s[(size_t)n * (K / 32) + k / 32] - 127);
}

typedef struct { const char *name; Ty a, w, c; int tm, tn; } Kern;

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc < 3) { fprintf(stderr, "usage: %s src.metal 40|41 [filter]\n", argv[0]); return 2; }
    const char *filter = argc > 3 ? argv[3] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("{\"probe\":\"device\",\"name\":\"%s\",\"metal4\":%d,\"apple10\":%d}\n", dev.name.UTF8String,
           (int)[dev supportsFamily:MTLGPUFamilyMetal4], (int)[dev supportsFamily:MTLGPUFamilyApple10]);
    NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // Raw values so this builds against SDKs that predate the 4.1 constant.
    opts.languageVersion = (MTLLanguageVersion)(atoi(argv[2]) >= 41 ? 0x40001 : 0x40000);
    NSError *err = nil;
    NSDate *t0 = [NSDate date];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("{\"probe\":\"compile\",\"seconds\":%.2f}\n", -[t0 timeIntervalSinceNow]);
    id<MTLCommandQueue> q = [dev newCommandQueue];

    Kern ks[] = {
      {"f16_64x64_relaxed", T_F16, T_F16, T_F32, 64, 64},
      {"f16_64x64_strict", T_F16, T_F16, T_F32, 64, 64},
      {"f16_128x128_relaxed", T_F16, T_F16, T_F32, 128, 128},
      {"f16_128x64_relaxed", T_F16, T_F16, T_F32, 128, 64},
      {"bf16_64x64_relaxed", T_BF16, T_BF16, T_F32, 64, 64},
      {"f32_64x64_relaxed", T_F32, T_F32, T_F32, 64, 64},
      {"f32_64x64_strict", T_F32, T_F32, T_F32, 64, 64},
      {"f32a_f16w_64x64_strict", T_F32, T_F16, T_F32, 64, 64},
      {"f16a_i8w_64x64_relaxed", T_F16, T_I8, T_F32, 64, 64},
      {"i8_64x64", T_I8, T_I8, T_I32, 64, 64},
      {"i8_128x128", T_I8, T_I8, T_I32, 128, 128},
      {"mxfp4_64x64_relaxed", T_F16, T_MXFP4, T_F32, 64, 64},
      {"mxfp4_64x64_strict", T_F16, T_MXFP4, T_F32, 64, 64},
    };
    Dims shapes[] = {{1024, 2304, 5120}, {1024, 5120, 8192}, {2048, 4096, 4096}, {256, 2304, 5120}};
    const int R = 20, SAMPLES = 512;

    for (size_t ki = 0; ki < sizeof ks / sizeof *ks; ki++) {
      Kern k = ks[ki];
      if (*filter && !strstr(k.name, filter)) continue;
      id<MTLFunction> fn = [lib newFunctionWithName:@(k.name)];
      if (!fn) { printf("{\"probe\":\"mm\",\"kernel\":\"%s\",\"skipped\":\"not in library\"}\n", k.name); continue; }
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) { printf("{\"probe\":\"mm\",\"kernel\":\"%s\",\"error\":\"%s\"}\n", k.name, err.localizedDescription.UTF8String); continue; }
      NSUInteger threads = pso.threadExecutionWidth * 4;
      for (size_t si = 0; si < sizeof shapes / sizeof *shapes; si++) {
        Dims d = shapes[si];
        size_t na = (size_t)d.M * d.K, nw = (size_t)d.N * d.K, nc = (size_t)d.M * d.N;
        size_t wbytes = k.w == T_MXFP4 ? nw / 2 : nw * esize(k.w);
        id<MTLBuffer> A = [dev newBufferWithLength:na * esize(k.a) options:MTLResourceStorageModeShared];
        id<MTLBuffer> W = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> C = [dev newBufferWithLength:nc * esize(k.c) options:MTLResourceStorageModeShared];
        id<MTLBuffer> S = k.w == T_MXFP4 ? [dev newBufferWithLength:(size_t)d.N * (d.K / 32) options:MTLResourceStorageModeShared] : nil;
        fill(A.contents, k.a, na, 1.0);
        if (k.w == T_MXFP4) {
          uint8_t *wq = W.contents, *s = S.contents;
          for (size_t i = 0; i < wbytes; i++) wq[i] = (uint8_t)rnd32();
          for (size_t i = 0; i < (size_t)d.N * (d.K / 32); i++) s[i] = (uint8_t)(120 + rnd32() % 6);
        } else {
          fill(W.contents, k.w, nw, k.a == T_I8 ? 1.0 : 1.0 / sqrt(d.K));
        }
        MTLSize grid = MTLSizeMake((d.N + k.tn - 1) / k.tn, (d.M + k.tm - 1) / k.tm, 1);
        void (^encode)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
          [e setComputePipelineState:pso];
          [e setBuffer:A offset:0 atIndex:0];
          [e setBuffer:W offset:0 atIndex:1];
          [e setBuffer:C offset:0 atIndex:2];
          [e setBytes:&d length:sizeof d atIndex:3];
          if (S) [e setBuffer:S offset:0 atIndex:4];
          [e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        };
        // Warm-up and correctness run.
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        encode(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("{\"probe\":\"mm\",\"kernel\":\"%s\",\"error\":\"%s\"}\n", k.name, cb.error.localizedDescription.UTF8String); continue; }
        // Timed run.
        cb = [q commandBuffer];
        e = [cb computeCommandEncoder];
        for (int r = 0; r < R; r++) encode(e);
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / R;
        double tflops = 2.0 * d.M * d.N * d.K / (ms / 1e3) / 1e12;
        // Correctness on sampled outputs.
        double se[2] = {0, 0}, me[2] = {0, 0}, sr = 0;
        int orders = k.w == T_MXFP4 ? 2 : 1;
        for (int s = 0; s < SAMPLES; s++) {
          int m = rnd32() % d.M, n = rnd32() % d.N;
          double got = val(C.contents, k.c, (size_t)m * d.N + n);
          for (int o = 0; o < orders; o++) {
            double ref = 0;
            for (int kk = 0; kk < d.K; kk++) {
              double w = k.w == T_MXFP4 ? mx_w(W.contents, S.contents, d.K, n, kk, o) : val(W.contents, k.w, (size_t)n * d.K + kk);
              ref += val(A.contents, k.a, (size_t)m * d.K + kk) * w;
            }
            if (o == 0) sr += ref * ref;
            double ae = fabs(got - ref);
            se[o] += ae * ae;
            if (ae > me[o]) me[o] = ae;
          }
        }
        double rms_ref = sqrt(sr / SAMPLES);
        printf("{\"probe\":\"mm\",\"kernel\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ms\":%.4f,\"tflops\":%.2f,"
               "\"rel_rms\":%.3e,\"rel_max\":%.3e", k.name, d.M, d.N, d.K, ms, tflops,
               sqrt(se[0] / SAMPLES) / rms_ref, me[0] / rms_ref);
        if (orders == 2) printf(",\"rel_rms_hi_first\":%.3e", sqrt(se[1] / SAMPLES) / rms_ref);
        printf("}\n");
        fflush(stdout);
      }
    }
  }
  return 0;
}
