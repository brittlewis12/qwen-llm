#include <metal_stdlib>
using namespace metal;
// Per-SIMD quant accumulation is copied from the incumbent mat-vec kernels.
// Four SIMD groups share each dispatch group; matrices remain independent.
constant constexpr int QK_K=256;
constant constexpr int Q4K_BYTES=144, Q5K_BYTES=176, Q6K_BYTES=210;
#define NR0_Q4K 2
#define NR0_Q5K 1
#define NR0_Q6K_R1 1
struct mat_vec_args { uint n_in; uint n_out; };
struct block_iq4_xs_local { half d; ushort scales_h; uchar scales_l[4]; uchar qs[128]; };
constant float iq4nl_values[16]={-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};

inline void fn_q4(uint simd_index, ushort tiisg, device const uchar * raw, device const float * x, device float * y, threadgroup float * shmem) {
 const mat_vec_args args={2560,640};
 device const uchar *weight=(device const uchar *)raw;
 threadgroup float *lut=shmem;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    // First output row this simdgroup is responsible for.
    const uint first_row = simd_index * 2;
    if (first_row >= args.n_out) return;

    // Per-row weight stride (in bytes). nblocks_per_row * Q4K_BYTES.
    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;

    // Pointer to the first row's super-block array.
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    // y starts at the iq-th half (offset 64*iq within super-block) plus
    // 8*ir elements into that half.
    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        // Load 32 y values per lane (two halves of 16, four 8-element strides).
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // For each output row in NR0, do its scale/quant work for this
        // super-block.
        for (short row = 0; row < NR0_Q4K; row++) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = row0 + row * row_stride_bytes
                                      + (ulong)ib * Q4K_BYTES;
            device const half     * dh = (device const half *) blk;
            device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
            device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
            device const uint16_t * q2 = q1 + 32;

            // Decode the four 6-bit (sc, min) pairs that this lane needs.
            sc16[0] =  sc[0]                & kmask1;
            sc16[1] =  sc[2]                & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Unrolled accumulate: pair y with raw 4-bit nibbles in 4 packed lanes.
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += (float)dh[0] * (
                  (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
            ) - (float)dh[1] * (
                  sumy[0] * sc8[2] + sumy[1] * sc8[3]
                + sumy[2] * sc8[6] + sumy[3] * sc8[7]
            );
        }

        y4 += 4 * QK_K;
    }

    // Final reduction across the 32 lanes for each row.
    for (short row = 0; row < NR0_Q4K; row++) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }

}

inline void fn_q5(uint simd_index, ushort tiisg, device const uchar * raw, device const float * x, device float * y, threadgroup float * shmem) {
 const mat_vec_args args={2560,640};
 device const uchar *weight=(device const uchar *)raw;
 threadgroup float *lut=shmem;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // Lane mapping (matches llama.cpp Q5_K kernel):
    //   tid = tiisg/4 in [0, 8)   — picks 1 of 8 (iq, ir) pairs
    //   ix  = tiisg%4 in [0, 4)   — picks 1 of 4 super-blocks per iter
    //   iq  = tid/4   in [0, 2)   — half-block (0=low 128, 1=high 128)
    //   ir  = tid%4   in [0, 4)   — chunk-of-8 within the half
    const ushort tid = tiisg / 4;
    const ushort ix  = tiisg % 4;
    const ushort iq  = tid / 4;
    const ushort ir  = tid % 4;

    const ushort l0 = 8u * ir;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;

    const uchar hm1 = 1u << (2u * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = simd_index * 1;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;

    float yl[16];
    float yh[16];
    float sumf = 0.0f;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = x + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // For NR0=1 we just have the one row this simdgroup is responsible for.
        device const uchar * blk = weight + (ulong)first_row * row_stride_bytes
                                          + (ulong)i * Q5K_BYTES;
        // Layout: half d, half dmin, u8 scales[12], u8 qh[32], u8 qs[128]
        device const half     * dh = (device const half *) blk;
        device const uint16_t * a  = (device const uint16_t *)(blk + 4) + iq;
        device const uchar    * qh = (blk + 4 + 12) + l0;
        device const uchar    * q1 = (blk + 4 + 12 + 32) + q_offset;
        device const uchar    * q2 = q1 + 64;

        sc16[0] =  a[0]                & kmask1;
        sc16[1] =  a[2]                & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
            acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
            acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
            acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
        }

        sumf += (float)dh[0] * (
              sc8[0] * (acc1[0]        + 16.f * acc2[0])
            + sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1])
            + sc8[4] * (acc1[2]        + 16.f * acc2[2])
            + sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])
        ) - (float)dh[1] * (
              sumy[0] * sc8[2]
            + sumy[1] * sc8[3]
            + sumy[2] * sc8[6]
            + sumy[3] * sc8[7]
        );

        y1 += 4 * QK_K;
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && first_row < args.n_out) {
        y[first_row] = tot;
    }

}

inline void fn_q6(uint simd_index, ushort tiisg, device const uchar * raw, device const float * x, device float * y, threadgroup float * shmem) {
 const mat_vec_args args={2560,640};
 device const uchar *weight=(device const uchar *)raw;
 threadgroup float *lut=shmem;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = simd_index * 1;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    // Lane mapping (matches llama.cpp's Q6_K kernel):
    // tid = tiisg/2  ∈ [0, 16)  picks one of 16 (ip, il) pairs
    // ix  = tiisg%2  ∈ [0, 2)   picks 1 of 2 super-blocks per iter
    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;        // 0 or 1: which 128-element half
    const ushort il  = tid % 8;        // 0..7: 4-element chunk inside that half
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf[NR0_Q6K_R1] = {0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const float * y_blk = x + (ulong)i * QK_K + y_offset;

        // Pre-load 16 y values that this lane needs (4 chunks of 4).
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y_blk[l +  0];
            yl[4*l + 1] = y_blk[l + 32];
            yl[4*l + 2] = y_blk[l + 64];
            yl[4*l + 3] = y_blk[l + 96];
        }

        for (short row = 0; row < NR0_Q6K_R1; ++row) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = row0 + row * row_stride_bytes
                                      + (ulong)i * Q6K_BYTES;
            // Layout in block: ql[128], qh[64], scales[16], d (half).
            device const uchar * q1 = blk + q_offset_l;        // ql start
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = blk + 128 + q_offset_h;
            device const int8_t * sc = (device const int8_t *)(blk + 128 + 64) + is;
            device const half   * dh = (device const half *)(blk + 128 + 64 + 16);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += (float)dh[0] * (
                  sums[0] * (float)sc[0]
                + sums[1] * (float)sc[2]
                + sums[2] * (float)sc[4]
                + sums[3] * (float)sc[6]
            );
        }
    }

    for (short row = 0; row < NR0_Q6K_R1; ++row) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }

}

inline void fn_iq4(uint simd_index, ushort tiisg, device const uchar * raw, device const float * x, device float * y, threadgroup float * shmem) {
 const mat_vec_args args={2560,640};
 device const block_iq4_xs_local *weight=(device const block_iq4_xs_local *)raw;
 threadgroup float *lut=shmem;

    const short NR0 = 2;
    const short NSG = 2;
    const uint first_row = simd_index * 2;
    if (first_row >= args.n_out) return;

    if ((simd_index & 3u) == 0) lut[tiisg] = iq4nl_values[tiisg & 15];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nb = args.n_in / 256u;
    const short ix = tiisg / 16;
    const short it = tiisg % 16;
    const short ib = it / 2;
    const short il = it % 2;
    device const float * yb = x + uint(ix) * 256u + uint(ib) * 32u + uint(il) * 8u;

    float sumf[2] = {0.0f, 0.0f};
    uint aux32[2];
    thread const uchar * q8 = (thread const uchar *)aux32;

    for (uint ibl = uint(ix); ibl < nb; ibl += 2u) {
        device const float4 * y4 = (device const float4 *)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        for (short row = 0; row < NR0; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;
            device const block_iq4_xs_local & b = weight[out_row * nb + ibl];
            device const uint * q4 = (device const uint *)(b.qs + 16 * ib + 8 * il);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            const float4 qf10 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf20 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl0 * qf10;
            acc2 += yl1 * qf20;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            const float4 qf11 = {lut[q8[0]], lut[q8[1]], lut[q8[2]], lut[q8[3]]};
            const float4 qf21 = {lut[q8[4]], lut[q8[5]], lut[q8[6]], lut[q8[7]]};
            acc1 += yl2 * qf11;
            acc2 += yl3 * qf21;
            acc1 += acc2;

            const int ls = int(((uint(b.scales_l[ib / 2]) >> (4 * (ib & 1))) & 0x0f)
                         | (((uint(b.scales_h) >> (2 * ib)) & 3u) << 4)) - 32;
            sumf[row] += float(b.d) * float(ls) * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 2u * 256u;
    }

    for (short row = 0; row < NR0; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) y[out_row] = total;
    }

}

inline void fn_router(uint tgpig,ushort sgitg,ushort tiisg,device const float* weight,device const float*x,device float*y,threadgroup float*shmem) {
 const mat_vec_args args={2560,512};

    constexpr ushort NW = 32;
    constexpr ushort NR0 = 2;
    constexpr ushort NSG = 4;

    const uint first_row = tgpig * NR0;
    if (first_row >= args.n_out) return;

    const uint n_in_v4 = args.n_in / 4;
    const uint i0 = uint(sgitg) * NW + uint(tiisg);
    device const float4 * x4 = (device const float4 *)x + i0;

    float sum[NR0] = {0.0f, 0.0f};
    for (uint i = i0; i < n_in_v4; i += NSG * NW) {
        const float4 xv = *x4;
        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            device const float4 * w4 =
                (device const float4 *)(weight + (first_row + row) * args.n_in);
            const float4 wv = w4[i];
            sum[row] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
        x4 += NSG * NW;
    }

    const uint tail_start = n_in_v4 * 4;
    for (uint i = tail_start + uint(sgitg) * NW + uint(tiisg);
         i < args.n_in;
         i += NSG * NW) {
        const float xv = x[i];
        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            sum[row] += weight[(first_row + row) * args.n_in + i] * xv;
        }
    }

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        if (sgitg == 0) row_shmem[tiisg] = 0.0f;
        sum[row] = simd_sum(sum[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        if (tiisg == 0) row_shmem[sgitg] = sum[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
        threadgroup float * row_shmem = shmem + NW * row;
        const float total = simd_sum(row_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            y[first_row + row] = total;
        }
    }

}

inline void fn_scalar(ushort sg,ushort lane,device const float*w,device const float*x,device float*y,threadgroup float*partial) {
 for(uint v=0;v<8;++v) {
  float sum=0.f;
  for(uint i=v*128+uint(sg)*32+lane;i<2560;i+=1024)sum+=w[i]*x[i];
  sum=simd_sum(sum);
  if(lane==0)partial[v*4+sg]=sum;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0) {const float sum=simd_sum(partial[lane]);if(lane==0)y[0]=1.f/(1.f+exp(-sum));}
}

template<int T> inline void fn_quant(uint simd_index,ushort lane,device const uchar*w,device const float*x,device float*y,threadgroup float*s) {
 if(T==0)fn_q4(simd_index,lane,w,x,y,s);
 else if(T==1)fn_q5(simd_index,lane,w,x,y,s);
 else if(T==2)fn_q6(simd_index,lane,w,x,y,s);
 else fn_iq4(simd_index,lane,w,x,y,s);
}
template<int G,int U> inline void fn_packet(uint group,ushort sg,ushort lane,device const uchar*gw,device const uchar*uw,device const float*rw,device const float*sw,device const float*x,device float*projections,device float*router,device float*scale,threadgroup float*s) {
 constexpr uint gr=(G==0||G==3)?2:1,ur=(U==0||U==3)?2:1;
 constexpr uint ng=640/(4*gr),nu=640/(4*ur);
 if(group<ng)fn_quant<G>(group*4+sg,lane,gw,x,projections,s);
 else if(group<ng+nu)fn_quant<U>((group-ng)*4+sg,lane,uw,x,projections+640,s);
 else if(group<ng+nu+256)fn_router(group-ng-nu,sg,lane,rw,x,router,s);
 else fn_scalar(sg,lane,sw,x,scale,s);
}
#define FN_PACKET(NAME,G,U) \
kernel void NAME(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*rw [[buffer(2)]],device const float*sw [[buffer(3)]],device const float*x [[buffer(4)]],device float*projections [[buffer(5)]],device float*router [[buffer(6)]],device float*scale [[buffer(7)]],threadgroup float*s [[threadgroup(0)]],uint group [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {fn_packet<G,U>(group,sg,lane,gw,uw,rw,sw,x,projections,router,scale,s);}

FN_PACKET(kernel_qwen4exp_ffn_projections_0_0,0,0)

FN_PACKET(kernel_qwen4exp_ffn_projections_0_1,0,1)

FN_PACKET(kernel_qwen4exp_ffn_projections_0_2,0,2)

FN_PACKET(kernel_qwen4exp_ffn_projections_0_3,0,3)

FN_PACKET(kernel_qwen4exp_ffn_projections_1_0,1,0)

FN_PACKET(kernel_qwen4exp_ffn_projections_1_1,1,1)

FN_PACKET(kernel_qwen4exp_ffn_projections_1_2,1,2)

FN_PACKET(kernel_qwen4exp_ffn_projections_1_3,1,3)

FN_PACKET(kernel_qwen4exp_ffn_projections_2_0,2,0)

FN_PACKET(kernel_qwen4exp_ffn_projections_2_1,2,1)

FN_PACKET(kernel_qwen4exp_ffn_projections_2_2,2,2)

FN_PACKET(kernel_qwen4exp_ffn_projections_2_3,2,3)

FN_PACKET(kernel_qwen4exp_ffn_projections_3_0,3,0)

FN_PACKET(kernel_qwen4exp_ffn_projections_3_1,3,1)

FN_PACKET(kernel_qwen4exp_ffn_projections_3_2,3,2)

FN_PACKET(kernel_qwen4exp_ffn_projections_3_3,3,3)

// Member-axis fused packet: one flat grid over (projection group, member).
// Each member has private input/projection/router/scale buffers; the four
// shared weights are bound once and served from cache. The per-row projection
// arithmetic and output layout are the one-token kernel's.
#define FN_PACKET_MEMBERS(NAME,G,U) \
kernel void NAME(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*rw [[buffer(2)]],device const float*sw [[buffer(3)]],device const float*x0 [[buffer(4)]],device const float*x1 [[buffer(5)]],device const float*x2 [[buffer(6)]],device const float*x3 [[buffer(7)]],device float*p0 [[buffer(8)]],device float*p1 [[buffer(9)]],device float*p2 [[buffer(10)]],device float*p3 [[buffer(11)]],device float*r0 [[buffer(12)]],device float*r1 [[buffer(13)]],device float*r2 [[buffer(14)]],device float*r3 [[buffer(15)]],device float*s0 [[buffer(16)]],device float*s1 [[buffer(17)]],device float*s2 [[buffer(18)]],device float*s3 [[buffer(19)]],constant uint &groups [[buffer(20)]],threadgroup float*s [[threadgroup(0)]],uint position [[threadgroup_position_in_grid]],ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) { \
    const uint member = position / groups; \
    const uint group = position % groups; \
    device const float*x = member==0?x0:member==1?x1:member==2?x2:x3; \
    device float*projections = member==0?p0:member==1?p1:member==2?p2:p3; \
    device float*router = member==0?r0:member==1?r1:member==2?r2:r3; \
    device float*scale = member==0?s0:member==1?s1:member==2?s2:s3; \
    fn_packet<G,U>(group,sg,lane,gw,uw,rw,sw,x,projections,router,scale,s); \
}

FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_0_0,0,0)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_0_1,0,1)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_0_2,0,2)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_0_3,0,3)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_1_0,1,0)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_1_1,1,1)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_1_2,1,2)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_1_3,1,3)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_2_0,2,0)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_2_1,2,1)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_2_2,2,2)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_2_3,2,3)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_3_0,3,0)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_3_1,3,1)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_3_2,3,2)
FN_PACKET_MEMBERS(kernel_qwen4exp_ffn_projections_members_3_3,3,3)
