// IQ3_XXS / IQ3_S mat-mat (W * X^T -> Y^T).
//
// Same 64x32x32 simdgroup_matrix tile as the QK/IQ prompt mat-mat kernels,
// specialized for GGML dense IQ3 blocks.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int IQ3_QK       = 256;
constant constexpr int IQ3_NL       = IQ3_QK / 16;

constant constexpr int NR0_IQ3      = 64;
constant constexpr int NR1_IQ3      = 32;
constant constexpr int NK_IQ3       = 32;
constant constexpr int NL0_IQ3      = NK_IQ3 / 16;
constant constexpr int NL1_IQ3      = NK_IQ3 / 8;
constant constexpr int NW_IQ3       = 32;
constant constexpr int NSG_IQ3      = 4;
constant constexpr int NTH_IQ3      = NW_IQ3 * NSG_IQ3;

constant uchar kmask_iq3_mm[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

constant uchar ksigns_iq3_mm[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant uint iq3xxs_grid_mm[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

constant uint iq3s_grid_mm[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

struct mat_mat_iq3_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

struct ds4_packed_grouped_down_args {
    uint M;
    uint K;
    uint nb01;
    uint stride_b;
    uint n_expert;
    uint top_k;
    uint n_tokens;
};

struct ds4_packed_grouped_projection_args {
    uint M;
    uint K;
    uint nb01;
    uint stride_b;
    uint n_expert;
    uint map_count;
    uint source_count;
    uint destination_count;
};

struct ds4_packed_grouped_swiglu_iq3_args {
    uint M;
    uint K;
    uint nb01;
    uint stride_b;
    uint n_expert;
    uint map_count;
    uint source_count;
    uint destination_count;
    float clamp;
};

struct ds4_packed_expert_tile {
    uint expert;
    uint start;
    uint count;
};

static_assert(sizeof(ds4_packed_expert_tile) == 12);

inline void dequantize_iq3_xxs_half(device const uchar * blk_bytes,
                                    short il,
                                    thread half4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs = blk_bytes + 2;
    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * q3 = qs + 8u * ib32;
    device const ushort * gas = (device const ushort *)(qs + IQ3_QK / 4) + 2u * ib32;
    const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 16);
    const float dl = float(d_h) * (0.5f + float(aux32 >> 28)) * 0.5f;

    constant uchar * grid1 = (constant uchar *)(iq3xxs_grid_mm + q3[4u * ih + 0u]);
    constant uchar * grid2 = (constant uchar *)(iq3xxs_grid_mm + q3[4u * ih + 1u]);
    uint signs = uint(ksigns_iq3_mm[(aux32 >> (14u * ih)) & 127u]);
    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((signs & uint(kmask_iq3_mm[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((signs & uint(kmask_iq3_mm[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[0][i] = (half)(dl * float(grid1[i]) * s1);
        reg[1][i] = (half)(dl * float(grid2[i]) * s2);
    }

    grid1 = (constant uchar *)(iq3xxs_grid_mm + q3[4u * ih + 2u]);
    grid2 = (constant uchar *)(iq3xxs_grid_mm + q3[4u * ih + 3u]);
    signs = uint(ksigns_iq3_mm[(aux32 >> (14u * ih + 7u)) & 127u]);
    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((signs & uint(kmask_iq3_mm[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((signs & uint(kmask_iq3_mm[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[2][i] = (half)(dl * float(grid1[i]) * s1);
        reg[3][i] = (half)(dl * float(grid2[i]) * s2);
    }
}

inline void dequantize_iq3_s_half(device const uchar * blk_bytes,
                                  short il,
                                  thread half4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs_base = blk_bytes + 2;
    device const uchar * qh = qs_base + IQ3_QK / 4;
    device const uchar * signs_base = qh + IQ3_QK / 32;
    device const uchar * scales = signs_base + IQ3_QK / 8;

    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * qs = qs_base + 8u * ib32;
    device const uchar * signs = signs_base + 4u * ib32 + 2u * ih;
    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const float dl = float(d_h) * (1.0f + 2.0f * float(scale));

    constant uchar * grid1 = (constant uchar *)(iq3s_grid_mm +
        (uint(qs[4u * ih + 0u]) | ((qh_lane << 8u) & 256u)));
    constant uchar * grid2 = (constant uchar *)(iq3s_grid_mm +
        (uint(qs[4u * ih + 1u]) | ((qh_lane << 7u) & 256u)));
    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[0]) & uint(kmask_iq3_mm[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[0]) & uint(kmask_iq3_mm[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[0][i] = (half)(dl * float(grid1[i]) * s1);
        reg[1][i] = (half)(dl * float(grid2[i]) * s2);
    }

    grid1 = (constant uchar *)(iq3s_grid_mm +
        (uint(qs[4u * ih + 2u]) | ((qh_lane << 6u) & 256u)));
    grid2 = (constant uchar *)(iq3s_grid_mm +
        (uint(qs[4u * ih + 3u]) | ((qh_lane << 5u) & 256u)));
    FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[1]) & uint(kmask_iq3_mm[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[1]) & uint(kmask_iq3_mm[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[2][i] = (half)(dl * float(grid1[i]) * s1);
        reg[3][i] = (half)(dl * float(grid2[i]) * s2);
    }
}

kernel void kernel_mat_mat_iq3_xxs_f32_mm(
        constant mat_mat_iq3_args & args [[buffer(0)]],
        device const uchar        * srcA [[buffer(1)]],
        device const float        * srcB [[buffer(2)]],
        device       float        * dst  [[buffer(3)]],
        threadgroup  uchar        * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_IQ3;
    const int r1 = tgpig.x * NR1_IQ3;
    const short nr0 = ((int)args.M - r0 < NR0_IQ3) ? (short)((int)args.M - r0) : NR0_IQ3;
    const short nr1 = ((int)args.N - r1 < NR1_IQ3) ? (short)((int)args.N - r1) : NR1_IQ3;

    const short lr0 = ((short)tiitg / NL0_IQ3) < nr0
                        ? ((short)tiitg / NL0_IQ3)
                        : nr0 - 1;
    const short il0 = tiitg % NL0_IQ3;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_IQ3) < nr1
                        ? ((short)tiitg / NL1_IQ3)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ3);

    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ3) {
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ3;
            const short sy = (tiitg / NL1_IQ3) / 8;
            const short ly = (tiitg / NL1_IQ3) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + 98 : x_ptr;
        y_ptr += NK_IQ3;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0_IQ3 <= (int)args.M && r1 + NR1_IQ3 <= (int)args.N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 16 * (sgitg >> 1)) * args.M;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                            args.M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (16 * (sgitg >> 1)) * NR0_IQ3;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                            NR0_IQ3, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_IQ3) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + j * NR0_IQ3;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}

kernel void kernel_deepseek_v4_packed_grouped_down_iq3_xxs_f32_mm(
        constant ds4_packed_grouped_down_args & args [[buffer(0)]],
        device const uchar * srcA [[buffer(1)]],
        device const float * srcB [[buffer(2)]],
        device const int * slots [[buffer(3)]],
        constant ds4_packed_expert_tile * tiles [[buffer(4)]],
        device float * dst [[buffer(5)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const ds4_packed_expert_tile tile = tiles[tgpig.x];
    if (tile.expert >= args.n_expert || tile.count == 0u || tile.count > 32u
            || tile.start + tile.count > args.n_tokens * args.top_k) return;

    const int r0 = tgpig.y * NR0_IQ3;
    const int nr0 = min((int)NR0_IQ3, (int)args.M - r0);
    const short nr1 = (short)tile.count;

    const short lr0 = ((short)tiitg / NL0_IQ3) < nr0
                        ? ((short)tiitg / NL0_IQ3)
                        : (short)nr0 - 1;
    const short il0 = tiitg % NL0_IQ3;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_IQ3) < nr1
                        ? ((short)tiitg / NL1_IQ3)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ3);
    const int input_slot = slots[tile.start + uint(lr1)];

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + (ulong)tile.expert * expert_stride
        + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * input_slot
        + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ3) {
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ3;
            const short sy = (tiitg / NL1_IQ3) / 8;
            const short ly = (tiitg / NL1_IQ3) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + 98 : x_ptr;
        y_ptr += NK_IQ3;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_IQ3;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                        NR0_IQ3, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        threadgroup float * tile_output = (threadgroup float *)shmem;
        for (int j = tiitg; j < nr1; j += NR1_IQ3) {
            const int output_slot = slots[tile.start + uint(j)];
            device float * D = dst + (ulong)output_slot * args.M + r0;
            threadgroup float * C = tile_output + j * NR0_IQ3;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_deepseek_v4_packed_grouped_mapped_iq3_xxs_f32_mm(
        constant ds4_packed_grouped_projection_args & args [[buffer(0)]],
        device const uchar * srcA [[buffer(1)]],
        device const float * srcB [[buffer(2)]],
        device const int * source_rows [[buffer(3)]],
        device const int * destination_slots [[buffer(4)]],
        constant ds4_packed_expert_tile * tiles [[buffer(5)]],
        device float * dst [[buffer(6)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const ds4_packed_expert_tile tile = tiles[tgpig.x];
    if (tile.expert >= args.n_expert || tile.count == 0u || tile.count > 32u
            || tile.start + tile.count > args.map_count) return;

    threadgroup uint * map_valid = (threadgroup uint *)shmem;
    if (tiitg == 0) {
        uint valid = 1u;
        for (uint j = 0; j < tile.count; ++j) {
            const int source = source_rows[tile.start + j];
            const int destination = destination_slots[tile.start + j];
            if (source < 0 || uint(source) >= args.source_count
                    || destination < 0 || uint(destination) >= args.destination_count) {
                valid = 0u;
            }
        }
        *map_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*map_valid == 0u) return;

    const int r0 = tgpig.y * NR0_IQ3;
    const int nr0 = min((int)NR0_IQ3, (int)args.M - r0);
    const short nr1 = (short)tile.count;

    const short lr0 = ((short)tiitg / NL0_IQ3) < nr0
                        ? ((short)tiitg / NL0_IQ3)
                        : (short)nr0 - 1;
    const short il0 = tiitg % NL0_IQ3;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_IQ3) < nr1
                        ? ((short)tiitg / NL1_IQ3)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ3);
    const int input_row = source_rows[tile.start + uint(lr1)];

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + (ulong)tile.expert * expert_stride
        + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * input_row
        + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ3) {
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ3;
            const short sy = (tiitg / NL1_IQ3) / 8;
            const short ly = (tiitg / NL1_IQ3) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + 98 : x_ptr;
        y_ptr += NK_IQ3;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_IQ3;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                        NR0_IQ3, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        threadgroup float * tile_output = (threadgroup float *)shmem;
        for (int j = tiitg; j < nr1; j += NR1_IQ3) {
            const int output_slot = destination_slots[tile.start + uint(j)];
            device float * D = dst + (ulong)output_slot * args.M + r0;
            threadgroup float * C = tile_output + j * NR0_IQ3;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_deepseek_v4_packed_grouped_mapped_swiglu_iq3_xxs_f32_mm(
        constant ds4_packed_grouped_swiglu_iq3_args & args [[buffer(0)]],
        device const uchar * gateA [[buffer(1)]],
        device const uchar * upA [[buffer(2)]],
        device const float * srcB [[buffer(3)]],
        device const int * source_rows [[buffer(4)]],
        device const int * destination_slots [[buffer(5)]],
        constant ds4_packed_expert_tile * tiles [[buffer(6)]],
        device float * gate_dst [[buffer(7)]],
        device float * up_dst [[buffer(8)]],
        device float * inner_dst [[buffer(9)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const ds4_packed_expert_tile tile = tiles[tgpig.x];
    if (tile.expert >= args.n_expert || tile.count == 0u || tile.count > 32u
            || tile.start + tile.count > args.map_count) return;

    threadgroup uint * map_valid = (threadgroup uint *)shmem;
    if (tiitg == 0) {
        uint valid = 1u;
        for (uint j = 0; j < tile.count; ++j) {
            const int source = source_rows[tile.start + j];
            const int destination = destination_slots[tile.start + j];
            if (source < 0 || uint(source) >= args.source_count
                    || destination < 0 || uint(destination) >= args.destination_count) {
                valid = 0u;
            }
        }
        *map_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*map_valid == 0u) return;

    const int r0 = tgpig.y * NR0_IQ3;
    const int nr0 = min((int)NR0_IQ3, (int)args.M - r0);
    const short nr1 = (short)tile.count;

    const short lr0 = ((short)tiitg / NL0_IQ3) < nr0
                        ? ((short)tiitg / NL0_IQ3)
                        : (short)nr0 - 1;
    const short il0 = tiitg % NL0_IQ3;
    short gate_il = il0;
    short up_il = il0;

    const short lr1 = ((short)tiitg / NL1_IQ3) < nr1
                        ? ((short)tiitg / NL1_IQ3)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ3);
    const int input_row = source_rows[tile.start + uint(lr1)];

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * gate_ptr = gateA + (ulong)tile.expert * expert_stride
        + (ulong)args.nb01 * (r0 + lr0);
    device const uchar * up_ptr = upA + (ulong)tile.expert * expert_stride
        + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * input_row
        + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 gate_mc[8];
    simdgroup_float8x8 up_mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        gate_mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        up_mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ3) {
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half(gate_ptr, gate_il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ3;
            const short sy = (tiitg / NL1_IQ3) / 8;
            const short ly = (tiitg / NL1_IQ3) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        gate_il = (gate_il + 2 < IQ3_NL) ? gate_il + 2 : gate_il % 2;
        gate_ptr = (gate_il < 2) ? gate_ptr + 98 : gate_ptr;
        y_ptr += NK_IQ3;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(
                    gate_mc[i], mb[i / 4], ma[i % 4], gate_mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }

        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half(up_ptr, up_il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        up_il = (up_il + 2 < IQ3_NL) ? up_il + 2 : up_il % 2;
        up_ptr = (up_il < 2) ? up_ptr + 98 : up_ptr;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        lsma = sa + 4 * 64 * (sgitg % 2);
        lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(
                    up_mc[i], mb[i / 4], ma[i % 4], up_mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_IQ3;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(gate_mc[i],
                        temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                        NR0_IQ3, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        threadgroup float * tile_output = (threadgroup float *)shmem;
        for (int j = tiitg; j < nr1; j += NR1_IQ3) {
            const int output_slot = destination_slots[tile.start + uint(j)];
            device float * D = gate_dst + (ulong)output_slot * args.M + r0;
            threadgroup float * C = tile_output + j * NR0_IQ3;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(up_mc[i],
                        temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                        NR0_IQ3, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        threadgroup float * tile_output = (threadgroup float *)shmem;
        for (int j = tiitg; j < nr1; j += NR1_IQ3) {
            const int output_slot = destination_slots[tile.start + uint(j)];
            device float * D = up_dst + (ulong)output_slot * args.M + r0;
            threadgroup float * C = tile_output + j * NR0_IQ3;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_IQ3) {
            const int output_slot = destination_slots[tile.start + uint(j)];
            device const float * G = gate_dst + (ulong)output_slot * args.M + r0;
            device const float * U = up_dst + (ulong)output_slot * args.M + r0;
            device float * D = inner_dst + (ulong)output_slot * args.M + r0;
            for (int i = 0; i < nr0; ++i) {
                const float clamped_gate = min(G[i], args.clamp);
                const float clamped_up = clamp(U[i], -args.clamp, args.clamp);
                D[i] = clamped_gate / (1.0f + exp(-clamped_gate)) * clamped_up;
            }
        }
    }
}

kernel void kernel_mat_mat_iq3_s_f32_mm(
        constant mat_mat_iq3_args & args [[buffer(0)]],
        device const uchar        * srcA [[buffer(1)]],
        device const float        * srcB [[buffer(2)]],
        device       float        * dst  [[buffer(3)]],
        threadgroup  uchar        * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_IQ3;
    const int r1 = tgpig.x * NR1_IQ3;
    const short nr0 = ((int)args.M - r0 < NR0_IQ3) ? (short)((int)args.M - r0) : NR0_IQ3;
    const short nr1 = ((int)args.N - r1 < NR1_IQ3) ? (short)((int)args.N - r1) : NR1_IQ3;

    const short lr0 = ((short)tiitg / NL0_IQ3) < nr0
                        ? ((short)tiitg / NL0_IQ3)
                        : nr0 - 1;
    const short il0 = tiitg % NL0_IQ3;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_IQ3) < nr1
                        ? ((short)tiitg / NL1_IQ3)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ3);

    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ3) {
        {
            half4x4 temp_a;
            dequantize_iq3_s_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ3) / 8;
                const short lx = (tiitg / NL0_IQ3) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ3;
            const short sy = (tiitg / NL1_IQ3) / 8;
            const short ly = (tiitg / NL1_IQ3) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + 110 : x_ptr;
        y_ptr += NK_IQ3;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ3 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0_IQ3 <= (int)args.M && r1 + NR1_IQ3 <= (int)args.N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 16 * (sgitg >> 1)) * args.M;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                            args.M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (16 * (sgitg >> 1)) * NR0_IQ3;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_IQ3 * (i / 4),
                            NR0_IQ3, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_IQ3) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + j * NR0_IQ3;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
