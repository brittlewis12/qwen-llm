// MoE expert-bank kernels for single-token decode.
//
// The qwen35moe A3B FFN selects top-k experts per token. The dense path can
// bind one 2D weight tensor per mat-vec, but MoE needs the expert id to choose
// a 2D slice from a 3D expert bank while staying GPU-resident.

#include <metal_stdlib>
using namespace metal;

typedef matrix<bfloat, 2, 4> bfloat2x4;

#define QK_K 256
#define QK8_0 32
#define Q4K_BYTES 144
#define Q5K_BYTES 176
#define Q6K_BYTES 210
#define Q8_0_BYTES 34
#define IQ3XXS_BYTES 98
#define IQ3S_BYTES 110
#define IQ4XS_BYTES 136
#define Q4K_NL (QK_K / 16)
#define Q8_0_NL 2

#define NR0_Q4K 2
#define NSG_Q4K 2
#define NR0_Q5K 1
#define NSG_Q5K 2
#define NR0_Q6K 2
#define NSG_Q6K 2
#define NR0_Q80 2
#define NSG_Q80 4
#define NQ_Q80 8
#define NR0_IQ3_FAST 4
#define NSG_IQ3_FAST 2
#define NSG_MOE_F32 4
#define NSG_MOE_IQ3_XXS 2
#define NSG_MOE_IQ4_XS 4

#define IQ3XXS_NL (QK_K / 16)
#define IQ3S_NL (QK_K / 16)
#define Q5K_NL (QK_K / 16)
#define Q6K_NL (QK_K / 16)
#define NR0_MM 64
#define NR1_MM 32
#define NK_MM 32
#define NL0_MM (NK_MM / 16)
#define NL1_MM (NK_MM / 8)

struct moe_q4k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_q5k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_q6k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_q8_0_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_f32_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_iq4xs_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

constant float moe_iq4nl_values[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f,
};


constant uchar moe_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

constant uchar moe_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static inline float moe_bf16_to_float(ushort v) {
    return as_type<float>((uint)v << 16);
}

constant uint moe_iq3xxs_grid[256] = {
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

constant uint moe_iq3s_grid[512] = {
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

struct block_iq4_xs_local {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

struct moe_sum_args {
    uint n_out;
    uint topk;
};

struct axpy_scalar_args {
    uint n;
};

struct topk_logits_args {
    uint n;
    uint k;
};

struct dot_sigmoid_args {
    uint n;
};

struct topk_dot_sigmoid_args {
    uint n_expert;
    uint topk;
    uint hidden;
    uint n_tokens;
};

struct moe_route_bucket_args {
    uint n_expert;
    uint n_tokens;
    uint topk;
};

inline float moe_silu_f(float x) {
    return x / (1.0f + exp(-x));
}

static inline float moe_deq_iq4_xs(device const block_iq4_xs_local & b, uint i) {
    const uint ib32 = i >> 5;
    const uint lane = i & 31u;
    const uint scale_l = (uint(b.scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const uint scale_h = (uint(b.scales_h) >> (2u * ib32)) & 3u;
    const float d = float(b.d) * float(int(scale_l | (scale_h << 4)) - 32);
    const uchar q = b.qs[ib32 * 16u + (lane & 15u)];
    const uint idx = (lane < 16u) ? uint(q & 0x0f) : uint(q >> 4);
    return d * moe_iq4nl_values[idx];
}

static inline float moe_deq_iq3_xxs(device const uchar * blk, uint i) {
    const float d = float(((device const half *)blk)[0]);
    device const uchar * qs = blk + 2;
    const uint ib32 = i >> 5;
    const uint lane = i & 31u;
    const uint il = lane >> 4;
    const uint local = lane & 15u;
    device const uchar * q3 = qs + 8u * ib32;
    device const ushort * gas = (device const ushort *)(qs + QK_K / 4) + 2u * ib32;
    const uint aux32 = uint(gas[0]) | (uint(gas[1]) << 16);
    const float dl = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    const uint grid = moe_iq3xxs_grid[q3[4u * il + (local >> 2)]];
    const float q = float((grid >> (8u * (local & 3u))) & 0xffu);
    const uint sign_bits = (local < 8u)
        ? uint(moe_ksigns_iq2xs[(aux32 >> (14u * il)) & 127u])
        : uint(moe_ksigns_iq2xs[(aux32 >> (14u * il + 7u)) & 127u]);
    const float sign = (sign_bits & uint(moe_kmask_iq2xs[local & 7u])) ? -1.0f : 1.0f;
    return dl * q * sign;
}

inline void dequantize_iq3_xxs_half_grouped(device const uchar * blk_bytes,
                                            short il,
                                            thread half4x4 & reg) {
    const uint base = 16u * (uint)il;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(moe_deq_iq3_xxs(blk_bytes, base + (uint)i));
    }
}

static inline float moe_deq_iq3_s(device const uchar * blk, uint i) {
    const float d = float(((device const half *)blk)[0]);
    device const uchar * qs = blk + 2;
    device const uchar * qh = qs + QK_K / 4;
    device const uchar * signs_base = qh + QK_K / 32;
    device const uchar * scales = signs_base + QK_K / 8;
    const uint ib32 = i >> 5;
    const uint lane = i & 31u;
    const uint il = lane >> 4;
    const uint local = lane & 15u;
    const uint qh_lane = uint(qh[ib32]) >> (4u * il);
    const uint scale = (uint(scales[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const float dl = d * (1.0f + 2.0f * float(scale));
    const uint grid_idx = uint(qs[8u * ib32 + 4u * il + (local >> 2)])
        | ((qh_lane << (8u - (local >> 2))) & 256u);
    const uint grid = moe_iq3s_grid[grid_idx];
    const float q = float((grid >> (8u * (local & 3u))) & 0xffu);
    const uchar signs = signs_base[4u * ib32 + 2u * il + (local >> 3)];
    const float sign = (signs & moe_kmask_iq2xs[local & 7u]) ? -1.0f : 1.0f;
    return dl * q * sign;
}

inline void dequantize_iq3_s_half_grouped(device const uchar * blk_bytes,
                                          short il,
                                          thread half4x4 & reg) {
    const uint base = 16u * (uint)il;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(moe_deq_iq3_s(blk_bytes, base + (uint)i));
    }
}

kernel void kernel_topk_logits_softmax_f32(
        constant topk_logits_args & args [[buffer(0)]],
        device const float       * logits [[buffer(1)]],
        device       int         * out_idx [[buffer(2)]],
        device       float       * out_w   [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint MAX_K = 16;
    if (args.k == 0 || args.k > MAX_K) return;

    int top_idx[MAX_K];
    float top_val[MAX_K];
    for (uint i = 0; i < args.k; ++i) {
        top_idx[i] = -1;
        top_val[i] = -INFINITY;
    }

    for (uint i = 0; i < args.n; ++i) {
        const float v = logits[i];
        for (uint j = 0; j < args.k; ++j) {
            const bool better = (v > top_val[j]) || (v == top_val[j] && (top_idx[j] < 0 || int(i) < top_idx[j]));
            if (better) {
                for (uint m = args.k - 1; m > j; --m) {
                    top_val[m] = top_val[m - 1];
                    top_idx[m] = top_idx[m - 1];
                }
                top_val[j] = v;
                top_idx[j] = int(i);
                break;
            }
        }
    }

    const float max_top = top_val[0];
    float sum = 0.0f;
    float exp_val[MAX_K];
    for (uint i = 0; i < args.k; ++i) {
        exp_val[i] = top_idx[i] >= 0 ? exp(top_val[i] - max_top) : 0.0f;
        sum += exp_val[i];
    }
    sum = max(sum, 6.103515625e-5f);
    for (uint i = 0; i < args.k; ++i) {
        out_idx[i] = max(top_idx[i], 0);
        out_w[i] = exp_val[i] / sum;
    }
}

kernel void kernel_dot_sigmoid_f32(
        constant dot_sigmoid_args & args [[buffer(0)]],
        device const float       * weight [[buffer(1)]],
        device const float       * x      [[buffer(2)]],
        device       float       * out    [[buffer(3)]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint i = tiisg; i < args.n; i += 32) {
        sum += weight[i] * x[i];
    }
    sum = simd_sum(sum);
    if (tiisg == 0) {
        out[0] = 1.0f / (1.0f + exp(-sum));
    }
}

inline bool moe_better_pair(float cand_v, int cand_i, float best_v, int best_i) {
    return cand_i >= 0 && (best_i < 0 || cand_v > best_v || (cand_v == best_v && cand_i < best_i));
}

kernel void kernel_topk_logits_softmax_dot_sigmoid_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        threadgroup  float             * sh_score      [[threadgroup(0)]],
        threadgroup  float             * red_val       [[threadgroup(1)]],
        threadgroup  int               * red_idx       [[threadgroup(2)]],
        uint  tid [[thread_position_in_threadgroup]],
        uint  ntg [[threads_per_threadgroup]]) {
    const uint MAX_K = 16;
    if (args.n_expert > ntg || args.topk == 0 || args.topk > MAX_K) return;

    float shared_sum = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        shared_sum += shared_weight[i] * x[i];
    }
    red_val[tid] = shared_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_val[tid] += red_val[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_out[0] = 1.0f / (1.0f + exp(-red_val[0]));

    sh_score[tid] = tid < args.n_expert ? logits[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.topk; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n_expert ? int(tid) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float cand_v = red_val[tid + stride];
                const int cand_i = red_idx[tid + stride];
                if (moe_better_pair(cand_v, cand_i, red_val[tid], red_idx[tid])) {
                    red_val[tid] = cand_v;
                    red_idx[tid] = cand_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out_idx[slot] = max(red_idx[0], 0);
            out_w[slot] = red_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) == red_idx[0]) sh_score[tid] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float max_top = out_w[0];
        float sum = 0.0f;
        float exp_val[MAX_K];
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w[i] = exp_val[i] / sum;
        }
    }
}

kernel void kernel_topk_logits_softmax_dot_sigmoid_packed_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        threadgroup  float             * sh_score      [[threadgroup(0)]],
        threadgroup  float             * red_val       [[threadgroup(1)]],
        threadgroup  int               * red_idx       [[threadgroup(2)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tid2 [[thread_position_in_threadgroup]],
        uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint token = tgpig.y;
    const uint tid = tid2.x;
    const uint ntg = ntg2.x;
    const uint MAX_K = 16;
    if (args.n_expert > ntg || args.topk == 0 || args.topk > MAX_K) return;

    device const float * logits_t = logits + (ulong)token * args.n_expert;
    device const float * x_t = x + (ulong)token * args.hidden;
    device int * out_idx_t = out_idx + (ulong)token * args.topk;
    device float * out_w_t = out_w + (ulong)token * args.topk;

    float shared_sum = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        shared_sum += shared_weight[i] * x_t[i];
    }
    red_val[tid] = shared_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_val[tid] += red_val[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_out[token] = 1.0f / (1.0f + exp(-red_val[0]));

    sh_score[tid] = tid < args.n_expert ? logits_t[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.topk; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n_expert ? int(tid) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float cand_v = red_val[tid + stride];
                const int cand_i = red_idx[tid + stride];
                if (moe_better_pair(cand_v, cand_i, red_val[tid], red_idx[tid])) {
                    red_val[tid] = cand_v;
                    red_idx[tid] = cand_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out_idx_t[slot] = max(red_idx[0], 0);
            out_w_t[slot] = red_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) == red_idx[0]) sh_score[tid] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float max_top = out_w_t[0];
        float sum = 0.0f;
        float exp_val[MAX_K];
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w_t[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w_t[i] = exp_val[i] / sum;
        }
    }
}

kernel void kernel_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        device atomic_int              * counts        [[buffer(7)]],
        device int                     * ids           [[buffer(8)]],
        threadgroup  float             * sh_score      [[threadgroup(0)]],
        threadgroup  float             * red_val       [[threadgroup(1)]],
        threadgroup  int               * red_idx       [[threadgroup(2)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tid2 [[thread_position_in_threadgroup]],
        uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint token = tgpig.y;
    const uint tid = tid2.x;
    const uint ntg = ntg2.x;
    const uint MAX_K = 16;
    if (args.n_expert > ntg || args.topk == 0 || args.topk > MAX_K) return;

    device const float * logits_t = logits + (ulong)token * args.n_expert;
    device const float * x_t = x + (ulong)token * args.hidden;
    device int * out_idx_t = out_idx + (ulong)token * args.topk;
    device float * out_w_t = out_w + (ulong)token * args.topk;

    float shared_sum = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        shared_sum += shared_weight[i] * x_t[i];
    }
    red_val[tid] = shared_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_val[tid] += red_val[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_out[token] = 1.0f / (1.0f + exp(-red_val[0]));

    sh_score[tid] = tid < args.n_expert ? logits_t[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.topk; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n_expert ? int(tid) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float cand_v = red_val[tid + stride];
                const int cand_i = red_idx[tid + stride];
                if (moe_better_pair(cand_v, cand_i, red_val[tid], red_idx[tid])) {
                    red_val[tid] = cand_v;
                    red_idx[tid] = cand_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out_idx_t[slot] = max(red_idx[0], 0);
            out_w_t[slot] = red_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) == red_idx[0]) sh_score[tid] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float max_top = out_w_t[0];
        float sum = 0.0f;
        float exp_val[MAX_K];
        const uint base = token * args.topk;
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w_t[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w_t[i] = exp_val[i] / sum;
            const int expert = out_idx_t[i];
            if (expert >= 0 && expert < int(args.n_expert)) {
                const int dst = atomic_fetch_add_explicit(&counts[expert], 1, memory_order_relaxed);
                ids[(ulong)expert * args.n_tokens + dst] = int(base + i);
            }
        }
    }
}

kernel void kernel_moe_route_bucket_slots_f32(
        constant moe_route_bucket_args & args [[buffer(0)]],
        device const int              * top_idx [[buffer(1)]],
        device       int              * counts  [[buffer(2)]],
        device       int              * ids     [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n_expert) return;
    int count = 0;
    for (uint token = 0; token < args.n_tokens; ++token) {
        const uint base = token * args.topk;
        for (uint k = 0; k < args.topk; ++k) {
            const int expert_i = top_idx[base + k];
            if (expert_i == int(tid)) {
                ids[(ulong)tid * args.n_tokens + count] = int(base + k);
                count += 1;
            }
        }
    }
    counts[tid] = count;
}

kernel void kernel_moe_swiglu_q4_K_f32(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * w_gate  [[buffer(1)]],
        device const uchar    * w_up    [[buffer(2)]],
        device const float    * x       [[buffer(3)]],
        device const int      * top_idx [[buffer(4)]],
        device       float    * inner   [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_gate = w_gate + (ulong)expert_i * expert_stride_bytes;
    device const uchar * expert_up   = w_up   + (ulong)expert_i * expert_stride_bytes;

    device const uchar * row0_g = expert_gate + (ulong)first_row * row_stride_bytes;
    device const uchar * row0_u = expert_up   + (ulong)first_row * row_stride_bytes;

    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf_g[NR0_Q4K] = {0.f, 0.f};
    float sumf_u[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; row++) {
            if (first_row + row >= args.n_out) break;

            {
                device const uchar * blk = row0_g + row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_g[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }

            {
                device const uchar * blk = row0_u + row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_u[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }
        }
        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0_Q4K; row++) {
        float total_g = simd_sum(sumf_g[row]);
        float total_u = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(total_g) * total_u;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_packed_slots(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * w_gate  [[buffer(1)]],
        device const uchar    * w_up    [[buffer(2)]],
        device const float    * x_pack  [[buffer(3)]],
        device const int      * top_idx [[buffer(4)]],
        device       float    * inner   [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_gate = w_gate + (ulong)expert_i * expert_stride_bytes;
    device const uchar * expert_up   = w_up   + (ulong)expert_i * expert_stride_bytes;

    device const uchar * row0_g = expert_gate + (ulong)first_row * row_stride_bytes;
    device const uchar * row0_u = expert_up   + (ulong)first_row * row_stride_bytes;
    const uint token = slot / args.topk;
    device const float * y4 = x_pack + (ulong)token * args.n_in + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf_g[NR0_Q4K] = {0.f, 0.f};
    float sumf_u[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; ++row) {
            if (first_row + row >= args.n_out) break;

            {
                device const uchar * blk = row0_g + (ulong)row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_g[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }

            {
                device const uchar * blk = row0_u + (ulong)row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_u[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }
        }
        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0_Q4K; row++) {
        float total_g = simd_sum(sumf_g[row]);
        float total_u = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(total_g) * total_u;
        }
    }
}

#define NR1_GROUP_Q4 16
#define NL1_GROUP_Q4 (NK_MM / 8)
#define B_LOAD_THREADS_GROUP_Q4 (NR1_GROUP_Q4 * NL1_GROUP_Q4)
#define NR1_GROUP_Q5 16
#define NL1_GROUP_Q5 (NK_MM / 8)
#define B_LOAD_THREADS_GROUP_Q5 (NR1_GROUP_Q5 * NL1_GROUP_Q5)

struct moe_group_q4k_args {
    uint ffn;
    uint hidden;
    uint n_expert;
    uint topk;
    uint n_tokens;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

inline void dequantize_q4_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + (il / 4) * 32 + 16 * (il & 1);
    short il_inner = il & 3;
    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;

    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(qs[i] & mask) - ml);
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    // Clamp lr1 to nr1-1 so partial-tile threads don't read stale ids[].
    // Mirrors the Q5 grouped kernel's lr1 clamp (line ~1359).
    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

inline void dequantize_q5_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg);
inline void dequantize_q6_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg);
inline void dequantize_q8_0_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg);

kernel void kernel_moe_swiglu_f32_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const float * srcA_gate     [[buffer(1)]],
        device const float * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const ulong expert_stride = (ulong)args.hidden * args.ffn;
    device const float * x_ptr_g = srcA_gate + expert_stride * (ulong)im
                                            + (ulong)args.hidden * (r0 + lr0)
                                            + 16u * (ulong)il0;
    device const float * x_ptr_u = srcA_up   + expert_stride * (ulong)im
                                            + (ulong)args.hidden * (r0 + lr0)
                                            + 16u * (ulong)il0;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = half(x_ptr_g[i]);
            }
        }
        {
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = half(x_ptr_u[i]);
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        x_ptr_g += NK_MM;
        x_ptr_u += NK_MM;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}
kernel void kernel_moe_swiglu_bf16_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const bfloat * srcA_gate     [[buffer(1)]],
        device const bfloat * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat * sa_g = (threadgroup bfloat *)(shmem);
    threadgroup bfloat * sa_u = (threadgroup bfloat *)(shmem + 4096);
    threadgroup bfloat * sb   = (threadgroup bfloat *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const ulong expert_stride = (ulong)args.hidden * args.ffn;
    device const bfloat * x_ptr_g = srcA_gate + expert_stride * (ulong)im
                                            + (ulong)args.hidden * (r0 + lr0)
                                            + 16u * (ulong)il0;
    device const bfloat * x_ptr_u = srcA_up   + expert_stride * (ulong)im
                                            + (ulong)args.hidden * (r0 + lr0)
                                            + 16u * (ulong)il0;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_bfloat8x8 ma_g[4];
    simdgroup_bfloat8x8 ma_u[4];
    simdgroup_bfloat8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = x_ptr_g[i];
            }
        }
        {
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = x_ptr_u[i];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup bfloat2x4 *)(sb + 64 * ib + 8 * ly) =
                (bfloat2x4)(*((device const float2x4 *)y_ptr));
        }

        x_ptr_g += NK_MM;
        x_ptr_u += NK_MM;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const bfloat * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const bfloat * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const bfloat * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / IQ3XXS_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * IQ3XXS_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * IQ3XXS_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_iq3_xxs_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3XXS_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + IQ3XXS_BYTES * ((2 + IQ3XXS_NL - 1) / IQ3XXS_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + IQ3XXS_BYTES * ((2 + IQ3XXS_NL - 1) / IQ3XXS_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}


kernel void kernel_moe_swiglu_iq3_s_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / IQ3S_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * IQ3S_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * IQ3S_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_iq3_s_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_iq3_s_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < IQ3S_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + IQ3S_BYTES * ((2 + IQ3S_NL - 1) / IQ3S_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + IQ3S_BYTES * ((2 + IQ3S_NL - 1) / IQ3S_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}


kernel void kernel_moe_swiglu_q5_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q5K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q5K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q6_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / Q6K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q6K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q6K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q6_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q6_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q8_0_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / Q8_0_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q8_0_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q8_0_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q8_0_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q8_0_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = il % 2;
        x_ptr_g += Q8_0_BYTES;
        x_ptr_u += Q8_0_BYTES;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_fused    [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn * 2;
    device const uchar * x_ptr = srcA_fused + expert_stride * (ulong)im
                                           + (ulong)(args.nb01 * 2) * (r0 + lr0)
                                           + (ulong)offset1 * (2 * Q4K_BYTES);
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr + Q4K_BYTES, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                    ? x_ptr + (2 * Q4K_BYTES) * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[8];
    simdgroup_float8x8 mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 8192))
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * Cg = temp_str_g + (j * NR0_MM);
                threadgroup float * Cu = temp_str_u + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = moe_silu_f(Cg[i]) * Cu[i];
                }
            }
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_fused    [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn * 2;
    device const uchar * x_ptr = srcA_fused + expert_stride * (ulong)im
                                           + (ulong)(args.nb01 * 2) * (r0 + lr0)
                                           + (ulong)offset1 * (2 * Q4K_BYTES);
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[8];
    simdgroup_float8x8 mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr + Q4K_BYTES, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                    ? x_ptr + (2 * Q4K_BYTES) * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 8192))
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * Cg = temp_str_g + (j * NR0_MM);
                threadgroup float * Cu = temp_str_u + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = moe_silu_f(Cg[i]) * Cu[i];
                }
            }
        }
    }
}

kernel void kernel_moe_matmul_q4_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA          [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                          + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc[4];

    for (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = temp_str[(8 * tile_i + mr) + c * NR0_MM];
        }
    }
}

kernel void kernel_moe_matmul_q4_K_f32_grouped_slots_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA          [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                          + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}

struct moe_fused_q4q5_args {
    uint hidden;
    uint ffn;
    uint n_expert;
    uint topk;
};

kernel void kernel_moe_fused_routed_q4q5_token_f32(
        constant moe_fused_q4q5_args & args [[buffer(0)]],
        device const uchar * w_gate        [[buffer(1)]],
        device const uchar * w_up          [[buffer(2)]],
        device const uchar * w_down        [[buffer(3)]],
        device const float * x_pack        [[buffer(4)]],
        device const int   * top_idx       [[buffer(5)]],
        device const float * top_w         [[buffer(6)]],
        device       float * out_pack      [[buffer(7)]],
        threadgroup  float * inner         [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;
    device const float * x = x_pack + (ulong)token * args.hidden;
    device float * out = out_pack + (ulong)token * args.hidden;
    const ulong base_slot = (ulong)token * args.topk;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix_q4 = tiisg / 8;
    const ushort it_q4 = tiisg % 8;
    const ushort iq_q4 = it_q4 / 4;
    const ushort ir_q4 = it_q4 % 4;

    const ushort tid_q5 = tiisg / 4;
    const ushort ix_q5  = tiisg % 4;
    const ushort iq_q5  = tid_q5 / 4;
    const ushort ir_q5  = tid_q5 % 4;

    const ushort l0_q5 = 8u * ir_q5;
    const ushort q_offset_q5 = 32u * iq_q5 + l0_q5;
    const ushort y_offset_q5 = 64u * iq_q5 + l0_q5;

    const uchar hm1 = 1u << (2u * iq_q5);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const uint nb_h = args.hidden / QK_K;
    const uint nb_f = args.ffn / QK_K;
    const ulong gate_row_stride_bytes = (ulong)nb_h * Q4K_BYTES;
    const ulong gate_expert_stride_bytes = (ulong)args.ffn * gate_row_stride_bytes;
    const ulong down_row_stride_bytes = (ulong)nb_f * Q5K_BYTES;
    const ulong down_expert_stride_bytes = (ulong)args.hidden * down_row_stride_bytes;

    float yl_q4[16];
    float yh_q4[16];
    float yl_q5[16];
    float yh_q5[16];
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *) sc16;

    if (tiisg == 0) {
        for (uint i = sgitg; i < args.hidden; i += 8) {
            out[i] = 0.0f;
        }
    }
    // mem_device is required because `out[]` is device memory and is
    // both written above and accumulated into (`out[first_row] += ...`)
    // by other threads of this threadgroup further down. A plain
    // mem_threadgroup barrier only orders threadgroup memory; device
    // writes can still be reordered across it.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint slot_k = 0; slot_k < args.topk; ++slot_k) {
        const ulong slot = base_slot + slot_k;
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;
        const float slot_w = top_w[slot];
        if (slot_w == 0.0f) continue;

        device const uchar * expert_gate = w_gate + (ulong)expert_i * gate_expert_stride_bytes;
        device const uchar * expert_up   = w_up   + (ulong)expert_i * gate_expert_stride_bytes;
        device const uchar * expert_down = w_down + (ulong)expert_i * down_expert_stride_bytes;

        for (uint f_base = 0; f_base < args.ffn; f_base += 16) {
            if (sgitg < 8) {
                const uint first_row = f_base + (uint)sgitg * NR0_Q4K;
                if (first_row < args.ffn) {
                    device const uchar * row0_g = expert_gate + (ulong)first_row * gate_row_stride_bytes;
                    device const uchar * row0_u = expert_up   + (ulong)first_row * gate_row_stride_bytes;
                    device const float * y4 = x + ix_q4 * QK_K + 64u * iq_q4 + 8u * ir_q4;

                    float sumf_g[NR0_Q4K] = {0.f, 0.f};
                    float sumf_u[NR0_Q4K] = {0.f, 0.f};

                    for (uint ib = ix_q4; ib < nb_h; ib += 4) {
                        float4 sumy = {0.f, 0.f, 0.f, 0.f};
                        for (short i = 0; i < 8; ++i) {
                            yl_q4[i+0] = y4[i+  0]; sumy[0] += yl_q4[i+0];
                            yl_q4[i+8] = y4[i+ 32]; sumy[1] += yl_q4[i+8];
                            yh_q4[i+0] = y4[i+128]; sumy[2] += yh_q4[i+0];
                            yh_q4[i+8] = y4[i+160]; sumy[3] += yh_q4[i+8];
                        }

                        for (short row = 0; row < NR0_Q4K; ++row) {
                            if (first_row + row >= args.ffn) break;

                            {
                                device const uchar * blk = row0_g + (ulong)row * gate_row_stride_bytes + (ulong)ib * Q4K_BYTES;
                                device const half     * dh = (device const half *) blk;
                                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq_q4;
                                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq_q4 + 4 * ir_q4;
                                device const uint16_t * q2 = q1 + 32;

                                sc16[0] =  sc[0]                & kmask1;
                                sc16[1] =  sc[2]                & kmask1;
                                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                                for (short i = 0; i < 4; ++i) {
                                    acc1[0] += yl_q4[2*i + 0] * (q1[i] & 0x000F);
                                    acc1[1] += yl_q4[2*i + 1] * (q1[i] & 0x0F00);
                                    acc1[2] += yl_q4[2*i + 8] * (q1[i] & 0x00F0);
                                    acc1[3] += yl_q4[2*i + 9] * (q1[i] & 0xF000);
                                    acc2[0] += yh_q4[2*i + 0] * (q2[i] & 0x000F);
                                    acc2[1] += yh_q4[2*i + 1] * (q2[i] & 0x0F00);
                                    acc2[2] += yh_q4[2*i + 8] * (q2[i] & 0x00F0);
                                    acc2[3] += yh_q4[2*i + 9] * (q2[i] & 0xF000);
                                }
                                sumf_g[row] += (float)dh[0] * (
                                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                                ) - (float)dh[1] * (
                                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                                );
                            }

                            {
                                device const uchar * blk = row0_u + (ulong)row * gate_row_stride_bytes + (ulong)ib * Q4K_BYTES;
                                device const half     * dh = (device const half *) blk;
                                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq_q4;
                                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq_q4 + 4 * ir_q4;
                                device const uint16_t * q2 = q1 + 32;

                                sc16[0] =  sc[0]                & kmask1;
                                sc16[1] =  sc[2]                & kmask1;
                                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                                for (short i = 0; i < 4; ++i) {
                                    acc1[0] += yl_q4[2*i + 0] * (q1[i] & 0x000F);
                                    acc1[1] += yl_q4[2*i + 1] * (q1[i] & 0x0F00);
                                    acc1[2] += yl_q4[2*i + 8] * (q1[i] & 0x00F0);
                                    acc1[3] += yl_q4[2*i + 9] * (q1[i] & 0xF000);
                                    acc2[0] += yh_q4[2*i + 0] * (q2[i] & 0x000F);
                                    acc2[1] += yh_q4[2*i + 1] * (q2[i] & 0x0F00);
                                    acc2[2] += yh_q4[2*i + 8] * (q2[i] & 0x00F0);
                                    acc2[3] += yh_q4[2*i + 9] * (q2[i] & 0xF000);
                                }
                                sumf_u[row] += (float)dh[0] * (
                                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                                ) - (float)dh[1] * (
                                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                                );
                            }
                        }
                        y4 += 4 * QK_K;
                    }

                    for (short row = 0; row < NR0_Q4K; ++row) {
                        float total_g = simd_sum(sumf_g[row]);
                        float total_u = simd_sum(sumf_u[row]);
                        if (tiisg == 0 && first_row + row < args.ffn) {
                            inner[first_row + row] = moe_silu_f(total_g) * total_u;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint row_base = 0; row_base < args.hidden; row_base += 16) {
            const uint first_row = row_base + sgitg;
            if (first_row < args.hidden) {
                float sumf = 0.0f;
                threadgroup const float * y1 = inner + ix_q5 * QK_K + y_offset_q5;

                for (uint i = ix_q5; i < nb_f; i += 4) {
                    threadgroup const float * y2 = y1 + 128;
                    float4 sumy = {0.f, 0.f, 0.f, 0.f};
                    for (short l = 0; l < 8; ++l) {
                        yl_q5[l+0] = y1[l+ 0]; sumy[0] += yl_q5[l+0];
                        yl_q5[l+8] = y1[l+32]; sumy[1] += yl_q5[l+8];
                        yh_q5[l+0] = y2[l+ 0]; sumy[2] += yh_q5[l+0];
                        yh_q5[l+8] = y2[l+32]; sumy[3] += yh_q5[l+8];
                    }

                    device const uchar * blk = expert_down + (ulong)first_row * down_row_stride_bytes + (ulong)i * Q5K_BYTES;
                    device const half     * dh = (device const half *) blk;
                    device const uint16_t * a  = (device const uint16_t *)(blk + 4) + iq_q5;
                    device const uchar    * qh = (blk + 4 + 12) + l0_q5;
                    device const uchar    * q1 = (blk + 4 + 12 + 32) + q_offset_q5;
                    device const uchar    * q2 = q1 + 64;

                    sc16[0] =  a[0]                & kmask1;
                    sc16[1] =  a[2]                & kmask1;
                    sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                    sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (short l = 0; l < 8; ++l) {
                        const uchar hq = qh[l];
                        acc1[0] += yl_q5[l+0] * (q1[l] & 0x0F);
                        acc1[1] += yl_q5[l+8] * (q1[l] & 0xF0);
                        acc1[2] += yh_q5[l+0] * (q2[l] & 0x0F);
                        acc1[3] += yh_q5[l+8] * (q2[l] & 0xF0);
                        acc2[0] += (hq & hm1) ? yl_q5[l+0] : 0.f;
                        acc2[1] += (hq & hm2) ? yl_q5[l+8] : 0.f;
                        acc2[2] += (hq & hm3) ? yh_q5[l+0] : 0.f;
                        acc2[3] += (hq & hm4) ? yh_q5[l+8] : 0.f;
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

                const float total = simd_sum(sumf);
                if (tiisg == 0) {
                    out[first_row] += slot_w * total;
                }
            }
        }
        // mem_device for the same reason as the initial-zero barrier:
        // the next expert iteration accumulates into `out[first_row]`,
        // so writes by other simdgroups in this iteration must be
        // visible across the boundary.
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

kernel void kernel_moe_down_q5_K_f32(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;

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

        device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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
        out[(ulong)slot * args.n_out + first_row] = tot;
    }
}

kernel void kernel_moe_mat_vec_f32_f32(
        constant moe_f32_args & args    [[buffer(0)]],
        device const float   * weight   [[buffer(1)]],
        device const float   * x        [[buffer(2)]],
        device const int     * top_idx  [[buffer(3)]],
        device       float   * out      [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_F32 + sgitg;
    if (row >= args.n_out) return;

    const ulong expert_stride = (ulong)args.n_out * args.n_in;
    device const float * w = weight + (ulong)expert_i * expert_stride + (ulong)row * args.n_in;

    float sum = 0.0f;
    const uint n_in_v4 = args.n_in / 4;
    device const float4 * w4 = (device const float4 *)w;
    device const float4 * x4 = (device const float4 *)x;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        const float4 a = w4[i];
        const float4 b = x4[i];
        sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    for (uint i = n_in_v4 * 4 + tiisg; i < args.n_in; i += 32) {
        sum += w[i] * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_mat_vec_bf16_f32(
        constant moe_f32_args & args    [[buffer(0)]],
        device const ushort  * weight   [[buffer(1)]],
        device const float   * x        [[buffer(2)]],
        device const int     * top_idx  [[buffer(3)]],
        device       float   * out      [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_F32 + sgitg;
    if (row >= args.n_out) return;

    const ulong expert_stride = (ulong)args.n_out * args.n_in;
    device const ushort * w = weight + (ulong)expert_i * expert_stride + (ulong)row * args.n_in;

    float sum = 0.0f;
    const uint n_in_v4 = args.n_in / 4;
    device const ushort4 * w4 = (device const ushort4 *)w;
    device const float4 * x4 = (device const float4 *)x;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        const ushort4 a = w4[i];
        const float4 b = x4[i];
        sum += moe_bf16_to_float(a.x) * b.x
             + moe_bf16_to_float(a.y) * b.y
             + moe_bf16_to_float(a.z) * b.z
             + moe_bf16_to_float(a.w) * b.w;
    }
    for (uint i = n_in_v4 * 4 + tiisg; i < args.n_in; i += 32) {
        sum += moe_bf16_to_float(w[i]) * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_down_bf16_f32(
        constant moe_f32_args & args    [[buffer(0)]],
        device const ushort  * weight   [[buffer(1)]],
        device const float   * inner    [[buffer(2)]],
        device const int     * top_idx  [[buffer(3)]],
        device       float   * out      [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_F32 + sgitg;
    if (row >= args.n_out) return;

    const ulong expert_stride = (ulong)args.n_out * args.n_in;
    device const ushort * w = weight + (ulong)expert_i * expert_stride + (ulong)row * args.n_in;
    device const float * x = inner + (ulong)slot * args.n_in;

    float sum = 0.0f;
    const uint n_in_v4 = args.n_in / 4;
    device const ushort4 * w4 = (device const ushort4 *)w;
    device const float4 * x4 = (device const float4 *)x;
    for (uint i = tiisg; i < n_in_v4; i += 32) {
        const ushort4 a = w4[i];
        const float4 b = x4[i];
        sum += moe_bf16_to_float(a.x) * b.x
             + moe_bf16_to_float(a.y) * b.y
             + moe_bf16_to_float(a.z) * b.z
             + moe_bf16_to_float(a.w) * b.w;
    }
    for (uint i = n_in_v4 * 4 + tiisg; i < args.n_in; i += 32) {
        sum += moe_bf16_to_float(w[i]) * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_mat_vec_iq3_xxs_f32(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * x       [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_IQ3_XXS + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const ulong row_stride_bytes = (ulong)nb * IQ3XXS_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * row_blocks = weight + (ulong)expert_i * expert_stride_bytes
                                            + (ulong)row * row_stride_bytes;

    float sum = 0.0f;
    for (uint i = tiisg; i < args.n_in; i += 32) {
        const uint bidx = i / QK_K;
        const uint qidx = i - bidx * QK_K;
        sum += moe_deq_iq3_xxs(row_blocks + (ulong)bidx * IQ3XXS_BYTES, qidx) * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_mat_vec_iq3_s_f32(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * x       [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_IQ3_XXS + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const ulong row_stride_bytes = (ulong)nb * IQ3S_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * row_blocks = weight + (ulong)expert_i * expert_stride_bytes
                                            + (ulong)row * row_stride_bytes;

    float sum = 0.0f;
    for (uint i = tiisg; i < args.n_in; i += 32) {
        const uint bidx = i / QK_K;
        const uint qidx = i - bidx * QK_K;
        sum += moe_deq_iq3_s(row_blocks + (ulong)bidx * IQ3S_BYTES, qidx) * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_swiglu_iq3_xxs_f32(
        constant moe_q4k_args & args      [[buffer(0)]],
        device const uchar    * w_gate    [[buffer(1)]],
        device const uchar    * w_up      [[buffer(2)]],
        device const float    * x         [[buffer(3)]],
        device const int      * top_idx   [[buffer(4)]],
        device       float    * inner     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_IQ3_XXS + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const ulong row_stride_bytes = (ulong)nb * IQ3XXS_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * gate_blocks = w_gate + (ulong)expert_i * expert_stride_bytes
                                             + (ulong)row * row_stride_bytes;
    device const uchar * up_blocks = w_up + (ulong)expert_i * expert_stride_bytes
                                         + (ulong)row * row_stride_bytes;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint i = tiisg; i < args.n_in; i += 32) {
        const uint bidx = i / QK_K;
        const uint qidx = i - bidx * QK_K;
        const float xv = x[i];
        gate_sum += moe_deq_iq3_xxs(gate_blocks + (ulong)bidx * IQ3XXS_BYTES, qidx) * xv;
        up_sum += moe_deq_iq3_xxs(up_blocks + (ulong)bidx * IQ3XXS_BYTES, qidx) * xv;
    }

    const float gate_tot = simd_sum(gate_sum);
    const float up_tot = simd_sum(up_sum);
    if (tiisg == 0) {
        inner[(ulong)slot * args.n_out + row] = moe_silu_f(gate_tot) * up_tot;
    }
}

kernel void kernel_moe_swiglu_iq3_s_f32(
        constant moe_q4k_args & args      [[buffer(0)]],
        device const uchar    * w_gate    [[buffer(1)]],
        device const uchar    * w_up      [[buffer(2)]],
        device const float    * x         [[buffer(3)]],
        device const int      * top_idx   [[buffer(4)]],
        device       float    * inner     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_IQ3_XXS + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const ulong row_stride_bytes = (ulong)nb * IQ3S_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * gate_blocks = w_gate + (ulong)expert_i * expert_stride_bytes
                                             + (ulong)row * row_stride_bytes;
    device const uchar * up_blocks = w_up + (ulong)expert_i * expert_stride_bytes
                                         + (ulong)row * row_stride_bytes;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint i = tiisg; i < args.n_in; i += 32) {
        const uint bidx = i / QK_K;
        const uint qidx = i - bidx * QK_K;
        const float xv = x[i];
        gate_sum += moe_deq_iq3_s(gate_blocks + (ulong)bidx * IQ3S_BYTES, qidx) * xv;
        up_sum += moe_deq_iq3_s(up_blocks + (ulong)bidx * IQ3S_BYTES, qidx) * xv;
    }

    const float gate_tot = simd_sum(gate_sum);
    const float up_tot = simd_sum(up_sum);
    if (tiisg == 0) {
        inner[(ulong)slot * args.n_out + row] = moe_silu_f(gate_tot) * up_tot;
    }
}

kernel void kernel_moe_swiglu_iq3_xxs_f32_fast(
        constant moe_q4k_args & args      [[buffer(0)]],
        device const uchar    * w_gate    [[buffer(1)]],
        device const uchar    * w_up      [[buffer(2)]],
        device const float    * x         [[buffer(3)]],
        device const int      * top_idx   [[buffer(4)]],
        device       float    * inner     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint first_row = (tgpig.x * NSG_IQ3_FAST + uint(sgitg)) * NR0_IQ3_FAST;
    if (first_row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const uint nb32 = nb * 8u;
    const ulong row_stride = (ulong)nb * IQ3XXS_BYTES;
    const ulong expert_stride = (ulong)args.n_out * row_stride;
    const ulong expert_base = (ulong)expert_i * expert_stride;

    const uint ix = uint(tiisg);
    device const float * x32 = x + 32u * ix;
    float gate_sum[NR0_IQ3_FAST] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sum[NR0_IQ3_FAST] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib32 = ix; ib32 < nb32; ib32 += 32u) {
        float xl[32];
        for (short i = 0; i < 32; ++i) {
            xl[i] = x32[i];
        }

        const uint ibl = ib32 / 8u;
        const uint ib = ib32 & 7u;

        for (short row = 0; row < NR0_IQ3_FAST; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;

            device const uchar * gate_blk = w_gate + expert_base + (ulong)out_row * row_stride
                                                   + (ulong)ibl * IQ3XXS_BYTES;
            device const uchar * up_blk = w_up + expert_base + (ulong)out_row * row_stride
                                             + (ulong)ibl * IQ3XXS_BYTES;

            const float gate_db = float(((device const half *)gate_blk)[0]);
            const float up_db = float(((device const half *)up_blk)[0]);
            device const uchar * gate_qs = gate_blk + 2;
            device const uchar * up_qs = up_blk + 2;
            device const uchar * gate_q3 = gate_qs + 8u * ib;
            device const uchar * up_q3 = up_qs + 8u * ib;
            device const ushort * gate_gas = (device const ushort *)(gate_qs + QK_K / 4) + 2u * ib;
            device const ushort * up_gas = (device const ushort *)(up_qs + QK_K / 4) + 2u * ib;
            const uint gate_aux = uint(gate_gas[0]) | (uint(gate_gas[1]) << 16);
            const uint up_aux = uint(up_gas[0]) | (uint(up_gas[1]) << 16);
            const float gate_d = gate_db * (0.5f + float(gate_aux >> 28));
            const float up_d = up_db * (0.5f + float(up_aux >> 28));

            float2 gate_part = {0.0f, 0.0f};
            float2 up_part = {0.0f, 0.0f};
            for (short l = 0; l < 4; ++l) {
                constant uchar * gate_grid1 = (constant uchar *)(moe_iq3xxs_grid + gate_q3[2 * l + 0]);
                constant uchar * gate_grid2 = (constant uchar *)(moe_iq3xxs_grid + gate_q3[2 * l + 1]);
                constant uchar * up_grid1 = (constant uchar *)(moe_iq3xxs_grid + up_q3[2 * l + 0]);
                constant uchar * up_grid2 = (constant uchar *)(moe_iq3xxs_grid + up_q3[2 * l + 1]);
                const uint gate_signs = uint(moe_ksigns_iq2xs[(gate_aux >> uint(7 * l)) & 127u]);
                const uint up_signs = uint(moe_ksigns_iq2xs[(up_aux >> uint(7 * l)) & 127u]);
                for (short j = 0; j < 4; ++j) {
                    const float gate_s1 = (gate_signs & uint(moe_kmask_iq2xs[j + 0])) ? -1.0f : 1.0f;
                    const float gate_s2 = (gate_signs & uint(moe_kmask_iq2xs[j + 4])) ? -1.0f : 1.0f;
                    const float up_s1 = (up_signs & uint(moe_kmask_iq2xs[j + 0])) ? -1.0f : 1.0f;
                    const float up_s2 = (up_signs & uint(moe_kmask_iq2xs[j + 4])) ? -1.0f : 1.0f;
                    const float x1 = xl[8 * l + j + 0];
                    const float x2 = xl[8 * l + j + 4];
                    gate_part[0] += x1 * float(gate_grid1[j]) * gate_s1;
                    gate_part[1] += x2 * float(gate_grid2[j]) * gate_s2;
                    up_part[0] += x1 * float(up_grid1[j]) * up_s1;
                    up_part[1] += x2 * float(up_grid2[j]) * up_s2;
                }
            }
            gate_sum[row] += gate_d * (gate_part[0] + gate_part[1]);
            up_sum[row] += up_d * (up_part[0] + up_part[1]);
        }

        x32 += 32u * 32u;
    }

    for (short row = 0; row < NR0_IQ3_FAST; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float gate_tot = simd_sum(gate_sum[row]) * 0.5f;
        const float up_tot = simd_sum(up_sum[row]) * 0.5f;
        if (tiisg == 0) {
            inner[(ulong)slot * args.n_out + out_row] = moe_silu_f(gate_tot) * up_tot;
        }
    }
}

kernel void kernel_moe_swiglu_iq3_s_f32_fast(
        constant moe_q4k_args & args      [[buffer(0)]],
        device const uchar    * w_gate    [[buffer(1)]],
        device const uchar    * w_up      [[buffer(2)]],
        device const float    * x         [[buffer(3)]],
        device const int      * top_idx   [[buffer(4)]],
        device       float    * inner     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint first_row = (tgpig.x * NSG_IQ3_FAST + uint(sgitg)) * NR0_IQ3_FAST;
    if (first_row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const uint nb32 = nb * 8u;
    const ulong row_stride = (ulong)nb * IQ3S_BYTES;
    const ulong expert_stride = (ulong)args.n_out * row_stride;
    const ulong expert_base = (ulong)expert_i * expert_stride;

    const uint ix = uint(tiisg);
    device const float * x32 = x + 32u * ix;
    float gate_sum[NR0_IQ3_FAST] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sum[NR0_IQ3_FAST] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib32 = ix; ib32 < nb32; ib32 += 32u) {
        float xl[32];
        for (short i = 0; i < 32; ++i) {
            xl[i] = x32[i];
        }

        const uint ibl = ib32 / 8u;
        const uint ib = ib32 & 7u;

        for (short row = 0; row < NR0_IQ3_FAST; ++row) {
            const uint out_row = first_row + uint(row);
            if (out_row >= args.n_out) continue;

            device const uchar * gate_blk = w_gate + expert_base + (ulong)out_row * row_stride
                                                   + (ulong)ibl * IQ3S_BYTES;
            device const uchar * up_blk = w_up + expert_base + (ulong)out_row * row_stride
                                             + (ulong)ibl * IQ3S_BYTES;

            const float gate_db = float(((device const half *)gate_blk)[0]);
            const float up_db = float(((device const half *)up_blk)[0]);
            device const uchar * gate_qbase = gate_blk + 2;
            device const uchar * up_qbase = up_blk + 2;
            device const uchar * gate_qs = gate_qbase + 8u * ib;
            device const uchar * up_qs = up_qbase + 8u * ib;
            device const uchar * gate_qh = gate_qbase + QK_K / 4 + ib;
            device const uchar * up_qh = up_qbase + QK_K / 4 + ib;
            device const uchar * gate_signs = gate_qbase + QK_K / 4 + QK_K / 32 + 4u * ib;
            device const uchar * up_signs = up_qbase + QK_K / 4 + QK_K / 32 + 4u * ib;
            device const uchar * gate_scales = gate_qbase + QK_K / 4 + QK_K / 32 + QK_K / 8 + (ib >> 1);
            device const uchar * up_scales = up_qbase + QK_K / 4 + QK_K / 32 + QK_K / 8 + (ib >> 1);
            const float gate_d = gate_db * (1.0f + 2.0f * float((uint(gate_scales[0]) >> (4u * (ib & 1u))) & 0x0fu));
            const float up_d = up_db * (1.0f + 2.0f * float((uint(up_scales[0]) >> (4u * (ib & 1u))) & 0x0fu));

            float2 gate_part = {0.0f, 0.0f};
            float2 up_part = {0.0f, 0.0f};
            for (short l = 0; l < 4; ++l) {
                const uint mask1 = uint(moe_kmask_iq2xs[2 * l + 0]);
                const uint mask2 = uint(moe_kmask_iq2xs[2 * l + 1]);
                const uint gate_idx1 = uint(gate_qs[2 * l + 0]) | (((uint(gate_qh[0]) & mask1) != 0u) ? 256u : 0u);
                const uint gate_idx2 = uint(gate_qs[2 * l + 1]) | (((uint(gate_qh[0]) & mask2) != 0u) ? 256u : 0u);
                const uint up_idx1 = uint(up_qs[2 * l + 0]) | (((uint(up_qh[0]) & mask1) != 0u) ? 256u : 0u);
                const uint up_idx2 = uint(up_qs[2 * l + 1]) | (((uint(up_qh[0]) & mask2) != 0u) ? 256u : 0u);
                constant uchar * gate_grid1 = (constant uchar *)(moe_iq3s_grid + gate_idx1);
                constant uchar * gate_grid2 = (constant uchar *)(moe_iq3s_grid + gate_idx2);
                constant uchar * up_grid1 = (constant uchar *)(moe_iq3s_grid + up_idx1);
                constant uchar * up_grid2 = (constant uchar *)(moe_iq3s_grid + up_idx2);
                for (short j = 0; j < 4; ++j) {
                    const float gate_s1 = (uint(gate_signs[l]) & uint(moe_kmask_iq2xs[j + 0])) ? -1.0f : 1.0f;
                    const float gate_s2 = (uint(gate_signs[l]) & uint(moe_kmask_iq2xs[j + 4])) ? -1.0f : 1.0f;
                    const float up_s1 = (uint(up_signs[l]) & uint(moe_kmask_iq2xs[j + 0])) ? -1.0f : 1.0f;
                    const float up_s2 = (uint(up_signs[l]) & uint(moe_kmask_iq2xs[j + 4])) ? -1.0f : 1.0f;
                    const float x1 = xl[8 * l + j + 0];
                    const float x2 = xl[8 * l + j + 4];
                    gate_part[0] += x1 * float(gate_grid1[j]) * gate_s1;
                    gate_part[1] += x2 * float(gate_grid2[j]) * gate_s2;
                    up_part[0] += x1 * float(up_grid1[j]) * up_s1;
                    up_part[1] += x2 * float(up_grid2[j]) * up_s2;
                }
            }
            gate_sum[row] += gate_d * (gate_part[0] + gate_part[1]);
            up_sum[row] += up_d * (up_part[0] + up_part[1]);
        }

        x32 += 32u * 32u;
    }

    for (short row = 0; row < NR0_IQ3_FAST; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float gate_tot = simd_sum(gate_sum[row]);
        const float up_tot = simd_sum(up_sum[row]);
        if (tiisg == 0) {
            inner[(ulong)slot * args.n_out + out_row] = moe_silu_f(gate_tot) * up_tot;
        }
    }
}

kernel void kernel_moe_swiglu_q6_K_f32(
        constant moe_q6k_args & args    [[buffer(0)]],
        device const uchar    * w_gate  [[buffer(1)]],
        device const uchar    * w_up    [[buffer(2)]],
        device const float    * x       [[buffer(3)]],
        device const int      * top_idx [[buffer(4)]],
        device       float    * inner   [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_gate = w_gate + (ulong)expert_i * expert_stride_bytes;
    device const uchar * expert_up = w_up + (ulong)expert_i * expert_stride_bytes;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf_g[NR0_Q6K] = {0.f, 0.f};
    float sumf_u[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const float * y_blk = x + (ulong)i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y_blk[l +  0];
            yl[4*l + 1] = y_blk[l + 32];
            yl[4*l + 2] = y_blk[l + 64];
            yl[4*l + 3] = y_blk[l + 96];
        }

        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row >= args.n_out) break;

            device const uchar * gate_blk = expert_gate + (ulong)(first_row + row) * row_stride_bytes
                                           + (ulong)i * Q6K_BYTES;
            device const uchar * up_blk = expert_up + (ulong)(first_row + row) * row_stride_bytes
                                         + (ulong)i * Q6K_BYTES;
            device const uchar  * gate_q1 = gate_blk + q_offset_l;
            device const uchar  * gate_q2 = gate_q1 + 32;
            device const uchar  * gate_qh = gate_blk + 128 + q_offset_h;
            device const int8_t * gate_sc = (device const int8_t *)(gate_blk + 128 + 64) + is;
            device const half   * gate_dh = (device const half *)(gate_blk + 128 + 64 + 16);
            device const uchar  * up_q1 = up_blk + q_offset_l;
            device const uchar  * up_q2 = up_q1 + 32;
            device const uchar  * up_qh = up_blk + 128 + q_offset_h;
            device const int8_t * up_sc = (device const int8_t *)(up_blk + 128 + 64) + is;
            device const half   * up_dh = (device const half *)(up_blk + 128 + 64 + 16);

            float4 gate_sums = {0.f, 0.f, 0.f, 0.f};
            float4 up_sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                gate_sums[0] += yl[4*l + 0] * ((int8_t)((gate_q1[l] & 0xF) | ((gate_qh[l] & kmask1) << 4)) - 32);
                gate_sums[1] += yl[4*l + 1] * ((int8_t)((gate_q2[l] & 0xF) | ((gate_qh[l] & kmask2) << 2)) - 32);
                gate_sums[2] += yl[4*l + 2] * ((int8_t)((gate_q1[l]  >> 4) | ((gate_qh[l] & kmask3) << 0)) - 32);
                gate_sums[3] += yl[4*l + 3] * ((int8_t)((gate_q2[l]  >> 4) | ((gate_qh[l] & kmask4) >> 2)) - 32);
                up_sums[0] += yl[4*l + 0] * ((int8_t)((up_q1[l] & 0xF) | ((up_qh[l] & kmask1) << 4)) - 32);
                up_sums[1] += yl[4*l + 1] * ((int8_t)((up_q2[l] & 0xF) | ((up_qh[l] & kmask2) << 2)) - 32);
                up_sums[2] += yl[4*l + 2] * ((int8_t)((up_q1[l]  >> 4) | ((up_qh[l] & kmask3) << 0)) - 32);
                up_sums[3] += yl[4*l + 3] * ((int8_t)((up_q2[l]  >> 4) | ((up_qh[l] & kmask4) >> 2)) - 32);
            }
            sumf_g[row] += (float)gate_dh[0] * (
                  gate_sums[0] * (float)gate_sc[0]
                + gate_sums[1] * (float)gate_sc[2]
                + gate_sums[2] * (float)gate_sc[4]
                + gate_sums[3] * (float)gate_sc[6]
            );
            sumf_u[row] += (float)up_dh[0] * (
                  up_sums[0] * (float)up_sc[0]
                + up_sums[1] * (float)up_sc[2]
                + up_sums[2] * (float)up_sc[4]
                + up_sums[3] * (float)up_sc[6]
            );
        }
    }

    for (short row = 0; row < NR0_Q6K; ++row) {
        const float gate_total = simd_sum(sumf_g[row]);
        const float up_total = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(gate_total) * up_total;
        }
    }
}

kernel void kernel_moe_swiglu_q8_0_f32(
        constant moe_q8_0_args & args    [[buffer(0)]],
        device const uchar     * w_gate  [[buffer(1)]],
        device const uchar     * w_up    [[buffer(2)]],
        device const float     * x       [[buffer(3)]],
        device const int       * top_idx [[buffer(4)]],
        device       float     * inner   [[buffer(5)]],
        threadgroup  float     * shmem   [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    constexpr ushort NR0 = NR0_Q80;
    constexpr ushort NSG = NSG_Q80;

    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint nb = args.n_in / QK8_0;
    const uint first_row = tgpig.x * NR0;
    if (first_row >= args.n_out) return;

    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;

    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * gate_row0 = w_gate + (ulong)expert_i * expert_stride_bytes
                                           + (ulong)first_row * row_stride_bytes;
    device const uchar * up_row0 = w_up + (ulong)expert_i * expert_stride_bytes
                                       + (ulong)first_row * row_stride_bytes;
    device const float * xb = x + (ulong)ib0 * QK8_0 + (ulong)il * NQ;

    float sumg[NR0] = {0.0f, 0.0f};
    float sumu[NR0] = {0.0f, 0.0f};
    float xv[NQ];

    for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
        for (ushort i = 0; i < NQ; ++i) {
            xv[i] = xb[i];
        }

        for (ushort row = 0; row < NR0; ++row) {
            if (first_row + row >= args.n_out) break;
            device const uchar * gate_blk = gate_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const uchar * up_blk = up_row0
                + (ulong)row * row_stride_bytes
                + (ulong)ib * Q8_0_BYTES;
            device const half   * gate_dh = (device const half *)gate_blk;
            device const half   * up_dh   = (device const half *)up_blk;
            device const int8_t * gate_qs = (device const int8_t *)(gate_blk + 2) + il * NQ;
            device const int8_t * up_qs   = (device const int8_t *)(up_blk + 2) + il * NQ;

            float sum_gate = 0.0f;
            float sum_up = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sum_gate += (float)gate_qs[i] * xv[i];
                sum_up += (float)up_qs[i] * xv[i];
            }
            sumg[row] += sum_gate * (float)gate_dh[0];
            sumu[row] += sum_up * (float)up_dh[0];
        }

        xb += (ulong)NSG * NQ * QK8_0;
    }

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (sgitg == 0) {
            gate_shmem[tiisg] = 0.0f;
            up_shmem[tiisg] = 0.0f;
        }
        sumg[row] = simd_sum(sumg[row]);
        sumu[row] = simd_sum(sumu[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        if (tiisg == 0) {
            gate_shmem[sgitg] = sumg[row];
            up_shmem[sgitg] = sumu[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
        threadgroup float * gate_shmem = shmem + NW * row;
        threadgroup float * up_shmem = shmem + NW * (NR0 + row);
        const float gate_total = simd_sum(gate_shmem[tiisg]);
        const float up_total = simd_sum(up_shmem[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(gate_total) * up_total;
        }
    }
}

kernel void kernel_moe_down_weighted_sum_q8_0_f32(
        constant moe_q8_0_args & args   [[buffer(0)]],
        device const uchar     * weight [[buffer(1)]],
        device const float     * inner  [[buffer(2)]],
        device const int       * top_idx [[buffer(3)]],
        device const float     * top_w  [[buffer(4)]],
        device       float     * out    [[buffer(5)]],
        threadgroup  float     * shmem  [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr ushort NW = 32;
    constexpr ushort NQ = NQ_Q80;
    constexpr ushort NR0 = NR0_Q80;
    constexpr ushort NSG = NSG_Q80;

    const uint nb = args.n_in / QK8_0;
    const uint first_row = tgpig * NR0;
    if (first_row >= args.n_out) return;

    const ushort ix = tiisg / (NW / NQ);
    const ushort il = tiisg % (NW / NQ);
    const uint ib0 = sgitg * NQ + ix;
    const ulong row_stride_bytes = (ulong)nb * Q8_0_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;

    float acc[NR0] = {0.0f, 0.0f};

    for (uint slot = 0; slot < args.topk; ++slot) {
        const int expert_i = top_idx[slot];
        const bool valid = expert_i >= 0 && expert_i < int(args.n_expert);
        device const uchar * row0 = valid
            ? weight + (ulong)expert_i * expert_stride_bytes + (ulong)first_row * row_stride_bytes
            : weight;
        device const float * xb = inner + (ulong)slot * args.n_in
                                + (ulong)ib0 * QK8_0 + (ulong)il * NQ;
        float sumf[NR0] = {0.0f, 0.0f};
        float xv[NQ];

        if (valid) {
            for (uint ib = ib0; ib < nb; ib += NSG * NQ) {
                for (ushort i = 0; i < NQ; ++i) {
                    xv[i] = xb[i];
                }

                for (ushort row = 0; row < NR0; ++row) {
                    if (first_row + row >= args.n_out) break;
                    device const uchar * blk = row0
                        + (ulong)row * row_stride_bytes
                        + (ulong)ib * Q8_0_BYTES;
                    device const half   * dh = (device const half *)blk;
                    device const int8_t * qs = (device const int8_t *)(blk + 2) + il * NQ;

                    float sumq = 0.0f;
                    for (ushort i = 0; i < NQ; ++i) {
                        sumq += (float)qs[i] * xv[i];
                    }
                    sumf[row] += sumq * (float)dh[0];
                }

                xb += (ulong)NSG * NQ * QK8_0;
            }
        }

        for (ushort row = 0; row < NR0; ++row) {
            threadgroup float * row_shmem = shmem + NW * row;
            if (sgitg == 0) {
                row_shmem[tiisg] = 0.0f;
            }
            sumf[row] = simd_sum(sumf[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort row = 0; row < NR0; ++row) {
            threadgroup float * row_shmem = shmem + NW * row;
            if (tiisg == 0) {
                row_shmem[sgitg] = sumf[row];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
            threadgroup float * row_shmem = shmem + NW * row;
            const float total = simd_sum(row_shmem[tiisg]);
            if (tiisg == 0 && sgitg == 0) {
                acc[row] += top_w[slot] * total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiisg == 0 && sgitg == 0) {
        for (ushort row = 0; row < NR0 && first_row + row < args.n_out; ++row) {
            out[first_row + row] = acc[row];
        }
    }
}

kernel void kernel_moe_down_iq4_xs_f32(
        constant moe_iq4xs_args & args           [[buffer(0)]],
        device const block_iq4_xs_local * weight [[buffer(1)]],
        device const float              * inner  [[buffer(2)]],
        device const int                * top_idx [[buffer(3)]],
        device       float              * out    [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    const uint row = tgpig.x * NSG_MOE_IQ4_XS + sgitg;
    if (row >= args.n_out) return;

    const uint nb = args.n_in / QK_K;
    const ulong expert_stride = (ulong)args.n_out * nb;
    device const block_iq4_xs_local * row_blocks =
        weight + (ulong)expert_i * expert_stride + (ulong)row * nb;
    device const float * x = inner + (ulong)slot * args.n_in;

    float sum = 0.0f;
    for (uint i = tiisg; i < args.n_in; i += 32) {
        const uint bidx = i / QK_K;
        const uint qidx = i - bidx * QK_K;
        sum += moe_deq_iq4_xs(row_blocks[bidx], qidx) * x[i];
    }

    const float tot = simd_sum(sum);
    if (tiisg == 0) {
        out[(ulong)slot * args.n_out + row] = tot;
    }
}

kernel void kernel_moe_down_q5_K_f32_grouped_rows(
        constant moe_q5k_args & args       [[buffer(0)]],
        device const uchar    * weight     [[buffer(1)]],
        device const float    * inner      [[buffer(2)]],
        device const int      * expert_idx [[buffer(3)]],
        device       float    * out        [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    const int expert_i = expert_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;

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

        device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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
        out[(ulong)slot * args.n_out + first_row] = tot;
    }
}

struct moe_group_q5k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

struct moe_group_q5k_tiny_args {
    uint M;
    uint N;
    uint K;
    uint n_expert;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

struct moe_group_q6k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

struct moe_group_bf16_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

inline void dequantize_q5_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qh     = blk_bytes + 4 + 12;
    device const uchar * qs     = blk_bytes + 4 + 12 + 32;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = 1u << (il / 2);
    short il_inner = il & 3;

    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;

    for (int i = 0; i < 16; ++i) {
        const float q_low  = (float)(qs[i] & mask);
        const float q_high = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (half)(dl * (q_low + q_high) - ml);
    }
}

inline void dequantize_iq4_xs_half_grouped(device const block_iq4_xs_local * blk,
                                           short il,
                                           thread half4x4 & reg) {
    const uint base = 16u * (uint)il;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = half(moe_deq_iq4_xs(*blk, base + (uint)i));
    }
}

kernel void kernel_moe_down_q5_K_f32_grouped_slots(
        constant moe_group_q5k_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q5K_BYTES;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_moe_down_q5_K_f32_grouped_slots_tiny8_r16(
        constant moe_group_q5k_tiny_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int MR = 16;
    constexpr int NR = 8;
    constexpr int SG_SMEM_BYTES = 1536;

    threadgroup uchar * sg_mem = shmem + (uint)sgitg * SG_SMEM_BYTES;
    threadgroup half * sa = (threadgroup half *)(sg_mem);
    threadgroup half * sb = (threadgroup half *)(sg_mem + 1024);
    threadgroup float * temp_str = ((threadgroup float *)(shmem + 6144)) + (uint)sgitg * (MR * NR);

    const int im = (int)tgpig.y * 4 + (int)sgitg;
    const int r0 = (int)tgpig.x * MR;

    int count = 0;
    if (im < (int)args.n_expert) {
        count = counts[im];
    }
    const bool active = im < (int)args.n_expert && count >= int(args.min_count)
                     && count <= int(args.max_count) && count > 0;
    const short nr0 = ((int)args.M - r0 < MR) ? (short)((int)args.M - r0) : MR;
    const short nr1 = (count < NR) ? (short)count : NR;

    const short row_local = (short)(tiisg / 2);
    const short il0 = (short)(tiisg & 1);
    short il = il0;

    const short slot_local = (short)(tiisg / 4);
    const short iy = 8 * (short)(tiisg & 3);

    const int global_m = r0 + row_local;
    const bool row_active = active && global_m < (int)args.M;
    const bool slot_active = active && slot_local < nr1;

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    const int im_safe = (im < (int)args.n_expert) ? im : 0;
    const int m_safe = (global_m < (int)args.M) ? global_m : 0;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im_safe
                                      + (ulong)args.nb01 * (ulong)m_safe
                                      + (ulong)offset1 * Q5K_BYTES;

    int slot_id = 0;
    if (slot_active) {
        slot_id = ids[(ulong)im * args.N + slot_local];
    }
    device const float * y_ptr = srcB + (ulong)args.stride_b * (ulong)max(slot_id, 0)
                                      + (ulong)iy;

    simdgroup_half8x8 ma[2];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc[2];
    mc[0] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    mc[1] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        half4x4 temp_a;
        if (row_active) {
            dequantize_q5_K_half_grouped(x_ptr, il, temp_a);
        } else {
            for (short i = 0; i < 16; ++i) {
                temp_a[i / 4][i % 4] = half(0.0f);
            }
        }
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = row_local / 8;
            const short lx = row_local & 7;
            const short ly = i & 7;
            const short ib = 2 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }

        {
            const short sx = (short)(tiisg & 3);
            threadgroup half * bd = sb + 64 * sx + 8 * slot_local;
            if (slot_active) {
                *(threadgroup half2x4 *)bd = (half2x4)(*((device const float2x4 *)y_ptr));
            } else {
                for (short i = 0; i < 8; ++i) {
                    bd[i] = half(0.0f);
                }
            }
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(ma[0], sa + 64 * (2 * ik + 0), 8, 0, false);
            simdgroup_load(ma[1], sa + 64 * (2 * ik + 1), 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, sb + 64 * ik, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(mc[0], mb, ma[0], mc[0]);
            simdgroup_multiply_accumulate(mc[1], mb, ma[1], mc[1]);
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(mc[0], temp_str, MR, 0, false);
    simdgroup_store(mc[1], temp_str + 8, MR, 0, false);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        for (int j = tiisg; j < nr1; j += 32) {
            const int slot = ids[(ulong)im * args.N + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + j * MR;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_moe_down_iq4_xs_f32_grouped_slots(
        constant moe_group_q5k_args & args           [[buffer(0)]],
        device const block_iq4_xs_local * srcA       [[buffer(1)]],
        device const float              * srcB       [[buffer(2)]],
        device const int                * counts     [[buffer(3)]],
        device const int                * ids        [[buffer(4)]],
        device       float              * dst        [[buffer(5)]],
        threadgroup  uchar              * shmem      [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void)tiisg;
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const block_iq4_xs_local * x_ptr = srcA + expert_stride * (ulong)im
                                                   + (ulong)args.nb01 * (r0 + lr0)
                                                   + (ulong)offset1;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_iq4_xs_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

inline void dequantize_q6_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    device const uint16_t * ql = (device const uint16_t *)(blk_bytes + 0);
    device const uint16_t * qh = (device const uint16_t *)(blk_bytes + 128);
    device const int8_t   * scales = (device const int8_t *)(blk_bytes + 128 + 64);
    const half d_all = ((device const half *)(blk_bytes + 128 + 64 + 16))[0];

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    const float sc = scales[(il % 2) + 2 * (il / 2)];
    const short il_inner = (il / 2) & 3;

    const uint32_t kmask1 = il_inner > 1
        ? (il_inner > 2 ? 0xC0C0C0C0 : 0x30303030)
        : (il_inner > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il_inner > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint8_t shr_h = il_inner > 2 ? 2 : 0;
    const uint8_t shl_h = il_inner > 1 ? 0 : (il_inner > 0 ? 2 : 4);
    const uint8_t shr_l = il_inner > 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        const uint32_t low  = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 * ((float)(q & 0xFF))         - ml);
        reg[i][1] = (half)(dl1 * ((float)(q & 0xFF00))       - ml);
        reg[i][2] = (half)(dl2 * ((float)(q & 0xFF0000))     - ml);
        reg[i][3] = (half)(dl3 * ((float)(q & 0xFF000000))   - ml);
    }
}

inline void dequantize_q8_0_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    const half d_h = ((device const half *)blk_bytes)[0];
    device const int8_t * qs = (device const int8_t *)(blk_bytes + 2);
    const float d = (float)d_h;
    const short base = 16 * il;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(d * (float)qs[base + i]);
    }
}

kernel void kernel_moe_down_q6_K_f32_grouped_slots(
        constant moe_group_q6k_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q6K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q6K_BYTES;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q6_K_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_moe_down_q8_0_f32_grouped_slots(
        constant moe_group_q6k_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q8_0_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q8_0_BYTES;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q8_0_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = il % 2;
        x_ptr += Q8_0_BYTES;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_moe_down_bf16_f32_grouped_slots(
        constant moe_group_bf16_args & args [[buffer(0)]],
        device const bfloat * srcA         [[buffer(1)]],
        device const float  * srcB         [[buffer(2)]],
        device const int    * counts       [[buffer(3)]],
        device const int    * ids          [[buffer(4)]],
        device       float  * dst          [[buffer(5)]],
        threadgroup  uchar  * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup bfloat * sa = (threadgroup bfloat *)(shmem);
    threadgroup bfloat * sb = (threadgroup bfloat *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count)
            || r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const ulong expert_stride = (ulong)args.K * args.M;
    device const bfloat * x_ptr = srcA + expert_stride * (ulong)im
                                       + (ulong)args.K * (r0 + lr0)
                                       + 16u * (ulong)il0;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_bfloat8x8   ma[4];
    simdgroup_bfloat8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = x_ptr[i];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup bfloat2x4 *)(sb + 64 * ib + 8 * ly) =
                (bfloat2x4)(*((device const float2x4 *)y_ptr));
        }

        x_ptr += NK_MM;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const bfloat * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const bfloat * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

kernel void kernel_moe_down_weighted_sum_q5_K_f32_packed_slots(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device const float    * top_w   [[buffer(4)]],
        device       float    * out     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    const ulong base_slot = (ulong)token * args.topk;

    float acc = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint slot_k = 0; slot_k < args.topk; ++slot_k) {
        const ulong slot = base_slot + slot_k;
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;

        device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
        device const float * x = inner + slot * args.n_in;
        device const float * y1 = x + ix * QK_K + y_offset;
        float sumf = 0.0f;

        for (uint i = ix; i < nb; i += 4) {
            device const float * y2 = y1 + 128;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 8; ++l) {
                yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
            }

            device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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

        const float total = simd_sum(sumf);
        if (tiisg == 0) {
            acc += top_w[slot] * total;
        }
    }

    if (tiisg == 0) {
        out[(ulong)token * args.n_out + first_row] = acc;
    }
}

kernel void kernel_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device const float    * top_w   [[buffer(4)]],
        device       float    * out     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort tid = tiisg / 4;
    const ushort ix  = tiisg % 4;
    const ushort k_ix = ix & 1u;
    const ushort row_in_sg = ix >> 1;
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
    if (nb != 2u) return;

    const uint first_row = tgpig.x * (NSG_Q5K * 2u) + sgitg * 2u;
    const uint row = first_row + row_in_sg;
    const bool row_active = row < args.n_out;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    const ulong base_slot = (ulong)token * args.topk;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint slot_k = 0; slot_k < args.topk; ++slot_k) {
        const ulong slot = base_slot + slot_k;
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;

        float sumf = 0.0f;
        if (row_active) {
            device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
            device const float * x = inner + slot * args.n_in;
            device const float * y1 = x + (uint)k_ix * QK_K + y_offset;
            device const float * y2 = y1 + 128;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 8; ++l) {
                yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
            }

            device const uchar * blk = expert_w + (ulong)row * row_stride_bytes + (ulong)k_ix * Q5K_BYTES;
            device const half     * dh = (device const half *) blk;
            device const uint16_t * a  = (device const uint16_t *)(blk + 4) + iq;
            device const uchar    * qh = (blk + 4 + 12) + l0;
            device const uchar    * q1 = (blk + 4 + 12 + 32) + q_offset;
            device const uchar    * q2 = q1 + 64;

            sc16[0] =  a[0]                & kmask1;
            sc16[1] =  a[2]                & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1v = {0.f, 0.f, 0.f, 0.f};
            float4 acc2v = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 8; ++l) {
                const uchar hbits = qh[l];
                acc1v[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1v[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1v[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1v[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2v[0] += (hbits & hm1) ? yl[l+0] : 0.f;
                acc2v[1] += (hbits & hm2) ? yl[l+8] : 0.f;
                acc2v[2] += (hbits & hm3) ? yh[l+0] : 0.f;
                acc2v[3] += (hbits & hm4) ? yh[l+8] : 0.f;
            }

            sumf = (float)dh[0] * (
                  sc8[0] * (acc1v[0]        + 16.f * acc2v[0])
                + sc8[1] * (acc1v[1] / 16.f + 16.f * acc2v[1])
                + sc8[4] * (acc1v[2]        + 16.f * acc2v[2])
                + sc8[5] * (acc1v[3] / 16.f + 16.f * acc2v[3])
            ) - (float)dh[1] * (
                  sumy[0] * sc8[2]
                + sumy[1] * sc8[3]
                + sumy[2] * sc8[6]
                + sumy[3] * sc8[7]
            );
        }

        const float total0 = simd_sum(row_in_sg == 0u ? sumf : 0.0f);
        const float total1 = simd_sum(row_in_sg == 1u ? sumf : 0.0f);
        if (tiisg == 0 && first_row < args.n_out) {
            acc0 += top_w[slot] * total0;
        }
        if (tiisg == 2 && first_row + 1u < args.n_out) {
            acc1 += top_w[slot] * total1;
        }
    }

    if (tiisg == 0 && first_row < args.n_out) {
        out[(ulong)token * args.n_out + first_row] = acc0;
    }
    if (tiisg == 2 && first_row + 1u < args.n_out) {
        out[(ulong)token * args.n_out + first_row + 1u] = acc1;
    }
}

kernel void kernel_moe_mat_vec_q5_K_f32(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * x       [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;

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

        device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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
        out[(ulong)slot * args.n_out + first_row] = tot;
    }
}

kernel void kernel_moe_down_q6_K_f32(
        constant moe_q6k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const float * y_blk = x + (ulong)i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y_blk[l +  0];
            yl[4*l + 1] = y_blk[l + 32];
            yl[4*l + 2] = y_blk[l + 64];
            yl[4*l + 3] = y_blk[l + 96];
        }

        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = expert_w + (ulong)(first_row + row) * row_stride_bytes
                                      + (ulong)i * Q6K_BYTES;
            device const uchar  * q1 = blk + q_offset_l;
            device const uchar  * q2 = q1 + 32;
            device const uchar  * qh = blk + 128 + q_offset_h;
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

    for (short row = 0; row < NR0_Q6K; ++row) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            out[(ulong)slot * args.n_out + first_row + row] = total;
        }
    }
}

kernel void kernel_moe_down_weighted_sum_q6_K_f32(
        constant moe_q6k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device const float    * top_w   [[buffer(4)]],
        device       float    * out     [[buffer(5)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float acc[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint slot = 0; slot < args.topk; ++slot) {
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;

        device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
        device const float * x = inner + (ulong)slot * args.n_in;
        float sumf[NR0_Q6K] = {0.f, 0.f};

        for (uint i = ix; i < nb; i += 2) {
            device const float * y_blk = x + (ulong)i * QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y_blk[l +  0];
                yl[4*l + 1] = y_blk[l + 32];
                yl[4*l + 2] = y_blk[l + 64];
                yl[4*l + 3] = y_blk[l + 96];
            }

            for (short row = 0; row < NR0_Q6K; ++row) {
                if (first_row + row >= args.n_out) break;

                device const uchar * blk = expert_w + (ulong)(first_row + row) * row_stride_bytes
                                          + (ulong)i * Q6K_BYTES;
                device const uchar  * q1 = blk + q_offset_l;
                device const uchar  * q2 = q1 + 32;
                device const uchar  * qh = blk + 128 + q_offset_h;
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

        for (short row = 0; row < NR0_Q6K; ++row) {
            const float total = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.n_out) {
                acc[row] += top_w[slot] * total;
            }
        }
    }

    if (tiisg == 0) {
        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row < args.n_out) out[first_row + row] = acc[row];
        }
    }
}

kernel void kernel_moe_weighted_sum_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out [[buffer(1)]],
        device const float   * weights    [[buffer(2)]],
        device       float   * out        [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n_out) return;
    float sum = 0.0f;
    for (uint e = 0; e < args.topk; ++e) {
        sum += weights[e] * expert_out[(ulong)e * args.n_out + tid];
    }
    out[tid] = sum;
}

kernel void kernel_moe_weighted_sum_packed_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out [[buffer(1)]],
        device const float   * weights    [[buffer(2)]],
        device       float   * out        [[buffer(3)]],
        uint2 tid2 [[thread_position_in_grid]]) {
    const uint tid = tid2.x;
    const uint token = tid2.y;
    if (tid >= args.n_out) return;
    float sum = 0.0f;
    const ulong base = (ulong)token * args.topk;
    for (uint e = 0; e < args.topk; ++e) {
        const ulong slot = base + e;
        sum += weights[slot] * expert_out[slot * args.n_out + tid];
    }
    out[(ulong)token * args.n_out + tid] = sum;
}

kernel void kernel_moe_grouped_finalizer_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out  [[buffer(1)]],
        device const float   * topk_w      [[buffer(2)]],
        device const float   * shared_gate [[buffer(3)]],
        device const float   * shared_out  [[buffer(4)]],
        device       float   * x_pack      [[buffer(5)]],
        uint2 tid2 [[thread_position_in_grid]]) {
    const uint col = tid2.x;
    const uint token = tid2.y;
    if (col >= args.n_out) return;
    float routed = 0.0f;
    const ulong base = (ulong)token * args.topk;
    for (uint e = 0; e < args.topk; ++e) {
        const ulong slot = base + e;
        routed += topk_w[slot] * expert_out[slot * args.n_out + col];
    }
    const ulong out_idx = (ulong)token * args.n_out + col;
    x_pack[out_idx] += routed + shared_gate[token] * shared_out[out_idx];
}

struct scatter_axpy_rows_args {
    uint n_cols;
    uint n_rows;
    uint out_rows;
};

kernel void kernel_scatter_rows_f32_unique(
        constant scatter_axpy_rows_args & args [[buffer(0)]],
        device const float             * x     [[buffer(1)]],
        device const int               * rows  [[buffer(2)]],
        device       float             * out   [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint src_row = tid / args.n_cols;
    const uint col = tid % args.n_cols;
    const int dst_row = rows[src_row];
    if (dst_row < 0 || dst_row >= int(args.out_rows)) return;
    out[(ulong)dst_row * args.n_cols + col] = x[tid];
}

kernel void kernel_axpy_scalar_f32(
        constant axpy_scalar_args & args [[buffer(0)]],
        device const float       * x     [[buffer(1)]],
        device const float       * scale [[buffer(2)]],
        device       float       * accum [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    accum[tid] += scale[0] * x[tid];
}

kernel void kernel_moe_shared_accum_resid_f32(
        constant axpy_scalar_args & args [[buffer(0)]],
        device const float       * shared_out [[buffer(1)]],
        device const float       * shared_gate [[buffer(2)]],
        device       float       * mixer_out  [[buffer(3)]],
        device       float       * x          [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    float mixed = mixer_out[tid];
    mixed += shared_gate[0] * shared_out[tid];
    mixer_out[tid] = mixed;
    x[tid] += mixed;
}

struct axpy_rowwise_args {
    uint n_cols;
    uint n_rows;
};

kernel void kernel_axpy_rowwise_f32(
        constant axpy_rowwise_args & args [[buffer(0)]],
        device const float        * x     [[buffer(1)]],
        device const float        * scale [[buffer(2)]],
        device       float        * accum [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint row = tid / args.n_cols;
    accum[tid] += scale[row] * x[tid];
}

kernel void kernel_scatter_axpy_rows_unique_f32(
        constant scatter_axpy_rows_args & args [[buffer(0)]],
        device const float             * x     [[buffer(1)]],
        device const int               * rows  [[buffer(2)]],
        device const float             * scale [[buffer(3)]],
        device       float             * accum [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint src_row = tid / args.n_cols;
    const uint col = tid % args.n_cols;
    const int dst_row = rows[src_row];
    if (dst_row < 0 || dst_row >= int(args.out_rows)) return;
    accum[(ulong)dst_row * args.n_cols + col] += scale[src_row] * x[tid];
}
