#define DS4_NO_GPU 1

#ifndef DSV4_DWARFSTAR_SOURCE
#error "compile with DSV4_DWARFSTAR_SOURCE pointing to ds4.c"
#endif

#include DSV4_DWARFSTAR_SOURCE

static void print_float_array(const float *values, size_t count) {
    putchar('[');
    for (size_t index = 0; index < count; index++) {
        if (index != 0) putchar(',');
        printf("%.9g", values[index]);
    }
    putchar(']');
}

typedef struct {
    uint8_t *map;
    ds4_model model;
    ds4_tensor wkv;
    ds4_tensor wgate;
    ds4_tensor ape;
    ds4_tensor norm;
} compressor_tensor_fixture;

typedef struct {
    const char *name;
    uint32_t seed;
    uint32_t input_dim;
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t layer;
    uint32_t positions;
} compressor_case;

static size_t oracle_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static ds4_tensor make_tensor(
        const char *name,
        uint32_t type,
        uint32_t ndim,
        uint64_t dim0,
        uint64_t dim1,
        uint64_t offset) {
    const uint64_t elements = dim0 * (ndim == 2 ? dim1 : 1u);
    const uint64_t bytes = elements * (type == DS4_TENSOR_F16 ? 2u : 4u);
    ds4_tensor tensor = {
        .name = {.ptr = name, .len = strlen(name)},
        .ndim = ndim,
        .dim = {dim0, dim1},
        .type = type,
        .rel_offset = offset,
        .abs_offset = offset,
        .elements = elements,
        .bytes = bytes,
    };
    return tensor;
}

static float compressor_recipe_value(
        uint64_t value,
        uint32_t modulus,
        int32_t center,
        float denominator) {
    return (float)((int32_t)(value % modulus) - center) / denominator;
}

static float compressor_input_value(uint32_t seed, uint32_t position, uint32_t column) {
    static const int8_t numerators[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
    const uint64_t index = (uint64_t)seed * 31u +
                           (uint64_t)position * 5u +
                           (uint64_t)column * 7u;
    return (float)numerators[index % 8u] / 4.0f;
}

static float compressor_signed_weight(
        uint64_t value,
        uint32_t base,
        uint32_t span,
        float denominator) {
    const float magnitude = (float)(base + (uint32_t)((value >> 1) % span)) / denominator;
    return (value & 1u) != 0 ? -magnitude : magnitude;
}

static float compressor_kv_weight(uint32_t seed, uint32_t row, uint32_t column) {
    const uint64_t value = (uint64_t)seed * 17u +
                           (uint64_t)row * 5u +
                           (uint64_t)column * 13u;
    return compressor_signed_weight(value, 500u, 1001u, 1000.0f);
}

static float compressor_gate_weight(uint32_t seed, uint32_t row, uint32_t column) {
    const uint64_t value = (uint64_t)seed * 23u +
                           (uint64_t)row * 11u +
                           (uint64_t)column * 19u;
    return compressor_signed_weight(value, 250u, 751u, 997.0f);
}

static float compressor_ape_value(uint32_t seed, uint32_t position, uint32_t row) {
    return compressor_recipe_value((uint64_t)seed * 29u +
                                   (uint64_t)position * 13u +
                                   (uint64_t)row * 3u,
                                   257u, 128, 1000.0f);
}

static float compressor_norm_value(uint32_t seed, uint32_t row) {
    return 0.75f + (float)(((uint64_t)seed * 5u + (uint64_t)row * 7u) % 101u) / 1000.0f;
}

static compressor_tensor_fixture make_compressor_tensors(
        uint32_t seed,
        uint32_t input_dim,
        uint32_t head_dim,
        uint32_t ratio) {
    const uint32_t width = (ratio == 4 ? 2u : 1u) * head_dim;
    const size_t matrix_elements = (size_t)input_dim * width;
    const size_t wkv_offset = 0;
    const size_t wgate_offset = oracle_align_up(wkv_offset + matrix_elements * 2u, 4u);
    const size_t ape_offset = oracle_align_up(wgate_offset + matrix_elements * 2u, 4u);
    const size_t norm_offset = oracle_align_up(ape_offset + (size_t)width * ratio * 2u, 4u);
    const size_t map_size = norm_offset + (size_t)head_dim * sizeof(float);
    uint8_t *map = xcalloc(1, map_size);

    compressor_tensor_fixture fixture = {
        .map = map,
        .model = {
            .fd = -1,
            .map = map,
            .size = map_size,
            .n_tensors = 4,
            .alignment = 4,
        },
        .wkv = make_tensor("synthetic.comp.wkv", DS4_TENSOR_F16, 2, input_dim, width, wkv_offset),
        .wgate = make_tensor("synthetic.comp.wgate", DS4_TENSOR_F16, 2, input_dim, width, wgate_offset),
        .ape = make_tensor("synthetic.comp.ape", DS4_TENSOR_F16, 2, width, ratio, ape_offset),
        .norm = make_tensor("synthetic.comp.norm", DS4_TENSOR_F32, 1, head_dim, 1, norm_offset),
    };

    uint16_t *wkv = (uint16_t *)(void *)(map + wkv_offset);
    uint16_t *wgate = (uint16_t *)(void *)(map + wgate_offset);
    uint16_t *ape = (uint16_t *)(void *)(map + ape_offset);
    float *norm = (float *)(void *)(map + norm_offset);
    for (uint32_t row = 0; row < width; row++) {
        for (uint32_t column = 0; column < input_dim; column++) {
            const uint64_t offset = (uint64_t)row * input_dim + column;
            wkv[offset] = f32_to_f16(compressor_kv_weight(seed, row, column));
            wgate[offset] = f32_to_f16(compressor_gate_weight(seed, row, column));
        }
    }
    for (uint32_t position = 0; position < ratio; position++) {
        for (uint32_t row = 0; row < width; row++) {
            ape[(uint64_t)position * width + row] =
                f32_to_f16(compressor_ape_value(seed, position, row));
        }
    }
    for (uint32_t row = 0; row < head_dim; row++) {
        norm[row] = compressor_norm_value(seed, row);
    }
    return fixture;
}

static uint64_t compressor_state_hash(const float *values, size_t count, bool scores) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < count; index++) {
        uint32_t bits;
        if (scores && values[index] <= DS4_NEG_INF * 0.5f) {
            bits = UINT32_C(0xff800000);
        } else {
            memcpy(&bits, values + index, sizeof(bits));
            if ((bits & UINT32_C(0x7fffffff)) == 0) bits = 0;
        }
        for (uint32_t byte = 0; byte < 4; byte++) {
            hash ^= (bits >> (byte * 8u)) & 0xffu;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static void print_compressor_case(const compressor_case *spec) {
    const uint32_t coefficient = spec->ratio == 4 ? 2u : 1u;
    const uint32_t width = coefficient * spec->head_dim;
    const uint32_t rows = coefficient * spec->ratio;
    const size_t state_count = (size_t)width * rows;
    compressor_tensor_fixture tensors =
        make_compressor_tensors(spec->seed, spec->input_dim, spec->head_dim, spec->ratio);
    float *state_kv = xcalloc(state_count, sizeof(float));
    float *state_score = xmalloc(state_count * sizeof(float));
    float *output = xmalloc((size_t)spec->head_dim * sizeof(float));
    for (size_t index = 0; index < state_count; index++) {
        state_score[index] = DS4_NEG_INF;
    }

    printf("{");
    printf("\"name\":\"%s\"", spec->name);
    printf(",\"seed\":%u", spec->seed);
    printf(",\"input_dim\":%u", spec->input_dim);
    printf(",\"head_dim\":%u", spec->head_dim);
    printf(",\"ratio\":%u", spec->ratio);
    printf(",\"layer\":%u", spec->layer);
    printf(",\"positions\":%u", spec->positions);
    printf(",\"projection_type\":\"f16\"");
    printf(",\"state_rows\":%u", rows);
    printf(",\"state_width\":%u", width);
    printf(",\"records\":[");
    for (uint32_t position = 0; position < spec->positions; position++) {
        float *input = xmalloc((size_t)spec->input_dim * sizeof(float));
        for (uint32_t column = 0; column < spec->input_dim; column++) {
            input[column] = compressor_input_value(spec->seed, position, column);
        }
        for (uint32_t index = 0; index < spec->head_dim; index++) {
            output[index] = 12345.0f;
        }
        const bool emitted = compressor_decode_one(output,
                                                   &tensors.model,
                                                   &tensors.wkv,
                                                   &tensors.wgate,
                                                   &tensors.ape,
                                                   &tensors.norm,
                                                   input,
                                                   state_kv,
                                                   state_score,
                                                   spec->head_dim,
                                                   spec->ratio,
                                                   spec->layer,
                                                   position);
        free(input);
        const bool expected_emission = ((position + 1u) % spec->ratio) == 0;
        if (emitted != expected_emission) ds4_die("synthetic compressor emission phase mismatch");
        if (!emitted) {
            for (uint32_t index = 0; index < spec->head_dim; index++) {
                if (output[index] != 12345.0f) {
                    ds4_die("synthetic compressor modified output off boundary");
                }
            }
        } else {
            for (uint32_t index = 0; index < spec->head_dim; index++) {
                if (!isfinite(output[index])) ds4_die("synthetic compressor emitted non-finite output");
            }
            if (spec->ratio == 4) {
                const size_t bank_count = (size_t)spec->ratio * width;
                if (memcmp(state_kv, state_kv + bank_count, bank_count * sizeof(float)) != 0 ||
                    memcmp(state_score, state_score + bank_count, bank_count * sizeof(float)) != 0) {
                    ds4_die("synthetic ratio-4 compressor banks did not mirror after emission");
                }
            }
        }

        if (position != 0) putchar(',');
        printf("{\"position\":%u", position);
        printf(",\"emitted\":%s", emitted ? "true" : "false");
        printf(",\"kv_hash\":\"%016" PRIx64 "\"",
               compressor_state_hash(state_kv, state_count, false));
        printf(",\"score_hash\":\"%016" PRIx64 "\"",
               compressor_state_hash(state_score, state_count, true));
        if (emitted) {
            printf(",\"start_position\":%u", position + 1u - spec->ratio);
            printf(",\"output\":");
            print_float_array(output, spec->head_dim);
        }
        putchar('}');
    }
    printf("]}");

    free(output);
    free(state_score);
    free(state_kv);
    free(tensors.map);
}

static void print_compressor_transitions(void) {
    const compressor_case cases[] = {
        {.name = "ratio4_attention", .seed = 1, .input_dim = 17,
         .head_dim = 512, .ratio = 4,
         .layer = 2, .positions = 9},
        {.name = "ratio4_indexer", .seed = 2, .input_dim = 17,
         .head_dim = 128, .ratio = 4,
         .layer = 2, .positions = 9},
        {.name = "ratio128_attention", .seed = 3, .input_dim = 17,
         .head_dim = 512, .ratio = 128,
         .layer = 3, .positions = 257},
    };
    g_ds4_shape = DS4_SHAPE_FLASH;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    g_ds4_compress_ratios[2] = 4;
    g_ds4_compress_ratios[3] = 128;

    printf("{");
    printf("\"hash_algorithm\":\"fnv1a64-f32-le; signed zero and score sentinel canonicalized\"");
    printf(",\"rms_epsilon\":%.9g", DS4_RMS_EPS);
    printf(",\"rotary_dim\":%u", DS4_N_ROT);
    printf(",\"rope_theta\":%.9g", DS4_COMPRESS_ROPE_FREQ_BASE);
    printf(",\"rope_scale_factor\":%.9g", DS4_ROPE_SCALE_FACTOR);
    printf(",\"original_context_length\":%" PRIu64, DS4_ROPE_ORIG_CTX);
    printf(",\"beta_fast\":%.9g", DS4_ROPE_YARN_BETA_FAST);
    printf(",\"beta_slow\":%.9g", DS4_ROPE_YARN_BETA_SLOW);
    printf(",\"recipes\":{");
    printf("\"input\":\"[-4,-3,-2,-1,1,2,3,4][(seed*31+position*5+column*7)%%8]/4\"");
    printf(",\"kv_weight\":\"value=seed*17+row*5+column*13; (value%%2?-1:1)*(500+((value>>1)%%1001))/1000\"");
    printf(",\"gate_weight\":\"value=seed*23+row*11+column*19; (value%%2?-1:1)*(250+((value>>1)%%751))/997\"");
    printf(",\"ape\":\"((seed*29+position*13+row*3)%%257-128)/1000\"");
    printf(",\"norm\":\"3/4+((seed*5+row*7)%%101)/1000\"");
    printf("}");
    printf(",\"cases\":[");
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        if (index != 0) putchar(',');
        print_compressor_case(&cases[index]);
    }
    printf("]}");
    ds4_threads_shutdown();
}

int main(void) {
    float mixes[24];
    float base[24];
    const float scale[3] = {0.7f, -0.4f, 1.2f};
    float split[24];
    for (int index = 0; index < 24; index++) {
        mixes[index] = ((float) index - 11.5f) * 0.13f;
        base[index] = ((float) ((index * 7) % 13) - 6.0f) * 0.04f;
    }
    hc_split_sinkhorn_one(split, mixes, scale, base, 4, 20, 1.0e-6f);

    float residual[12];
    const float block_output[3] = {0.4f, -0.7f, 1.1f};
    float post_output[12];
    for (int index = 0; index < 12; index++) {
        residual[index] = ((float) index - 5.0f) * 0.2f;
    }
    hc_post_one(post_output, block_output, residual, split + 4, split + 8, 3, 4);

    const float rope_input[8] = {9.0f, 8.0f, 7.0f, 6.0f, 1.0f, -2.0f, 3.0f, -4.0f};
    float rope_local[8];
    float rope_yarn[8];
    float rope_yarn_inverse[8];
    memcpy(rope_local, rope_input, sizeof(rope_input));
    memcpy(rope_yarn, rope_input, sizeof(rope_input));
    memcpy(rope_yarn_inverse, rope_input, sizeof(rope_input));
    rope_tail_ext_inplace(rope_local, 1, 8, 4, 17, 0, 10000.0f, 1.0f,
                          0.0f, 1.0f, 32.0f, 1.0f, false);
    const float yarn_scale = 1.0f / 16.0f;
    const float yarn_attn = 1.0f / (1.0f + 0.1f * logf(1.0f / yarn_scale));
    rope_tail_ext_inplace(rope_yarn, 1, 8, 4, 65536, 65536, 160000.0f,
                          yarn_scale, 1.0f, yarn_attn, 32.0f, 1.0f, false);
    rope_tail_ext_inplace(rope_yarn_inverse, 1, 8, 4, 65536, 65536, 160000.0f,
                          yarn_scale, 1.0f, yarn_attn, 32.0f, 1.0f, true);

    float ratio4_kv[8 * 8];
    float ratio4_scores[8 * 8];
    float ratio4_pool[4];
    for (int row = 0; row < 8; row++) {
        for (int column = 0; column < 8; column++) {
            const int offset = row * 8 + column;
            ratio4_kv[offset] = ((float) ((row * 11 + column * 5) % 19) - 9.0f) * 0.17f;
            ratio4_scores[offset] = ((float) ((row * 7 + column * 3) % 13) - 6.0f) * 0.09f;
        }
    }
    compressor_pool_decode_state(ratio4_pool, ratio4_kv, ratio4_scores, 4, 4);

    float ratio128_kv[128 * 2];
    float ratio128_scores[128 * 2];
    float ratio128_pool[2];
    for (int row = 0; row < 128; row++) {
        ratio128_kv[row * 2] = sinf((float) row * 0.07f);
        ratio128_kv[row * 2 + 1] = cosf((float) row * 0.11f);
        ratio128_scores[row * 2] = ((float) ((row * 3) % 17) - 8.0f) * 0.08f;
        ratio128_scores[row * 2 + 1] = ((float) ((row * 7) % 19) - 9.0f) * 0.06f;
    }
    compressor_pool_decode_state(ratio128_pool, ratio128_kv, ratio128_scores, 2, 128);

    float indexer_qat[128];
    for (int index = 0; index < 128; index++) {
        indexer_qat[index] = sinf((float) index * 0.17f) + cosf((float) index * 0.031f) * 0.25f;
    }
    dsv4_indexer_qat_row_inplace_cpu(indexer_qat, 128);

    const float logits[8] = {-4.0f, -0.5f, 0.0f, 1.25f, 3.0f, 0.0f, -2.5f, 2.0f};
    const float bias[8] = {0.0f, 0.25f, 0.0f, -0.1f, -3.0f, 0.0f, 2.5f, 0.2f};
    float router_scores[8];
    float selection_scores[8];
    int selected[3];
    float selected_weights[3];
    for (int index = 0; index < 8; index++) {
        router_scores[index] = sqrtf(softplus_stable(logits[index]));
        selection_scores[index] = router_scores[index] + bias[index];
    }
    topk_desc(selection_scores, 8, 3, selected);
    float selected_sum = 0.0f;
    for (int index = 0; index < 3; index++) {
        selected_weights[index] = router_scores[selected[index]];
        selected_sum += selected_weights[index];
    }
    if (selected_sum < 6.103515625e-5f) selected_sum = 6.103515625e-5f;
    for (int index = 0; index < 3; index++) {
        selected_weights[index] = selected_weights[index] / selected_sum * 1.5f;
    }

    const float swiglu_gate[4] = {-20.0f, -0.5f, 2.0f, 20.0f};
    const float swiglu_up[4] = {-20.0f, 0.25f, -3.0f, 20.0f};
    float swiglu_output[4];
    swiglu(swiglu_output, swiglu_gate, swiglu_up, 4, 10.0f);

    printf("{");
    printf("\"sinkhorn\":"); print_float_array(split, 24);
    printf(",\"hc_post\":"); print_float_array(post_output, 12);
    printf(",\"rope_local\":"); print_float_array(rope_local, 8);
    printf(",\"rope_yarn\":"); print_float_array(rope_yarn, 8);
    printf(",\"rope_yarn_inverse\":"); print_float_array(rope_yarn_inverse, 8);
    printf(",\"ratio4_pool\":"); print_float_array(ratio4_pool, 4);
    printf(",\"ratio128_pool\":"); print_float_array(ratio128_pool, 2);
    printf(",\"indexer_qat\":"); print_float_array(indexer_qat, 128);
    printf(",\"router_scores\":"); print_float_array(router_scores, 8);
    printf(",\"router_selected\":[%d,%d,%d]", selected[0], selected[1], selected[2]);
    printf(",\"router_weights\":"); print_float_array(selected_weights, 3);
    printf(",\"swiglu\":"); print_float_array(swiglu_output, 4);
    printf(",\"compressor_transitions\":"); print_compressor_transitions();
    printf("}\n");
    return 0;
}
