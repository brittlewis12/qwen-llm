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
    printf("}\n");
    return 0;
}
