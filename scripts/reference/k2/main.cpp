#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t number(const char * text) {
    const std::string input(text);
    if (input.empty() || input.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("expected unsigned decimal argument");
    }
    const auto value = std::stoull(input);
    if (value > UINT32_MAX) throw std::runtime_error("argument exceeds u32");
    return static_cast<uint32_t>(value);
}

static void write_u32(std::ostream & out, uint32_t value) {
    const char bytes[] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    out.write(bytes, 4);
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--identity") {
            std::cout << K2_REFERENCE_REVISION << " F16-KV serial flash=off wrapper="
                      << K2_WRAPPER_SHA256 << " cmake=" << K2_CMAKE_SHA256 << '\n';
            return 0;
        }
        if (argc < 5 || argc > 36) throw std::runtime_error("usage: oracle MODEL OUTPUT BASE ID... (1..32 IDs)");
        const uint32_t base = number(argv[3]);
        const uint32_t count = argc - 4;
        if (base > 524288 - count) throw std::runtime_error("absolute positions exceed K2 ceiling");
        std::vector<llama_token> tokens;
        for (int i = 4; i < argc; ++i) {
            const auto id = number(argv[i]);
            if (id >= 250624) throw std::runtime_error("ID outside K2 vocabulary");
            tokens.push_back(static_cast<llama_token>(id));
        }
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab != 250624) throw std::runtime_error("unexpected vocabulary");
        auto cp = llama_context_default_params();
        cp.n_ctx = 32;
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        cp.rope_freq_scale = 1.0f;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.no_perf = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!context) throw std::runtime_error("context creation failed");
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output");
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write("K2REF001", 8);
        write_u32(output, vocab);
        write_u32(output, count);
        write_u32(output, base);
        write_u32(output, 16);
        auto batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        for (uint32_t i = 0; i < count; ++i) {
            batch.token[0] = tokens[i];
            batch.pos[0] = base + i;
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("decode failed");
            const float * logits = llama_get_logits_ith(context.get(), -1);
            if (!logits) throw std::runtime_error("missing logits");
            write_u32(output, base + i);
            write_u32(output, tokens[i]);
            for (int32_t j = 0; j < vocab; ++j) {
                if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite reference logits");
                uint32_t bits;
                std::memcpy(&bits, &logits[j], sizeof(bits));
                write_u32(output, bits);
            }
        }
        llama_batch_free(batch);
        output.close();
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "K2 oracle: " << error.what() << '\n';
        return 1;
    }
}
