#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
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

struct layer_capture {
    bool active = false;
    bool invalid = false;
    std::array<bool, 36> seen{};
    std::array<std::array<float, 4096>, 36> rows{};
    uint32_t step = 0;
    struct attention_probe {
        uint32_t layer;
        std::array<float, 4096> query{}, output{};
        std::vector<float> keys, values;
        std::vector<bool> keys_seen, values_seen;
        bool query_seen = false, output_seen = false;
        attention_probe(uint32_t layer, uint32_t count) : layer(layer), keys(count * 1024), values(count * 1024),
            keys_seen(count, false), values_seen(count, false) {}
    };
    std::vector<attention_probe> attention;
};

static bool capture_layer(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & capture = *static_cast<layer_capture *>(user_data);
    for (auto & probe : capture.attention) {
        const std::string suffix = "-" + std::to_string(probe.layer);
        const std::string name(tensor->name);
        const bool query = capture.active && name == "Qcur" + suffix;
        const bool output = capture.active && name == "kqv" + suffix;
        const bool key = name == "Kcur" + suffix;
        const bool value = name == "Vcur" + suffix;
        if (!query && !output && !key && !value) continue;
        if (ask) return true;
        const size_t size = query || output ? 4096 : 1024;
        const std::array<int64_t, 4> shape = output ? std::array<int64_t, 4>{128, 1, 32, 1} :
            query ? std::array<int64_t, 4>{128, 32, 1, 1} : std::array<int64_t, 4>{128, 8, 1, 1};
        const bool seen = query ? probe.query_seen : output ? probe.output_seen :
            key ? probe.keys_seen[capture.step] : probe.values_seen[capture.step];
        if (seen || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
            ggml_nelements(tensor) != static_cast<int64_t>(size) ||
            !std::equal(shape.begin(), shape.end(), tensor->ne)) {
            capture.invalid = true;
            return false;
        }
        float * data = query ? probe.query.data() : output ? probe.output.data() :
            key ? probe.keys.data() + capture.step * 1024 : probe.values.data() + capture.step * 1024;
        ggml_backend_tensor_get(tensor, data, 0, size * sizeof(float));
        if (query) probe.query_seen = true;
        if (output) probe.output_seen = true;
        if (key) probe.keys_seen[capture.step] = true;
        if (value) probe.values_seen[capture.step] = true;
        return true;
    }
    if (!capture.active) return ask ? false : true;
    int layer = -1;
    for (int i = 0; i < 36; ++i) {
        if (std::string(tensor->name) == "l_out-" + std::to_string(i)) layer = i;
    }
    if (ask) return layer >= 0;
    if (layer < 0 || capture.seen[layer] || tensor->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(tensor) || tensor->ne[0] != 4096 || ggml_nelements(tensor) != 4096) {
        capture.invalid = true;
        return false;
    }
    ggml_backend_tensor_get(tensor, capture.rows[layer].data(), 0, 4096 * sizeof(float));
    capture.seen[layer] = true;
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--identity") {
            std::cout << K2_REFERENCE_REVISION << " default=F16-KV diagnostic=F32-KV serial flash=off wrapper="
                      << K2_WRAPPER_SHA256 << " cmake=" << K2_CMAKE_SHA256 << '\n';
            return 0;
        }
        const bool capture_last = argc > 1 && std::string(argv[1]) == "--capture-last";
        const bool f32_cache = argc > 1 && std::string(argv[1]) == "--f32-kv";
        if (capture_last || f32_cache) { --argc; ++argv; }
        if (argc < 5 || argc > 260 || std::string(argv[1]).rfind("--", 0) == 0) {
            throw std::runtime_error("usage: oracle [--capture-last | --f32-kv] MODEL OUTPUT BASE ID... (1..256 IDs)");
        }
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
        cp.n_ctx = std::max(32u, count);
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.type_k = f32_cache ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.type_v = cp.type_k;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
        cp.rope_freq_scale = 1.0f;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.no_perf = true;
        auto capture = std::make_unique<layer_capture>();
        if (capture_last) {
            capture->attention.emplace_back(0, count);
            capture->attention.emplace_back(20, count);
            cp.cb_eval = capture_layer;
            cp.cb_eval_user_data = capture.get();
        }
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
        write_u32(output, f32_cache ? 32 : 16);
        auto batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        for (uint32_t i = 0; i < count; ++i) {
            capture->active = capture_last && i + 1 == count;
            capture->step = i;
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
        if (capture_last) {
            if (capture->invalid || !std::all_of(capture->seen.begin(), capture->seen.end(), [](bool v) { return v; })) {
                throw std::runtime_error("missing or invalid layer captures");
            }
            std::ofstream layers(std::string(argv[2]) + ".layers", std::ios::binary | std::ios::trunc);
            layers.exceptions(std::ios::badbit | std::ios::failbit);
            layers.write("K2LAY001", 8);
            for (uint32_t value : {36u, 4096u, base + count - 1, static_cast<uint32_t>(tokens.back())}) write_u32(layers, value);
            for (const auto & row : capture->rows) {
                for (float value : row) {
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite layer capture");
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    write_u32(layers, bits);
                }
            }
            layers.close();
            std::ofstream attention(std::string(argv[2]) + ".attention", std::ios::binary | std::ios::trunc);
            attention.exceptions(std::ios::badbit | std::ios::failbit);
            attention.write("K2ATN001", 8);
            for (uint32_t value : {base, count, 2u, 128u, 32u, 8u}) write_u32(attention, value);
            for (auto token : tokens) write_u32(attention, token);
            for (const auto & probe : capture->attention) {
                if (!probe.query_seen || !probe.output_seen ||
                    !std::all_of(probe.keys_seen.begin(), probe.keys_seen.end(), [](bool v) { return v; }) ||
                    !std::all_of(probe.values_seen.begin(), probe.values_seen.end(), [](bool v) { return v; })) {
                    throw std::runtime_error("missing attention probe tensors");
                }
                write_u32(attention, probe.layer);
                const auto write_values = [&](const auto & values) {
                    for (float value : values) {
                        if (!std::isfinite(value)) throw std::runtime_error("nonfinite attention probe");
                        uint32_t bits;
                        std::memcpy(&bits, &value, sizeof(bits));
                        write_u32(attention, bits);
                    }
                };
                write_values(probe.query);
                write_values(probe.keys);
                write_values(probe.values);
                write_values(probe.output);
            }
            attention.close();
        }
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "K2 oracle: " << error.what() << '\n';
        return 1;
    }
}
