// Same-artifact GLM-5.3-Flash oracle: serial llama.cpp decode of exact token
// IDs, writing full-vocabulary logits per position and, optionally, named
// graph tensors streamed to disk. See README.md for formats and policy.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int32_t kVocab = 154880;
static constexpr uint32_t kMaxTokens = 4096;

static uint32_t number(const std::string & input) {
    if (input.empty() || input.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("expected unsigned decimal argument: " + input);
    }
    const auto value = std::stoull(input);
    if (value > UINT32_MAX) throw std::runtime_error("argument exceeds u32");
    return static_cast<uint32_t>(value);
}

static void write_u32(std::ostream & out, uint32_t value) {
    const char bytes[] = {char(value), char(value >> 8), char(value >> 16), char(value >> 24)};
    out.write(bytes, 4);
}

static void write_i64(std::ostream & out, int64_t value) {
    write_u32(out, static_cast<uint32_t>(static_cast<uint64_t>(value)));
    write_u32(out, static_cast<uint32_t>(static_cast<uint64_t>(value) >> 32));
}

// Default capture set: per-block residual and sub-block outputs, the mHC
// controls, KDA/MLA internals that decide the first checkpoint, and the head.
static const char * kDefaultCaptures =
    "hc_mixes,hc_pre,hc_post,hc_comb,hc_attn_pre,hc_attn_post,l_out,"
    "kda_q_conv,kda_k_conv,kda_v_conv,kda_g1,kda_beta,kda_scan_out,kda_g2,kda_normed,kda_out,"
    "q_absorbed,kv_cmpr,kqv_out,indexer_q,indexer_k,indexer_gate,indexer_weights,"
    "ffn_moe_out,ffn_shexp,ffn_out,result_norm";

struct capture_state {
    std::set<std::string> names;
    std::ofstream out;
    uint32_t step = 0;
    uint64_t records = 0;
    std::string error;
};

// Splits "name-12" into ("name", 12); names without a numeric suffix get -1.
static std::pair<std::string, int32_t> split_name(const char * raw) {
    const std::string name(raw);
    const auto dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size() &&
        name.find_first_not_of("0123456789", dash + 1) == std::string::npos) {
        return {name.substr(0, dash), static_cast<int32_t>(std::stol(name.substr(dash + 1)))};
    }
    return {name, -1};
}

static bool capture_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<capture_state *>(user_data);
    const auto [base, layer] = split_name(tensor->name);
    if (!state.names.count(base)) return ask ? false : true;
    if (ask) return true;
    if (!state.error.empty()) return false;
    if (!ggml_is_contiguous(tensor)) {
        state.error = std::string("non-contiguous capture ") + tensor->name;
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        state.error = std::string("unsupported capture type for ") + tensor->name;
        return false;
    }
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    write_u32(state.out, state.step);
    write_u32(state.out, static_cast<uint32_t>(layer));
    write_u32(state.out, static_cast<uint32_t>(base.size()));
    state.out.write(base.data(), static_cast<std::streamsize>(base.size()));
    write_u32(state.out, static_cast<uint32_t>(tensor->type));
    for (int i = 0; i < 4; ++i) write_i64(state.out, tensor->ne[i]);
    write_i64(state.out, static_cast<int64_t>(bytes.size()));
    state.out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ++state.records;
    return true;
}

int main(int argc, char ** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (args.size() == 1 && args[0] == "--identity") {
            std::cout << GLM53_REFERENCE_REVISION
                      << " serial n_batch=1 kv=F16 flash=off fused=defaults wrapper=" << GLM53_WRAPPER_SHA256
                      << " cmake=" << GLM53_CMAKE_SHA256 << '\n';
            return 0;
        }
        std::string captures;
        if (!args.empty() && args[0] == "--capture-default") {
            captures = kDefaultCaptures;
            args.erase(args.begin());
        } else if (args.size() > 1 && args[0] == "--capture") {
            captures = args[1];
            args.erase(args.begin(), args.begin() + 2);
        }
        if (args.size() < 3 || args[0].rfind("--", 0) == 0) {
            throw std::runtime_error(
                "usage: glm53_oracle [--capture NAME,... | --capture-default] MODEL OUTPUT ID... (1..4096 IDs)");
        }
        const std::string model_path = args[0];
        const std::string output_path = args[1];
        std::vector<llama_token> tokens;
        for (size_t i = 2; i < args.size(); ++i) {
            const auto id = number(args[i]);
            if (id >= static_cast<uint32_t>(kVocab)) throw std::runtime_error("ID outside the GLM vocabulary");
            tokens.push_back(static_cast<llama_token>(id));
        }
        if (tokens.size() > kMaxTokens) throw std::runtime_error("too many IDs");
        const auto count = static_cast<uint32_t>(tokens.size());

        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab != kVocab) throw std::runtime_error("unexpected vocabulary size");

        auto state = std::make_unique<capture_state>();
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(256u, (count + 255u) / 256u * 256u);
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = true;
        cp.op_offload = true;
        cp.no_perf = true;
        if (!captures.empty()) {
            std::stringstream list(captures);
            for (std::string name; std::getline(list, name, ',');) {
                if (!name.empty()) state->names.insert(name);
            }
            state->out.open(output_path + ".captures", std::ios::binary | std::ios::trunc);
            if (!state->out) throw std::runtime_error("cannot open capture output");
            state->out.exceptions(std::ios::badbit | std::ios::failbit);
            state->out.write("GLMCAP01", 8);
            cp.cb_eval = capture_tensor;
            cp.cb_eval_user_data = state.get();
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!context) throw std::runtime_error("context creation failed");

        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open output");
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write("GLMREF01", 8);
        write_u32(output, static_cast<uint32_t>(vocab));
        write_u32(output, count);
        auto batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        for (uint32_t i = 0; i < count; ++i) {
            state->step = i;
            batch.token[0] = tokens[i];
            batch.pos[0] = static_cast<llama_pos>(i);
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("decode failed");
            if (!state->error.empty()) throw std::runtime_error(state->error);
            const float * logits = llama_get_logits_ith(context.get(), -1);
            if (!logits) throw std::runtime_error("missing logits");
            write_u32(output, i);
            write_u32(output, static_cast<uint32_t>(tokens[i]));
            for (int32_t j = 0; j < vocab; ++j) {
                if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite reference logits");
                uint32_t bits;
                std::memcpy(&bits, &logits[j], sizeof(bits));
                write_u32(output, bits);
            }
        }
        llama_batch_free(batch);
        output.close();
        if (state->out.is_open()) {
            state->out.close();
            std::cerr << "GLM oracle: " << state->records << " capture records\n";
        }
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "GLM oracle: " << error.what() << '\n';
        return 1;
    }
}
