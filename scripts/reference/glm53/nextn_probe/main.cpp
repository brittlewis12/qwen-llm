#include "capture.hpp"
#include "build_info.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

using namespace probe;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Options {
    fs::path model, input, output;
    std::string hidden_kind;
    size_t history_batch = 512;
};

static Options options(int argc, char ** argv) {
    const char * usage = "glm53_nextn_probe --model HEAD.gguf --input-dir CAPTURE --output-dir NEW_DIR "
                        "--hidden-kind post-norm|raw [--history-batch 1..512]";
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") { std::cout << usage << '\n'; std::exit(0); }
        require(seen.insert(key).second && i + 1 < argc, "duplicate option or missing value: " + key);
        const std::string value = argv[++i];
        if (key == "--model") o.model = fs::absolute(value);
        else if (key == "--input-dir") o.input = fs::absolute(value);
        else if (key == "--output-dir") o.output = fs::absolute(value);
        else if (key == "--hidden-kind") o.hidden_kind = value;
        else if (key == "--history-batch") {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "invalid history batch");
            o.history_batch = std::stoul(value);
        } else throw std::runtime_error("unknown option: " + key);
    }
    require(!o.model.empty() && !o.input.empty() && !o.output.empty(), usage);
    require(o.hidden_kind == "post-norm" || o.hidden_kind == "raw", "explicit --hidden-kind is required");
    require(o.history_batch >= 1 && o.history_batch <= 512, "history batch must be 1..512");
    return o;
}

static json head_inventory(const fs::path & path) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
        gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    require(bool(header), "cannot read head-only GGUF header");
    const auto arch = gguf_find_key(header.get(), "general.architecture");
    require(arch >= 0 && gguf_get_kv_type(header.get(), arch) == GGUF_TYPE_STRING
            && std::string(gguf_get_val_str(header.get(), arch)) == "glm5-next", "expected glm5-next GGUF");
    for (const auto & [name, expected] : std::map<std::string, uint32_t>{
             {"block_count", 46}, {"nextn_predict_layers", 1},
             {"embedding_length", 4096}, {"vocab_size", 154880}}) {
        const auto key = gguf_find_key(header.get(), ("glm5-next." + name).c_str());
        require(key >= 0 && gguf_get_kv_type(header.get(), key) == GGUF_TYPE_UINT32
                && gguf_get_val_u32(header.get(), key) == expected, "incompatible head metadata: " + name);
    }
    require(gguf_get_n_tensors(header.get()) == 32, "expected exactly 29 NextN + 3 fallback tensors");
    std::set<std::string> globals{"token_embd.weight", "output_norm.weight", "output.weight"};
    size_t head_count = 0;
    uint64_t payload = 0;
    json tensors = json::array();
    for (int64_t i = 0; i < gguf_get_n_tensors(header.get()); ++i) {
        const std::string name = gguf_get_tensor_name(header.get(), i);
        if (name.rfind("blk.45.", 0) == 0) ++head_count;
        else require(globals.erase(name) == 1, "refusing non-head tensor: " + name);
        const auto bytes = gguf_get_tensor_size(header.get(), i);
        payload += bytes;
        tensors.push_back({{"name", name}, {"bytes", bytes},
                           {"dtype", ggml_type_name(gguf_get_tensor_type(header.get(), i))}});
    }
    require(head_count == 29 && globals.empty(), "incomplete head-only inventory");
    for (int64_t i = 0; i < gguf_get_n_kv(header.get()); ++i) {
        require(std::string(gguf_get_key(header.get(), i)).rfind("split.", 0) != 0,
                "head-only GGUF must not declare split metadata");
    }
    json result{{"payload_bytes", payload}, {"tensors", tensors}};
    const auto p = gguf_find_key(header.get(), "qwen.nextn_probe.provenance");
    if (p >= 0 && gguf_get_kv_type(header.get(), p) == GGUF_TYPE_STRING) {
        result["extraction"] = json::parse(gguf_get_val_str(header.get(), p));
    }
    return result;
}

static void save_report(const fs::path & directory, const json & report) {
    const auto tmp = directory / "report.pending";
    std::ofstream file(tmp);
    file << report.dump(2) << '\n';
    file.close();
    require(bool(file), "cannot write report");
    fs::rename(tmp, directory / "report.json");
}

static json build_identity() {
    Dl_info info{};
    require(dladdr(reinterpret_cast<void *>(&llama_version), &info) != 0, "cannot locate loaded libllama");
    return {{"source_revision_at_build", PROBE_LLAMA_REVISION},
            {"source_tracked_dirty_at_build", PROBE_LLAMA_DIRTY},
            {"library_build_info_commit", PROBE_LIBRARY_COMMIT},
            {"linked_library", PROBE_LIBRARY_PATH}, {"loaded_library", info.dli_fname},
            {"library_sha256_at_build", PROBE_LIBRARY_SHA256},
            {"llama_header_sha256", PROBE_HEADER_SHA256},
            {"wrapper_sha256", PROBE_WRAPPER_SHA256},
            {"capture_header_sha256", PROBE_CAPTURE_SHA256},
            {"runtime_version", llama_version()}};
}

static void run(const Options & o) {
    check_host();
    std::ifstream manifest_file(o.input / "manifest.json");
    require(bool(manifest_file), "cannot open capture manifest");
    const json manifest = json::parse(manifest_file);
    const auto c = Capture::parse(manifest);
    require(c.hidden == 4096 && c.vocab == 154880, "expected GLM-5.3-Flash capture geometry");
    c.check_files(o.input);
    std::ifstream token_file(o.input / "tokens.u32", std::ios::binary);
    std::vector<uint32_t> tokens(c.count);
    read_exact(token_file, tokens.data(), tokens.size() * sizeof(uint32_t));
    for (auto token : tokens) require(token < c.vocab, "token outside vocabulary");
    const auto hidden_path = o.input / (o.hidden_kind == "raw" ? "hidden-raw.f32" : "hidden.f32");
    std::ifstream hidden_file(hidden_path, std::ios::binary);
    require(bool(hidden_file), "cannot open hidden capture");
    json report{{"schema", "qwen.glm-nextn-probe.v1"}, {"status", "running"},
                {"model", o.model.string()}, {"capture_directory", o.input.string()},
                {"capture_manifest", manifest}, {"hidden_kind", o.hidden_kind},
                {"history_batch", o.history_batch}, {"build", build_identity()},
                {"head_inventory", head_inventory(o.model)},
                {"history_rows_processed", 0}, {"scored_rows_returned", 0},
                {"history_requested_logit_rows", 0}, {"score_steps", json::array()},
                {"body_only_ms", nullptr},
                {"timing_scope", "history wall includes hidden IO and batch setup; score step wall is process + synchronize + full-logit retrieval, excluding input setup and output-file IO"}};
    require(::mkdir(o.output.c_str(), 0700) == 0, "output directory must not exist and its parent must exist");
    save_report(o.output, report);
    try {
        const auto load_start = Clock::now();
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.load_mtp = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
        require(bool(model), "head model load failed; inspect upstream diagnostics, do not synthesize missing tensors");
        require(llama_model_n_embd_out(model.get()) == int(c.hidden)
                && llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == int(c.vocab),
                "loaded model/capture geometry mismatch");
        auto cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = static_cast<uint32_t>(std::max<size_t>(c.count, 512));
        cp.n_seq_max = 1;
        cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(o.history_batch);
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
        cp.embeddings = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "cannot create MTP context");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        require(bool(batch), "cannot create extended batch");
        report["load_and_context_ms"] = ms(load_start);
        report["context_capacity"] = llama_n_ctx(ctx.get());
        report["n_gpu_layers_requested"] = mp.n_gpu_layers;
        report["phase"] = "history";
        save_report(o.output, report);

        auto prepare = [&](size_t start, size_t rows, bool logits) {
            auto states = shifted_hidden(hidden_file, start, rows, c.hidden);
            llama_batch_ext_clear(batch.get());
            for (size_t j = 0; j < rows; ++j) {
                const auto index = llama_batch_ext_add_token(batch.get(), 0, static_cast<llama_token>(tokens[start + j]));
                const auto pos = static_cast<llama_pos>(start + j);
                require(index == int(j)
                        && llama_batch_ext_set_pos(batch.get(), index, &pos)
                        && llama_batch_ext_set_embd_state(batch.get(), index, {states.data() + j * c.hidden, 1, c.hidden})
                        && llama_batch_ext_set_output_logits(batch.get(), index, logits), "extended batch rejected input");
            }
            // The extended API copies state rows into its own storage.
        };
        auto process = [&] {
            const int rc = llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            require(rc == 0, "llama_process failed: " + std::to_string(rc));
            llama_synchronize(ctx.get());
        };
        const auto history_start = Clock::now();
        for (size_t start = 0; start < c.score_start;) {
            const size_t rows = std::min(o.history_batch, c.score_start - start);
            prepare(start, rows, false);
            process();
            start += rows;
            report["history_rows_processed"] = start;
        }
        report["head_history_init_ms"] = ms(history_start);
        report["phase"] = "scoring";
        save_report(o.output, report);
        std::unique_ptr<FILE, decltype(&std::fclose)> logits_file(
            std::fopen((o.output / "draft-logits.f32").c_str(), "wbx"), std::fclose);
        require(bool(logits_file), "cannot create draft-logits.f32");
        double score_sum = 0;
        for (size_t i = c.score_start; i < c.count; ++i) {
            prepare(i, 1, true);
            const auto started = Clock::now();
            process();
            const float * logits = llama_get_logits_ith(ctx.get(), 0);
            require(logits != nullptr, "MTP did not return the requested scored row");
            const double step_ms = ms(started);
            finite(logits, c.vocab, "draft logits");
            require(std::fwrite(logits, sizeof(float), c.vocab, logits_file.get()) == c.vocab,
                    "draft logits write failed");
            score_sum += step_ms;
            report["score_steps"].push_back({{"position", i}, {"target_logit_row", i - c.score_start},
                {"hidden_source_position", i == 0 ? json(nullptr) : json(i - 1)},
                {"full_head_step_ms", step_ms}});
            report["scored_rows_returned"] = i - c.score_start + 1;
        }
        require(std::fflush(logits_file.get()) == 0 && ::fsync(fileno(logits_file.get())) == 0,
                "draft logits flush failed");
        require(fs::file_size(o.output / "draft-logits.f32") == uint64_t(c.score_count) * c.vocab * 4,
                "draft logits extent mismatch");
        report["score_steps_total_ms"] = score_sum;
        report["status"] = "complete";
        report["phase"] = "complete";
        save_report(o.output, report);
    } catch (const std::exception & e) {
        report["status"] = "failed";
        report["error"] = e.what();
        save_report(o.output, report);
        throw;
    }
}

int main(int argc, char ** argv) {
    try { run(options(argc, argv)); return 0; }
    catch (const std::exception & e) { std::cerr << "nextn probe: " << e.what() << '\n'; return 1; }
}
