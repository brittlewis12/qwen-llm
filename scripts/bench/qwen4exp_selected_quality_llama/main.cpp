#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr std::string_view kPacketId = "2026-08-28-qwen4exp-selected-quality-v3";
constexpr std::string_view kManifestSha256 =
    "b3e649e99ecd069022e577d09efb6f6a968a508050257b3a34b9c204354079c5";
constexpr int32_t kVocabSize = 248320;
constexpr uint32_t kRequestedContextTokens = 4224;
constexpr uint32_t kEffectiveContextTokens = 4352;
constexpr uint32_t kRequiredDecodedTokens = 4195;
constexpr uint32_t kContextPaddingMultiple = 256;
constexpr uint32_t kBatchTokens = 512;

static_assert(kRequiredDecodedTokens <= kRequestedContextTokens);
static_assert(
    ((kRequestedContextTokens + kContextPaddingMultiple - 1) /
     kContextPaddingMultiple) *
        kContextPaddingMultiple ==
    kEffectiveContextTokens);

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

std::string hex_digest(const unsigned char * digest, size_t size) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(digest[i]);
    }
    return output.str();
}

std::string sha256_bytes(const void * data, size_t size) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> digest{};
    sha256_hash(
        digest.data(),
        static_cast<const unsigned char *>(data),
        size);
    return hex_digest(digest.data(), digest.size());
}

std::vector<uint8_t> read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open " + path.string());
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    require(length >= 0, "cannot size " + path.string());
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.read(reinterpret_cast<char *>(bytes.data()), length);
    require(input.good() || input.eof(), "cannot read " + path.string());
    require(static_cast<size_t>(input.gcount()) == bytes.size(), "short read " + path.string());
    return bytes;
}

std::string sha256_file(const std::filesystem::path & path) {
    const auto bytes = read_file(path);
    return sha256_bytes(bytes.data(), bytes.size());
}

std::filesystem::path canonical_file(const std::filesystem::path & path, std::string_view role) {
    require(std::filesystem::is_regular_file(path), std::string(role) + " is not a regular file");
    const auto canonical = std::filesystem::canonical(path);
    require(std::filesystem::is_regular_file(canonical), std::string(role) + " canonical target");
    return canonical;
}

void sha256_update_u32_le(sha256_t & digest, uint32_t value) {
    const std::array<unsigned char, 4> bytes = {
        static_cast<unsigned char>(value),
        static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16),
        static_cast<unsigned char>(value >> 24),
    };
    sha256_update(&digest, bytes.data(), bytes.size());
}

void sha256_update_u64_le(sha256_t & digest, uint64_t value) {
    std::array<unsigned char, 8> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<unsigned char>(value >> (i * 8));
    }
    sha256_update(&digest, bytes.data(), bytes.size());
}

std::string finish_sha256(sha256_t & digest) {
    std::array<unsigned char, SHA256_DIGEST_SIZE> bytes{};
    sha256_final(&digest, bytes.data());
    return hex_digest(bytes.data(), bytes.size());
}

std::string model_metadata(const llama_model * model, const char * key) {
    const int32_t length = llama_model_meta_val_str(model, key, nullptr, 0);
    require(length >= 0, std::string("missing model metadata ") + key);
    std::vector<char> value(static_cast<size_t>(length) + 1);
    const int32_t observed = llama_model_meta_val_str(model, key, value.data(), value.size());
    require(observed == length, std::string("unstable model metadata ") + key);
    return std::string(value.data(), static_cast<size_t>(length));
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:
            return "META";
    }
    fail("unknown backend device type");
}

void require_static_backend_registry() {
    require(ggml_backend_reg_count() == 2, "static backend registry count");
    std::set<std::string> names;
    for (size_t index = 0; index < ggml_backend_reg_count(); ++index) {
        names.emplace(ggml_backend_reg_name(ggml_backend_reg_get(index)));
    }
    require(names == std::set<std::string>{ "CPU", "MTL" }, "static backend registry names");
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        const auto device = ggml_backend_dev_get(index);
        const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        require(names.count(name) == 1, "device from undeclared backend");
    }
}

json initialize_static_backends() {
    require_static_backend_registry();
    llama_backend_init();
    require_static_backend_registry();

    json registries = json::array();
    for (size_t index = 0; index < ggml_backend_reg_count(); ++index) {
        const auto registry = ggml_backend_reg_get(index);
        registries.push_back({
            { "ordinal", index },
            { "name", ggml_backend_reg_name(registry) },
            { "device_count", ggml_backend_reg_dev_count(registry) },
        });
    }

    json devices = json::array();
    bool has_cpu = false;
    bool has_metal_accelerator = false;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        const auto device = ggml_backend_dev_get(index);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const std::string registry =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        has_cpu = has_cpu ||
            (registry == "CPU" && props.type == GGML_BACKEND_DEVICE_TYPE_CPU);
        has_metal_accelerator = has_metal_accelerator ||
            (registry == "MTL" &&
             (props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
              props.type == GGML_BACKEND_DEVICE_TYPE_IGPU));
        devices.push_back({
            { "ordinal", index },
            { "registry", registry },
            { "name", props.name == nullptr ? "" : props.name },
            { "description", props.description == nullptr ? "" : props.description },
            { "device_id", props.device_id == nullptr ? json(nullptr) : json(props.device_id) },
            { "type", device_type_name(props.type) },
            { "memory_free", props.memory_free },
            { "memory_total", props.memory_total },
            { "capabilities", {
                { "async", props.caps.async },
                { "host_buffer", props.caps.host_buffer },
                { "buffer_from_host_ptr", props.caps.buffer_from_host_ptr },
                { "events", props.caps.events },
                { "mmap_support", props.caps.mmap_support },
            } },
        });
    }
    require(has_cpu, "static CPU device unavailable");
    require(has_metal_accelerator, "static Metal accelerator unavailable");
    return {
        { "registration", "link-time CPU and Metal registry populated before llama_backend_init" },
        { "dynamic_backend_discovery_invoked", false },
        { "registries", std::move(registries) },
        { "devices", std::move(devices) },
    };
}

void attach_operation_binding(json & operation) {
    const std::string compact = operation.dump();
    operation["binding"] = {
        { "schema", "qwen4exp-selected-quality-llama-operation-binding-v1" },
        { "semantic_payload_encoding", "nlohmann::json compact UTF-8 before binding" },
        { "semantic_payload_bytes", compact.size() },
        { "semantic_payload_sha256", sha256_bytes(compact.data(), compact.size()) },
        { "semantic_payload_json_compact", compact },
    };
}

std::vector<llama_token> read_tokens(
    const std::filesystem::path & fixture_root,
    const std::string & fixture_id,
    const json & record) {
    const auto path = fixture_root / record.at("path").get<std::string>();
    const auto bytes = read_file(path);
    require(bytes.size() == record.at("bytes").get<size_t>(), fixture_id + " byte count");
    require(bytes.size() % 4 == 0, fixture_id + " token alignment");
    const auto raw_sha256 = sha256_bytes(bytes.data(), bytes.size());
    require(raw_sha256 == record.at("sha256").get<std::string>(), fixture_id + " raw SHA-256");
    require(
        raw_sha256 == record.at("sha256_raw_i32le").get<std::string>(),
        fixture_id + " raw token SHA-256");
    std::vector<llama_token> tokens;
    tokens.reserve(bytes.size() / 4);
    for (size_t offset = 0; offset < bytes.size(); offset += 4) {
        const uint32_t bits = static_cast<uint32_t>(bytes[offset]) |
            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
            (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        const int32_t token = static_cast<int32_t>(bits);
        require(token >= 0 && token < kVocabSize, fixture_id + " token range");
        tokens.push_back(token);
    }
    require(tokens.size() == record.at("token_count").get<size_t>(), fixture_id + " token count");
    return tokens;
}

class LogitTrace {
public:
    LogitTrace() {
        sha256_init(&digest_);
        const std::string domain =
            std::string("qwen4exp-selected-quality-llama-logit-trace-f32le-v1") + '\0';
        sha256_update(
            &digest_,
            reinterpret_cast<const unsigned char *>(domain.data()),
            domain.size());
    }

    void push(const std::vector<float> & logits) {
        require(logits.size() == static_cast<size_t>(kVocabSize), "trace vocabulary size");
        sha256_update_u64_le(digest_, logits.size());
        for (float value : logits) {
            require(std::isfinite(value), "nonfinite trace logit");
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            sha256_update_u32_le(digest_, bits);
        }
        ++rows_;
    }

    std::pair<std::string, size_t> finish() {
        return { finish_sha256(digest_), rows_ };
    }

private:
    sha256_t digest_{};
    size_t rows_ = 0;
};

std::string hash_logits(std::string_view phase, const std::vector<float> & logits) {
    sha256_t digest{};
    sha256_init(&digest);
    const std::string domain =
        "qwen4exp-selected-quality-llama-" + std::string(phase) + "-logits-f32le-v1" + '\0';
    sha256_update(
        &digest,
        reinterpret_cast<const unsigned char *>(domain.data()),
        domain.size());
    for (float value : logits) {
        require(std::isfinite(value), "nonfinite snapshot logit");
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        sha256_update_u32_le(digest, bits);
    }
    return finish_sha256(digest);
}

uint32_t argmax_lowest(const std::vector<float> & logits) {
    require(logits.size() == static_cast<size_t>(kVocabSize), "argmax vocabulary size");
    uint32_t best = 0;
    require(std::isfinite(logits[0]), "nonfinite argmax logit");
    for (uint32_t token = 1; token < logits.size(); ++token) {
        require(std::isfinite(logits[token]), "nonfinite argmax logit");
        if (logits[token] > logits[best]) {
            best = token;
        }
    }
    return best;
}

struct ScoredToken {
    json row;
    double nll;
    bool top1;
};

ScoredToken score_token(const std::vector<float> & logits, size_t ordinal, uint32_t target) {
    require(logits.size() == static_cast<size_t>(kVocabSize), "score vocabulary size");
    require(target < logits.size(), "score target range");
    double maximum = -std::numeric_limits<double>::infinity();
    for (float value : logits) {
        require(std::isfinite(value), "nonfinite score logit");
        maximum = std::max(maximum, static_cast<double>(value));
    }
    double sum_exp = 0.0;
    for (float value : logits) {
        sum_exp += std::exp(static_cast<double>(value) - maximum);
    }
    require(std::isfinite(sum_exp) && sum_exp > 0.0, "invalid logsumexp sum");
    const double logsumexp = maximum + std::log(sum_exp);
    const float target_logit = logits[target];
    const double nll = logsumexp - static_cast<double>(target_logit);
    require(std::isfinite(logsumexp) && std::isfinite(nll), "nonfinite score");
    const uint32_t prediction = argmax_lowest(logits);
    return {
        {
            { "ordinal", ordinal },
            { "target_token_id", target },
            { "target_logit_f32", target_logit },
            { "logsumexp_f64", logsumexp },
            { "nll_f64", nll },
            { "argmax_token_id", prediction },
            { "top1", prediction == target },
        },
        nll,
        prediction == target,
    };
}

struct ContextDeleter {
    void operator()(llama_context * context) const {
        llama_free(context);
    }
};

struct ModelDeleter {
    void operator()(llama_model * model) const {
        llama_model_free(model);
    }
};

using Context = std::unique_ptr<llama_context, ContextDeleter>;
using Model = std::unique_ptr<llama_model, ModelDeleter>;

void decode_tokens(llama_context * context, const llama_token * tokens, size_t count) {
    size_t offset = 0;
    while (offset < count) {
        const size_t chunk = std::min<size_t>(kBatchTokens, count - offset);
        auto batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens + offset),
            static_cast<int32_t>(chunk));
        const int32_t result = llama_decode(context, batch);
        require(result == 0, "llama_decode failed with code " + std::to_string(result));
        offset += chunk;
    }
}

void decode_token(llama_context * context, llama_token token) {
    decode_tokens(context, &token, 1);
}

std::vector<float> current_logits(llama_context * context) {
    float * logits = llama_get_logits_ith(context, -1);
    require(logits != nullptr, "llama returned no final logits");
    std::vector<float> copied(logits, logits + kVocabSize);
    require(
        std::all_of(copied.begin(), copied.end(), [](float value) { return std::isfinite(value); }),
        "llama returned nonfinite logits");
    return copied;
}

void reset_context(llama_context * context) {
    llama_memory_clear(llama_get_memory(context), true);
}

json run_natural(
    llama_context * context,
    const json & fixture,
    const std::vector<llama_token> & tokens,
    const json & operation) {
    const size_t prompt_count = fixture.at("prompt_token_count").get<size_t>();
    const size_t continuation_count = fixture.at("continuation_token_count").get<size_t>();
    require(continuation_count == 96, "natural continuation count");
    require(tokens.size() == prompt_count + continuation_count, "natural token count");
    reset_context(context);
    decode_tokens(context, tokens.data(), prompt_count);
    auto logits = current_logits(context);
    const std::string endpoint_sha256 = hash_logits("endpoint", logits);
    LogitTrace trace;
    json rows = json::array();
    double nll_sum = 0.0;
    size_t top1_hits = 0;
    for (size_t ordinal = 0; ordinal < continuation_count; ++ordinal) {
        trace.push(logits);
        const auto target = static_cast<uint32_t>(tokens[prompt_count + ordinal]);
        auto scored = score_token(logits, ordinal, target);
        nll_sum += scored.nll;
        top1_hits += static_cast<size_t>(scored.top1);
        rows.push_back(std::move(scored.row));
        decode_token(context, static_cast<llama_token>(target));
        logits = current_logits(context);
    }
    trace.push(logits);
    auto [trace_sha256, trace_rows] = trace.finish();
    require(trace_rows == continuation_count + 1, "natural trace row count");
    json result = {
        { "schema_version", 1 },
        { "operation_ordinal", operation.at("ordinal") },
        { "fixture_id", fixture.at("fixture_id") },
        { "mode", operation.at("mode") },
        { "arm", "D" },
        { "document_source_ordinal", fixture.at("document").at("source_ordinal") },
        { "prompt_token_count", prompt_count },
        { "selected_suffix_tokens", fixture.at("selected_suffix_tokens") },
        { "continuation", {
            { "tokens", continuation_count },
            { "nll_sum_f64", nll_sum },
            { "mean_nll_f64", nll_sum / continuation_count },
            { "top1_hits", top1_hits },
            { "scored_rows", std::move(rows) },
        } },
        { "logit_identity", {
            { "endpoint_logits_sha256_f32le", endpoint_sha256 },
            { "terminal_logits_sha256_f32le", hash_logits("terminal", logits) },
            { "trace_sha256_f32le", trace_sha256 },
            { "trace_rows", trace_rows },
        } },
    };
    return result;
}

json run_retrieval(
    llama_context * context,
    const json & fixture,
    const std::vector<llama_token> & tokens,
    const json & operation) {
    require(tokens.size() == 4099, "retrieval prompt count");
    const auto answers = fixture.at("answer_token_ids").get<std::vector<uint32_t>>();
    const auto stop_tokens = fixture.at("producer_stop_token_ids").get<std::vector<uint32_t>>();
    require(!answers.empty() && answers.size() <= 2, "retrieval answer count");
    reset_context(context);
    decode_tokens(context, tokens.data(), tokens.size());
    auto logits = current_logits(context);
    const std::string endpoint_sha256 = hash_logits("endpoint", logits);
    LogitTrace trace;
    json rows = json::array();
    std::vector<uint32_t> greedy_prefix;
    bool prefix_matches = true;
    double nll_sum = 0.0;
    for (size_t ordinal = 0; ordinal < answers.size(); ++ordinal) {
        trace.push(logits);
        const uint32_t prediction = argmax_lowest(logits);
        if (prefix_matches) {
            greedy_prefix.push_back(prediction);
            prefix_matches = prediction == answers[ordinal];
        }
        auto scored = score_token(logits, ordinal, answers[ordinal]);
        nll_sum += scored.nll;
        rows.push_back(std::move(scored.row));
        decode_token(context, static_cast<llama_token>(answers[ordinal]));
        logits = current_logits(context);
    }
    trace.push(logits);
    bool exact_pass = false;
    if (prefix_matches) {
        const uint32_t following = argmax_lowest(logits);
        greedy_prefix.push_back(following);
        exact_pass = std::find(stop_tokens.begin(), stop_tokens.end(), following) != stop_tokens.end();
    }
    auto [trace_sha256, trace_rows] = trace.finish();
    require(trace_rows == answers.size() + 1, "retrieval trace row count");
    json result = {
        { "schema_version", 1 },
        { "operation_ordinal", operation.at("ordinal") },
        { "fixture_id", fixture.at("fixture_id") },
        { "mode", operation.at("mode") },
        { "arm", "D" },
        { "kind", fixture.at("kind") },
        { "document_source_ordinal", fixture.at("document").at("source_ordinal") },
        { "prompt_token_count", tokens.size() },
        { "selected_suffix_tokens", tokens.size() - 2051 },
        { "answer", {
            { "expected_token_ids", answers },
            { "tokens", answers.size() },
            { "nll_sum_f64", nll_sum },
            { "mean_nll_f64", nll_sum / answers.size() },
            { "scored_rows", std::move(rows) },
            { "greedy_prefix_through_first_mismatch_or_stop", greedy_prefix },
            { "greedy_prefix_contract", "true free-running prefix; after a mismatch, remaining expected answer tokens are teacher-forced only" },
            { "exact_pass", exact_pass },
        } },
        { "logit_identity", {
            { "endpoint_logits_sha256_f32le", endpoint_sha256 },
            { "terminal_logits_sha256_f32le", hash_logits("terminal", logits) },
            { "trace_sha256_f32le", trace_sha256 },
            { "trace_rows", trace_rows },
        } },
    };
    return result;
}

struct ExpectedOperation {
    std::string ordering_key;
    std::string fixture_id;
    std::string mode;
};

std::vector<ExpectedOperation> expected_operations(const json & manifest) {
    std::vector<ExpectedOperation> expected;
    const auto append = [&expected](const json & fixture, std::string mode) {
        const auto fixture_id = fixture.at("fixture_id").get<std::string>();
        const auto token_sha = fixture.at("tokens").at("sha256_raw_i32le").get<std::string>();
        std::string domain = "qwen4exp-selected-quality-llama-order-v1";
        domain.push_back('\0');
        domain += "fixture_id=" + fixture_id + "\ntokens_sha256=" + token_sha + "\n";
        expected.push_back({ sha256_bytes(domain.data(), domain.size()), fixture_id, std::move(mode) });
    };
    for (const auto & fixture : manifest.at("natural_fixtures")) {
        append(fixture, "teacher_forced_nll_96");
    }
    for (const auto & fixture : manifest.at("retrieval_fixtures")) {
        append(fixture, "answer_nll_and_exact_prefix");
    }
    std::sort(
        expected.begin(),
        expected.end(),
        [](const auto & left, const auto & right) { return left.ordering_key < right.ordering_key; });
    return expected;
}

void write_exclusive(const std::filesystem::path & path, const std::string & bytes) {
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(descriptor >= 0, "cannot exclusively create output: " + std::string(std::strerror(errno)));
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t result = write(descriptor, bytes.data() + written, bytes.size() - written);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            const auto message = std::string(std::strerror(errno));
            close(descriptor);
            std::filesystem::remove(path);
            fail("cannot write output: " + message);
        }
        written += static_cast<size_t>(result);
    }
    if (fsync(descriptor) != 0) {
        const auto message = std::string(std::strerror(errno));
        close(descriptor);
        std::filesystem::remove(path);
        fail("cannot sync output: " + message);
    }
    require(close(descriptor) == 0, "cannot close output");
}

struct Arguments {
    std::vector<std::filesystem::path> model_shards;
    std::filesystem::path fixtures;
    std::filesystem::path output;
};

Arguments parse_arguments(int argc, char ** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        require(index + 1 < argc, "missing value for " + option);
        const std::filesystem::path value = argv[++index];
        if (option == "--model-shard") {
            arguments.model_shards.push_back(value);
        } else if (option == "--fixtures") {
            arguments.fixtures = value;
        } else if (option == "--output") {
            arguments.output = value;
        } else {
            fail("unknown option " + option);
        }
    }
    require(arguments.model_shards.size() == 3, "exactly three --model-shard values are required");
    require(!arguments.fixtures.empty(), "--fixtures is required");
    require(!arguments.output.empty(), "--output is required");
    require(!std::filesystem::exists(arguments.output), "core output already exists");
    return arguments;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        auto arguments = parse_arguments(argc, argv);
        for (size_t index = 0; index < arguments.model_shards.size(); ++index) {
            arguments.model_shards[index] = canonical_file(
                arguments.model_shards[index],
                "model shard " + std::to_string(index));
        }
        require(
            std::set<std::filesystem::path>(
                arguments.model_shards.begin(), arguments.model_shards.end()).size() ==
                arguments.model_shards.size(),
            "model shard paths must be distinct");
        arguments.fixtures = canonical_file(arguments.fixtures, "fixture manifest");
        const auto manifest_bytes = read_file(arguments.fixtures);
        require(
            sha256_bytes(manifest_bytes.data(), manifest_bytes.size()) == kManifestSha256,
            "fixture manifest SHA-256");
        const json manifest = json::parse(manifest_bytes);
        require(manifest.at("packet_id").get<std::string>() == kPacketId, "packet ID");
        require(manifest.at("schema_version").get<uint64_t>() == 3, "fixture schema version");
        const auto fixture_root = arguments.fixtures.parent_path();

        const auto expected = expected_operations(manifest);
        require(expected.size() == 20, "D operation count");
        const auto & plan = manifest.at("execution").at("operation_plan");
        require(plan.size() == 98, "operation plan count");
        std::map<std::string, const json *> natural;
        std::map<std::string, const json *> retrieval;
        for (const auto & fixture : manifest.at("natural_fixtures")) {
            natural.emplace(fixture.at("fixture_id").get<std::string>(), &fixture);
        }
        for (const auto & fixture : manifest.at("retrieval_fixtures")) {
            retrieval.emplace(fixture.at("fixture_id").get<std::string>(), &fixture);
        }
        uint32_t required_decoded_tokens = 0;
        for (const auto & fixture : manifest.at("natural_fixtures")) {
            required_decoded_tokens = std::max(
                required_decoded_tokens,
                fixture.at("prompt_token_count").get<uint32_t>() +
                    fixture.at("continuation_token_count").get<uint32_t>());
        }
        for (const auto & fixture : manifest.at("retrieval_fixtures")) {
            required_decoded_tokens = std::max(
                required_decoded_tokens,
                fixture.at("tokens").at("token_count").get<uint32_t>() +
                    fixture.at("answer_token_count").get<uint32_t>());
        }
        require(
            required_decoded_tokens == kRequiredDecodedTokens,
            "fixture required decoded-token capacity");
        for (size_t index = 0; index < expected.size(); ++index) {
            const auto & operation = plan.at(78 + index);
            require(operation.at("ordinal").get<size_t>() == 78 + index, "D ordinal");
            require(operation.at("phase") == "llama_cpp_triangulation", "D phase");
            require(operation.at("arm") == "D", "D arm");
            require(operation.at("fixture_id") == expected[index].fixture_id, "D fixture order");
            require(operation.at("mode") == expected[index].mode, "D mode");
            require(
                operation.at("ordering_key_sha256") == expected[index].ordering_key,
                "D ordering key");
        }

        const json static_backends = initialize_static_backends();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.check_tensors = true;
        model_params.load_mtp = false;
        std::vector<std::string> model_shard_strings;
        std::vector<const char *> model_shard_paths;
        for (const auto & path : arguments.model_shards) {
            model_shard_strings.push_back(path.string());
        }
        for (const auto & path : model_shard_strings) {
            model_shard_paths.push_back(path.c_str());
        }
        Model model(llama_model_load_from_splits(
            model_shard_paths.data(), model_shard_paths.size(), model_params));
        require(model != nullptr, "cannot load model");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        require(vocab != nullptr, "model has no vocabulary");
        require(llama_vocab_n_tokens(vocab) == kVocabSize, "model vocabulary size");
        const std::string architecture = model_metadata(model.get(), "general.architecture");
        const std::string tokenizer_model = model_metadata(model.get(), "tokenizer.ggml.model");
        const std::string tokenizer_pre = model_metadata(model.get(), "tokenizer.ggml.pre");
        require(architecture == "qwen4exp", "model architecture metadata");
        require(tokenizer_model == "gpt2", "tokenizer model metadata");
        require(tokenizer_pre == "qwen35", "tokenizer pretokenizer metadata");
        require(llama_vocab_is_eog(vocab, 248046), "producer stop token is not EOG");

        auto context_params = llama_context_default_params();
        context_params.n_ctx = kRequestedContextTokens;
        context_params.n_batch = kBatchTokens;
        context_params.n_ubatch = kBatchTokens;
        context_params.n_seq_max = 1;
        context_params.n_outputs_max = 1;
        context_params.n_outputs_max_per_seq = 1;
        context_params.type_k = GGML_TYPE_F16;
        context_params.type_v = GGML_TYPE_F16;
        context_params.kv_unified = false;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.offload_kqv = true;
        context_params.op_offload = true;
        context_params.no_perf = false;
        Context context(llama_init_from_model(model.get(), context_params));
        require(context != nullptr, "cannot create context");
        require(
            llama_n_ctx(context.get()) == kEffectiveContextTokens,
            "effective global context capacity");
        require(
            llama_n_ctx_seq(context.get()) == kEffectiveContextTokens,
            "effective per-sequence context capacity");
        require(
            llama_n_ctx(context.get()) >= kRequiredDecodedTokens,
            "required decoded-token capacity");
        require(llama_n_batch(context.get()) == kBatchTokens, "logical batch capacity");
        require(llama_n_ubatch(context.get()) == kBatchTokens, "physical batch capacity");
        require(llama_n_seq_max(context.get()) == 1, "sequence capacity");

        json operations = json::array();
        for (size_t index = 0; index < expected.size(); ++index) {
            const auto & operation = plan.at(78 + index);
            const auto fixture_id = expected[index].fixture_id;
            const json * fixture = nullptr;
            if (const auto found = natural.find(fixture_id); found != natural.end()) {
                fixture = found->second;
            } else if (const auto found = retrieval.find(fixture_id); found != retrieval.end()) {
                fixture = found->second;
            }
            require(fixture != nullptr, "unknown D fixture " + fixture_id);
            const auto tokens = read_tokens(fixture_root, fixture_id, fixture->at("tokens"));
            json observed;
            if (operation.at("mode") == "teacher_forced_nll_96") {
                observed = run_natural(context.get(), *fixture, tokens, operation);
            } else if (operation.at("mode") == "answer_nll_and_exact_prefix") {
                observed = run_retrieval(context.get(), *fixture, tokens, operation);
            } else {
                fail("unknown D mode");
            }
            observed["input_token_ids_sha256_raw_i32le"] =
                fixture->at("tokens").at("sha256_raw_i32le");
            observed["ordering_key_sha256"] = expected[index].ordering_key;
            attach_operation_binding(observed);
            operations.push_back(std::move(observed));
        }

        require_static_backend_registry();
        char description[1024] = {};
        const int32_t description_length = llama_model_desc(model.get(), description, sizeof(description));
        require(
            description_length >= 0 && static_cast<size_t>(description_length) < sizeof(description),
            "model description");
        json report = {
            { "schema", "qwen4exp-selected-quality-llama-core" },
            { "schema_version", 2 },
            { "packet_id", std::string(kPacketId) },
            { "fixture_manifest_sha256", std::string(kManifestSha256) },
            { "runner_build", {
                { "main_cpp_sha256", QWEN4EXP_RUNNER_MAIN_SHA256 },
                { "cmake_lists_sha256", QWEN4EXP_RUNNER_CMAKE_SHA256 },
                { "source_manifest_schema", "qwen4exp-selected-quality-llama-core-sources-v1" },
                { "source_manifest_sha256", QWEN4EXP_RUNNER_SOURCE_MANIFEST_SHA256 },
                { "llama_cpp_tree", QWEN4EXP_LLAMA_TREE },
                { "llama_cpp_export_manifest_sha256", QWEN4EXP_LLAMA_EXPORT_MANIFEST_SHA256 },
                { "build_policy_sha256", QWEN4EXP_BUILD_POLICY_SHA256 },
                { "compiler_id", QWEN4EXP_RUNNER_COMPILER_ID },
                { "compiler_version", QWEN4EXP_RUNNER_COMPILER_VERSION },
                { "build_type", QWEN4EXP_RUNNER_BUILD_TYPE },
                { "cxx_flags_sha256", QWEN4EXP_RUNNER_FLAGS_SHA256 },
            } },
            { "llama_cpp", {
                { "commit", QWEN4EXP_LLAMA_COMMIT },
                { "version", llama_version() },
            } },
            { "backends", static_backends },
            { "model", {
                { "split_paths", model_shard_strings },
                { "split_count", model_shard_strings.size() },
                { "description", std::string(description) },
                { "bytes", llama_model_size(model.get()) },
                { "parameters", llama_model_n_params(model.get()) },
                { "vocab_size", llama_vocab_n_tokens(vocab) },
                { "architecture", architecture },
                { "tokenizer_model", tokenizer_model },
                { "tokenizer_pre", tokenizer_pre },
                { "producer_stop_token_id", 248046 },
                { "producer_stop_token_is_eog", true },
                { "ftype", static_cast<int>(llama_model_ftype(model.get())) },
            } },
            { "context", {
                { "requested_n_ctx", kRequestedContextTokens },
                { "effective_n_ctx", llama_n_ctx(context.get()) },
                { "effective_n_ctx_seq", llama_n_ctx_seq(context.get()) },
                { "required_decoded_tokens", kRequiredDecodedTokens },
                { "context_padding_multiple", kContextPaddingMultiple },
                { "n_batch", llama_n_batch(context.get()) },
                { "n_ubatch", llama_n_ubatch(context.get()) },
                { "n_seq_max", llama_n_seq_max(context.get()) },
                { "kv_unified", false },
                { "kv_type_k", ggml_type_name(context_params.type_k) },
                { "kv_type_v", ggml_type_name(context_params.type_v) },
                { "flash_attention", "enabled" },
                { "gpu_layers", "all" },
                { "memory_cleared_with_data_before_each_operation", true },
            } },
            { "scoring", {
                { "vocab_size", kVocabSize },
                { "logsumexp", "max-subtracted F64 over every finite F32 logit" },
                { "argmax_tie_policy", "lowest token ID" },
                { "natural_forwards", "score each of 96 current rows, feed target once, retain one unscored terminal row" },
                { "retrieval_exact", "answer from generated token zero followed immediately by producer stop" },
            } },
            { "operations", std::move(operations) },
        };
        const std::string compact = report.dump();
        report["binding"] = {
            { "schema", "qwen4exp-selected-quality-llama-core-binding-v1" },
            { "semantic_payload_encoding", "nlohmann::json compact UTF-8 before binding" },
            { "semantic_payload_bytes", compact.size() },
            { "semantic_payload_sha256", sha256_bytes(compact.data(), compact.size()) },
            { "semantic_payload_json_compact", compact },
        };
        const std::string output = report.dump(2) + "\n";
        write_exclusive(arguments.output, output);

        context.reset();
        model.reset();
        llama_backend_free();
        std::cerr << "wrote llama.cpp core evidence to " << arguments.output
                  << " bytes=" << output.size()
                  << " sha256=" << sha256_bytes(output.data(), output.size()) << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
