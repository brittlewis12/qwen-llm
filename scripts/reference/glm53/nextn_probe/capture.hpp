#pragma once

#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace probe {
using json = nlohmann::json;
namespace fs = std::filesystem;

inline void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

inline size_t integer(const json & j, const char * key, bool positive = false) {
    const auto & v = j.at(key);
    require(v.is_number_integer() && v.get<double>() >= (positive ? 1 : 0)
            && v.get<double>() <= INT32_MAX, std::string("invalid integer: ") + key);
    return v.get<size_t>();
}

struct Capture {
    size_t hidden, vocab, count, score_start, score_count;

    static Capture parse(const json & j) {
        require(j.at("schema") == "qwen.glm-nextn-capture.v1", "unsupported capture schema");
        require(j.at("status") == "complete", "capture has no complete evaluable rows");
        Capture c{integer(j, "hidden_size", true), integer(j, "vocab_size", true),
                  integer(j, "token_count", true), integer(j, "score_start"),
                  integer(j, "score_count", true)};
        require(c.count == c.score_start + c.score_count, "token_count must equal score_start + score_count");
        return c;
    }

    std::map<std::string, uint64_t> file_sizes() const {
        return {{"tokens.u32", uint64_t(count) * 4},
                {"hidden.f32", uint64_t(count) * hidden * 4},
                {"hidden-raw.f32", uint64_t(count) * hidden * 4},
                {"target-logits.f32", uint64_t(score_count) * vocab * 4}};
    }

    void check_files(const fs::path & directory) const {
        for (const auto & [name, bytes] : file_sizes()) {
            require(fs::is_regular_file(directory / name) && fs::file_size(directory / name) == bytes,
                    "missing or wrong-sized capture file: " + name);
        }
    }
};

inline void read_exact(std::istream & in, void * data, size_t bytes) {
    require(bytes <= size_t(std::numeric_limits<std::streamsize>::max()), "input too large");
    if (bytes) in.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    require(bool(in), "short input read");
}

inline void finite(const float * data, size_t count, const std::string & label) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::runtime_error(label + " contains a non-finite value at " + std::to_string(i));
        }
    }
}

// Absolute positions make the previous hidden row survive every prompt chunk boundary.
inline std::vector<float> shifted_hidden(std::istream & input, size_t start, size_t rows, size_t hidden) {
    require(rows > 0 && rows <= 512, "hidden batch must contain 1..512 rows");
    std::vector<float> result(rows * hidden, 0.0f);
    const size_t zero_rows = start == 0 ? 1 : 0;
    const size_t source_row = start == 0 ? 0 : start - 1;
    input.clear();
    input.seekg(static_cast<std::streamoff>(uint64_t(source_row) * hidden * sizeof(float)));
    read_exact(input, result.data() + zero_rows * hidden, (rows - zero_rows) * hidden * sizeof(float));
    finite(result.data(), result.size(), "hidden input");
    return result;
}

inline void check_host() {
    const uint32_t one = 1;
    require(*reinterpret_cast<const uint8_t *>(&one) == 1 && sizeof(float) == 4
            && std::numeric_limits<float>::is_iec559, "capture requires little-endian IEEE F32 host");
}
} // namespace probe
