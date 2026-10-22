#include "capture.hpp"
#include <iostream>
#include <sstream>

using namespace probe;

template<class F> static void refuses(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    require(failed, "expected refusal");
}

int main() {
    try {
        check_host();
        const json base{{"schema", "qwen.glm-nextn-capture.v1"}, {"status", "complete"}, {"hidden_size", 2},
                        {"vocab_size", 3}, {"token_count", 11}, {"score_start", 7}, {"score_count", 4}};
        const auto c = Capture::parse(base);
        require(c.file_sizes().at("hidden.f32") == 88 && c.file_sizes().at("target-logits.f32") == 48,
                "capture byte accounting");
        for (const auto & mutation : std::vector<json>{
                 {{"score_count", 5}}, {{"token_count", -1}}, {{"hidden_size", 0}},
                 {{"score_start", true}}, {{"score_count", 4.0}}, {{"schema", "wrong"}},
                 {{"token_count", uint64_t(1) << 40}}, {{"status", "no_evaluable_rows"}},
                 {{"score_count", 0}}}) {
            auto j = base;
            j.update(mutation);
            refuses([&] { Capture::parse(j); });
        }
        std::vector<float> hidden(22);
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(i + 10);
        const std::string bytes(reinterpret_cast<const char *>(hidden.data()), hidden.size() * sizeof(float));
        for (size_t score_start : {size_t(0), size_t(1), size_t(7), size_t(10)}) {
            for (size_t chunk : {size_t(1), size_t(2), size_t(3), size_t(8), size_t(512)}) {
                std::istringstream stream(bytes);
                std::vector<float> supplied;
                for (size_t i = 0; i < 11;) {
                    const size_t rows = i < score_start ? std::min(chunk, score_start - i) : 1;
                    const auto block = shifted_hidden(stream, i, rows, 2);
                    supplied.insert(supplied.end(), block.begin(), block.end());
                    i += rows;
                }
                require(supplied.size() == 22 && supplied[0] == 0 && supplied[1] == 0, "zero initial hidden");
                for (size_t i = 1; i < 11; ++i) {
                    require(supplied[2 * i] == hidden[2 * (i - 1)]
                            && supplied[2 * i + 1] == hidden[2 * (i - 1) + 1], "chunk carry alignment");
                    if (i >= score_start) require(i - score_start < 11 - score_start, "target row alignment");
                }
            }
        }
        std::istringstream short_file(bytes.substr(0, 4));
        refuses([&] { shifted_hidden(short_file, 1, 1, 2); });
        std::istringstream stream(bytes);
        refuses([&] { shifted_hidden(stream, 0, 513, 2); });
        const float nan = std::numeric_limits<float>::quiet_NaN();
        refuses([&] { finite(&nan, 1, "fixture"); });
        std::cout << "capture manifest, extents, zero row and chunked alignment: PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
