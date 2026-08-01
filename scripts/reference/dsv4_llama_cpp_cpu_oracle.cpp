#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kConnectionCount = 4;
constexpr int64_t kHiddenSize = 7;
constexpr int64_t kTokenCount = 3;
constexpr int64_t kMixWidth = (2 + kConnectionCount) * kConnectionCount;
constexpr float kEpsilon = 1.0e-6f;
constexpr int32_t kSinkhornIterations = 20;

float modular_value(int64_t value, int64_t modulus, int64_t center, float scale) {
    return static_cast<float>(value % modulus - center) * scale;
}

void set_tensor(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("unexpected tensor type or length");
    }
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
}

std::vector<float> tensor_values(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || tensor->data == nullptr) {
        throw std::runtime_error("unexpected output tensor");
    }
    const auto * begin = static_cast<const float *>(tensor->data);
    std::vector<float> values(begin, begin + ggml_nelements(tensor));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite output");
        }
    }
    return values;
}

void print_float_array(const std::vector<float> & values) {
    std::cout << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("refusing to emit a non-finite value");
        }
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_shape(std::initializer_list<int64_t> dimensions) {
    std::cout << '[';
    size_t index = 0;
    for (int64_t dimension : dimensions) {
        if (index++ != 0) {
            std::cout << ',';
        }
        std::cout << dimension;
    }
    std::cout << ']';
}

} // namespace

int main() {
    try {
        std::locale::global(std::locale::classic());
        std::cout.imbue(std::locale::classic());
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

        std::vector<float> mixes(kMixWidth * kTokenCount);
        for (int64_t token = 0; token < kTokenCount; ++token) {
            for (int64_t parameter = 0; parameter < kMixWidth; ++parameter) {
                mixes[token * kMixWidth + parameter] =
                    modular_value(token * 19 + parameter * 7, 41, 20, 0.09f) +
                    static_cast<float>(token) * 0.013f;
            }
        }
        const std::vector<float> scale = {0.7f, -0.4f, 1.15f};
        std::vector<float> base(kMixWidth);
        for (int64_t parameter = 0; parameter < kMixWidth; ++parameter) {
            base[parameter] = modular_value(parameter * 11, 29, 14, 0.025f);
        }

        std::vector<float> residual(kHiddenSize * kConnectionCount * kTokenCount);
        for (int64_t token = 0; token < kTokenCount; ++token) {
            for (int64_t stream = 0; stream < kConnectionCount; ++stream) {
                for (int64_t dimension = 0; dimension < kHiddenSize; ++dimension) {
                    const int64_t offset =
                        token * kHiddenSize * kConnectionCount + stream * kHiddenSize + dimension;
                    residual[offset] =
                        modular_value(token * 23 + stream * 9 + dimension * 5, 37, 18, 0.07f);
                }
            }
        }
        std::vector<float> pre_weights(kConnectionCount * kTokenCount);
        std::vector<float> post_weights(kConnectionCount * kTokenCount);
        for (int64_t token = 0; token < kTokenCount; ++token) {
            for (int64_t stream = 0; stream < kConnectionCount; ++stream) {
                const int64_t offset = token * kConnectionCount + stream;
                pre_weights[offset] =
                    0.15f + static_cast<float>((token * 7 + stream * 5) % 11) * 0.07f;
                post_weights[offset] =
                    0.2f + static_cast<float>((token * 5 + stream * 3) % 9) * 0.19f;
            }
        }
        std::vector<float> block_output(kHiddenSize * kTokenCount);
        for (int64_t token = 0; token < kTokenCount; ++token) {
            for (int64_t dimension = 0; dimension < kHiddenSize; ++dimension) {
                block_output[token * kHiddenSize + dimension] =
                    modular_value(token * 13 + dimension * 8, 31, 15, 0.065f);
            }
        }

        const ggml_init_params params = {
            /* .mem_size = */ static_cast<size_t>(64) << 20,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc = */ false,
        };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
        if (!context) {
            throw std::runtime_error("ggml_init failed");
        }

        ggml_tensor * mixes_tensor =
            ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, kMixWidth, kTokenCount);
        ggml_tensor * scale_tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 3);
        ggml_tensor * base_tensor =
            ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, kMixWidth);
        ggml_tensor * residual_tensor = ggml_new_tensor_3d(
            context.get(), GGML_TYPE_F32, kHiddenSize, kConnectionCount, kTokenCount);
        ggml_tensor * pre_weights_tensor =
            ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, kConnectionCount, kTokenCount);
        ggml_tensor * post_weights_tensor =
            ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, kConnectionCount, kTokenCount);
        ggml_tensor * block_output_tensor =
            ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, kHiddenSize, kTokenCount);

        set_tensor(mixes_tensor, mixes);
        set_tensor(scale_tensor, scale);
        set_tensor(base_tensor, base);
        set_tensor(residual_tensor, residual);
        set_tensor(pre_weights_tensor, pre_weights);
        set_tensor(post_weights_tensor, post_weights);
        set_tensor(block_output_tensor, block_output);

        ggml_tensor * combination = ggml_dsv4_hc_comb(
            context.get(), mixes_tensor, scale_tensor, base_tensor, kEpsilon, kSinkhornIterations);
        ggml_tensor * pre_output =
            ggml_dsv4_hc_pre(context.get(), residual_tensor, pre_weights_tensor);
        ggml_tensor * post_output = ggml_dsv4_hc_post(
            context.get(),
            block_output_tensor,
            residual_tensor,
            post_weights_tensor,
            combination);

        ggml_cgraph * graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, pre_output);
        ggml_build_forward_expand(graph, post_output);
        ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
        plan.use_ref = true;
        if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GGML CPU graph execution failed");
        }

        const std::vector<float> combination_values = tensor_values(combination);
        const std::vector<float> pre_output_values = tensor_values(pre_output);
        const std::vector<float> post_output_values = tensor_values(post_output);

        std::cout << '{';
        std::cout << "\"status\":\"success\"";
        std::cout << ",\"backend\":\"ggml-cpu-reference\"";
        std::cout << ",\"thread_count\":1";
        std::cout << ",\"tensor_layout\":\"flat GGML ne[0]-fastest\"";
        std::cout << ",\"hc_comb\":{";
        std::cout << "\"connection_count\":" << kConnectionCount;
        std::cout << ",\"token_count\":" << kTokenCount;
        std::cout << ",\"mixes_shape\":"; print_shape({kMixWidth, kTokenCount});
        std::cout << ",\"mixes\":"; print_float_array(mixes);
        std::cout << ",\"scale_shape\":"; print_shape({3});
        std::cout << ",\"scale\":"; print_float_array(scale);
        std::cout << ",\"base_shape\":"; print_shape({kMixWidth});
        std::cout << ",\"base\":"; print_float_array(base);
        std::cout << ",\"epsilon\":" << kEpsilon;
        std::cout << ",\"iterations\":" << kSinkhornIterations;
        std::cout << ",\"output_shape\":";
        print_shape({kConnectionCount, kConnectionCount, kTokenCount});
        std::cout << ",\"output\":"; print_float_array(combination_values);
        std::cout << '}';
        std::cout << ",\"hc_pre\":{";
        std::cout << "\"hidden_size\":" << kHiddenSize;
        std::cout << ",\"connection_count\":" << kConnectionCount;
        std::cout << ",\"token_count\":" << kTokenCount;
        std::cout << ",\"residual_shape\":";
        print_shape({kHiddenSize, kConnectionCount, kTokenCount});
        std::cout << ",\"residual\":"; print_float_array(residual);
        std::cout << ",\"weights_shape\":"; print_shape({kConnectionCount, kTokenCount});
        std::cout << ",\"weights\":"; print_float_array(pre_weights);
        std::cout << ",\"output_shape\":"; print_shape({kHiddenSize, kTokenCount});
        std::cout << ",\"output\":"; print_float_array(pre_output_values);
        std::cout << '}';
        std::cout << ",\"hc_post\":{";
        std::cout << "\"hidden_size\":" << kHiddenSize;
        std::cout << ",\"connection_count\":" << kConnectionCount;
        std::cout << ",\"token_count\":" << kTokenCount;
        std::cout << ",\"block_output_shape\":"; print_shape({kHiddenSize, kTokenCount});
        std::cout << ",\"block_output\":"; print_float_array(block_output);
        std::cout << ",\"residual_shape\":";
        print_shape({kHiddenSize, kConnectionCount, kTokenCount});
        std::cout << ",\"residual\":"; print_float_array(residual);
        std::cout << ",\"weights_shape\":"; print_shape({kConnectionCount, kTokenCount});
        std::cout << ",\"weights\":"; print_float_array(post_weights);
        std::cout << ",\"combination_shape\":";
        print_shape({kConnectionCount, kConnectionCount, kTokenCount});
        std::cout << ",\"combination\":"; print_float_array(combination_values);
        std::cout << ",\"output_shape\":";
        print_shape({kHiddenSize, kConnectionCount, kTokenCount});
        std::cout << ",\"output\":"; print_float_array(post_output_values);
        std::cout << '}';
        std::cout << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "dsv4 llama.cpp CPU oracle: " << error.what() << '\n';
        return 1;
    }
}
