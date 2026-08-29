#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
template<class T>
std::vector<T> read_values(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const auto bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(T) != 0U)
        throw std::runtime_error("invalid input size");
    std::vector<T> output(static_cast<std::size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!output.empty()) stream.read(reinterpret_cast<char *>(output.data()), bytes);
    if (!stream) throw std::runtime_error("truncated input");
    return output;
}

skintokens::device_kind device(std::string_view value) {
    return value == "cpu" ? skintokens::device_kind::cpu : skintokens::device_kind::vulkan;
}
}

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-qwen-batch-parity MODEL_DIR FIXTURE_DIR cpu|vulkan\n";
        return 2;
    }
    skintokens::runtime_options options;
    options.device = device(argv[3]);
    auto backend = skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "tokenrig.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);
    const auto fixture = std::filesystem::path{argv[2]};
    const auto full_condition = fixture / "mesh-condition.f32";
    const auto condition = read_values<float>(std::filesystem::exists(full_condition) ?
        full_condition : fixture / "mesh-encoder-output.f32");
    const auto full_prefix = fixture / "token-generation-prefix.i32";
    const std::vector<std::int32_t> first = std::filesystem::exists(full_prefix) ?
        read_values<std::int32_t>(full_prefix) :
        std::vector<std::int32_t>{257, 263, 128, 128, 128, 128, 196, 128, 258};
    if (first.empty()) throw std::runtime_error("token input is empty");
    auto second = first;
    second.back() = second.back() == 267 ? 268 : 267;
    std::vector<std::int32_t> batched = first;
    batched.insert(batched.end(), second.begin(), second.end());
    auto actual = skintokens::detail::run_qwen_logits_batch(
        **weights, (*backend)->value, condition, batched, first.size(), 2U);
    std::vector<std::int32_t> shared_prefix{first.begin(), first.end() - 1};
    const std::int32_t branch_tokens[]{first.back(), second.back()};
    auto cached = skintokens::detail::run_qwen_kv_branches(
        **weights, (*backend)->value, condition, shared_prefix, branch_tokens);
    auto expected_first = skintokens::detail::run_qwen_logits(
        **weights, (*backend)->value, condition, first);
    auto expected_second = skintokens::detail::run_qwen_logits(
        **weights, (*backend)->value, condition, second);
    if (!actual) throw std::runtime_error(actual.error().message);
    if (!cached) throw std::runtime_error(cached.error().message);
    if (!expected_first) throw std::runtime_error(expected_first.error().message);
    if (!expected_second) throw std::runtime_error(expected_second.error().message);
    const std::size_t vocabulary = expected_first->size();
    if (expected_second->size() != vocabulary || actual->size() != vocabulary * 2U ||
        cached->size() != vocabulary * 2U)
        throw std::runtime_error("batched output shape mismatch");
    bool passed = true;
    const bool vulkan = std::string_view{argv[3]} == "vulkan";
    const auto compare = [&](std::string_view label, const std::vector<float> & values,
                             double relative_limit, float maximum_limit) {
        bool result = true;
        for (std::size_t beam = 0; beam < 2U; ++beam) {
            const auto & expected = beam == 0U ? *expected_first : *expected_second;
            double numerator = 0.0, denominator = 0.0;
            float maximum = 0.0F;
            for (std::size_t index = 0; index < vocabulary; ++index) {
                const float difference = values[beam * vocabulary + index] - expected[index];
                maximum = std::max(maximum, std::abs(difference));
                numerator += static_cast<double>(difference) * difference;
                denominator += static_cast<double>(expected[index]) * expected[index];
            }
            const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
            const auto actual_top = std::max_element(values.begin() + static_cast<std::ptrdiff_t>(beam * vocabulary + 267U),
                values.begin() + static_cast<std::ptrdiff_t>(beam * vocabulary + 33035U));
            const auto expected_top = std::max_element(expected.begin() + 267, expected.begin() + 33035);
            const auto actual_id = static_cast<std::size_t>(actual_top - values.begin()) - beam * vocabulary;
            const auto expected_id = static_cast<std::size_t>(expected_top - expected.begin());
            std::cout << label << " beam=" << beam << " max_abs=" << maximum << " rel_l2=" << relative
                      << " top=" << actual_id << " serial_top=" << expected_id << '\n';
            result = result && relative < relative_limit && maximum < maximum_limit && actual_id == expected_id;
        }
        return result;
    };
    passed = compare("batch", *actual, 1e-5, 1e-4F) && passed;
    passed = compare("kv_branch", *cached, vulkan ? 0.001 : 2e-5,
                     vulkan ? 0.01F : 1e-4F) && passed;
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << exception.what() << '\n';
    return 1;
}
