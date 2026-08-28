#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
template<class T> std::vector<T> read_values(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open fixture: " + path.string());
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::size_t>(size) % sizeof(T) != 0U)
        throw std::runtime_error("invalid fixture: " + path.string());
    std::vector<T> output(static_cast<std::size_t>(size) / sizeof(T));
    stream.seekg(0); if (!output.empty()) stream.read(reinterpret_cast<char *>(output.data()), size);
    if (!stream) throw std::runtime_error("truncated fixture: " + path.string());
    return output;
}
skintokens::device_kind device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    return skintokens::device_kind::automatic;
}
}

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-generation-parity MODEL_DIR FIXTURE_DIR auto|cpu|vulkan\n";
        return 2;
    }
    skintokens::runtime_options runtime; runtime.device = device(argv[3]);
    const bool vulkan = runtime.device == skintokens::device_kind::vulkan;
    auto backend = skintokens::detail::make_backend(runtime);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "tokenrig.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);
    const auto fixtures = std::filesystem::path{argv[2]};
    const auto full_mesh = fixtures / "mesh-condition.f32";
    const auto mesh = read_values<float>(std::filesystem::exists(full_mesh) ? full_mesh :
                                         fixtures / "mesh-encoder-output.f32");
    const auto full_prefix = fixtures / "token-generation-prefix.i32";
    const std::vector<std::int32_t> prefix = std::filesystem::exists(full_prefix) ?
        read_values<std::int32_t>(full_prefix) :
        std::vector<std::int32_t>{257, 263, 128, 128, 128, 128, 196, 128, 258};
    auto expected = read_values<std::int32_t>(fixtures / "token-generation-codes.i32");
    if (expected.empty() || expected.size() % 4U != 0U)
        throw std::runtime_error("generation fixture must contain four codes per joint");
    skintokens::generation_options options;
    options.temperature = 0.0F; options.top_k = 0U; options.beams = 1U; options.max_tokens = 8U;
    options.repetition_penalty = 1.0F; options.max_tokens = expected.size();
    auto actual = skintokens::detail::generate_skin_codes(**weights, (*backend)->value,
        mesh, prefix, expected.size() / 4U, options);
    if (!actual) throw std::runtime_error(actual.error().message);
    const auto reference_logits = read_values<float>(fixtures / "token-generation-logits.f32");
    if (reference_logits.size() != expected.size() * 33036U)
        throw std::runtime_error("generation logits fixture size mismatch");
    std::vector<std::int32_t> context{prefix};
    bool logits_pass = true;
    for (std::size_t step = 0; step < expected.size(); ++step) {
        auto logits = skintokens::detail::run_qwen_logits(**weights, (*backend)->value, mesh, context);
        if (!logits) throw std::runtime_error(logits.error().message);
        double numerator = 0.0, denominator = 0.0; float maximum = 0.0F;
        const float * reference = reference_logits.data() + step * 33036U;
        for (std::size_t index = 0; index < 33036U; ++index) {
            const float difference = (*logits)[index] - reference[index];
            maximum = std::max(maximum, std::abs(difference));
            numerator += static_cast<double>(difference) * difference;
            denominator += static_cast<double>(reference[index]) * reference[index];
        }
        const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
        const auto expected_top = static_cast<std::int32_t>(std::distance(reference,
            std::max_element(reference + 267, reference + 33035)));
        const auto actual_top = static_cast<std::int32_t>(std::distance(logits->begin(),
            std::max_element(logits->begin() + 267, logits->begin() + 33035)));
        std::cout << "step_" << (step + 1U) << " max_abs=" << maximum << " rel_l2=" << relative
                  << " top=" << actual_top << " reference_top=" << expected_top << '\n';
        logits_pass = logits_pass && std::isfinite(relative) &&
            relative < (vulkan ? 0.001 : 2e-5) && maximum < (vulkan ? 0.005F : 1e-4F);
        context.push_back(expected[step]);
    }
    for (auto & token : expected) token -= 267;
    std::cout << "expected:"; for (auto value : expected) std::cout << ' ' << value;
    std::cout << "\nactual:  "; for (auto value : *actual) std::cout << ' ' << value;
    std::cout << '\n';
    return *actual == expected && logits_pass ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << exception.what() << '\n'; return 1;
}
