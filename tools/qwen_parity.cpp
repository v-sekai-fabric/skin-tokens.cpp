#include "internal.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace {

std::vector<float> read_values(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open fixture: " + path.string());
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::size_t>(size) % sizeof(float) != 0U)
        throw std::runtime_error("invalid F32 fixture: " + path.string());
    std::vector<float> result(static_cast<std::size_t>(size) / sizeof(float));
    stream.seekg(0);
    if (!result.empty()) stream.read(reinterpret_cast<char *>(result.data()), size);
    if (!stream) throw std::runtime_error("truncated F32 fixture: " + path.string());
    return result;
}

skintokens::device_kind parse_device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    return skintokens::device_kind::automatic;
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-qwen-parity MODEL_DIR FIXTURE_DIR auto|cpu|vulkan\n";
        return 2;
    }
    skintokens::runtime_options options;
    options.device = parse_device(argv[3]);
    auto backend = skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "tokenrig.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);
    const std::filesystem::path fixtures = argv[2];
    auto input = read_values(fixtures / "qwen-layer0-input.f32");
    constexpr std::size_t sequence = 9U;
    auto actual = skintokens::detail::run_qwen_layer0(**weights, (*backend)->value, input, sequence);
    if (!actual) throw std::runtime_error(actual.error().message);
    bool passed = true;
    const char * names[] = {"attn_norm", "q_linear", "k_linear", "v_linear", "q_norm", "k_norm",
                            "q_rope", "k_rope", "scores", "probabilities", "attention",
                            "o_linear", "ffn_norm", "gate_linear", "up_linear", "down_linear", "layer_output"};
    for (const char * name : names) {
        auto expected = read_values(fixtures / (std::string{"qwen-layer0-"} + name + ".f32"));
        const auto found = actual->find(name);
        if (found == actual->end() || found->second.size() != expected.size())
            throw std::runtime_error(std::string{"snapshot size mismatch: "} + name);
        double numerator = 0.0;
        double denominator = 0.0;
        float maximum = 0.0F;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const float difference = found->second[index] - expected[index];
            maximum = std::max(maximum, std::abs(difference));
            numerator += static_cast<double>(difference) * difference;
            denominator += static_cast<double>(expected[index]) * expected[index];
        }
        const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
        std::cout << name << " max_abs=" << maximum << " rel_l2=" << relative << '\n';
        // F16 GGUF weights are compared to the released BF16 checkpoint. This
        // broad gate is only the first-operation diagnostic; component-specific
        // acceptance thresholds are tightened after locating any divergence.
        if (!std::isfinite(relative) || relative > 0.08 || maximum > 0.5F) passed = false;
    }
    std::vector<float> state = input;
    for (std::size_t layer = 0; layer < 28U; ++layer) {
        auto output = skintokens::detail::run_qwen_layer(**weights, (*backend)->value, state, sequence, layer);
        if (!output) throw std::runtime_error(output.error().message);
        char filename[64];
        std::snprintf(filename, sizeof(filename), "qwen-layer0-hidden_%02zu.f32", layer + 1U);
        const auto expected = read_values(fixtures / filename);
        if (expected.size() != output->size()) throw std::runtime_error("hidden-state fixture size mismatch");
        double numerator = 0.0, denominator = 0.0;
        float maximum = 0.0F;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const float difference = (*output)[index] - expected[index];
            maximum = std::max(maximum, std::abs(difference));
            numerator += static_cast<double>(difference) * difference;
            denominator += static_cast<double>(expected[index]) * expected[index];
        }
        const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
        std::cout << "hidden_" << (layer + 1U) << " max_abs=" << maximum << " rel_l2=" << relative << '\n';
        if (!std::isfinite(relative) || relative > 0.12 || maximum > 1.0F) passed = false;
        state = std::move(*output);
    }
    auto logits = skintokens::detail::run_qwen_head(**weights, (*backend)->value, state, sequence);
    if (!logits) throw std::runtime_error(logits.error().message);
    const auto expected_logits = read_values(fixtures / "qwen-layer0-logits.f32");
    if (expected_logits.size() != logits->size()) throw std::runtime_error("logit fixture size mismatch");
    double numerator = 0.0, denominator = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < logits->size(); ++index) {
        const float difference = (*logits)[index] - expected_logits[index];
        maximum = std::max(maximum, std::abs(difference));
        numerator += static_cast<double>(difference) * difference;
        denominator += static_cast<double>(expected_logits[index]) * expected_logits[index];
    }
    const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
    std::cout << "logits max_abs=" << maximum << " rel_l2=" << relative << '\n';
    if (!std::isfinite(relative) || relative > 0.15 || maximum > 2.0F) passed = false;
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << exception.what() << '\n';
    return 1;
}
