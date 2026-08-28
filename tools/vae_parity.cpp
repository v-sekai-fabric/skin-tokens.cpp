#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {

std::vector<float> read_values(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open fixture: " + path.string());
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::size_t>(size) % sizeof(float) != 0U)
        throw std::runtime_error("invalid F32 fixture: " + path.string());
    std::vector<float> output(static_cast<std::size_t>(size) / sizeof(float));
    stream.seekg(0);
    if (!output.empty()) stream.read(reinterpret_cast<char *>(output.data()), size);
    if (!stream) throw std::runtime_error("truncated F32 fixture: " + path.string());
    return output;
}

skintokens::device_kind parse_device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    if (value == "auto") return skintokens::device_kind::automatic;
    throw std::runtime_error("device must be auto, cpu, or vulkan");
}

bool compare(std::string_view name, const std::vector<float> & actual,
             const std::vector<float> & expected) {
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string{name} + " fixture size mismatch");
    double numerator = 0.0, denominator = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float difference = actual[index] - expected[index];
        maximum = std::max(maximum, std::abs(difference));
        numerator += static_cast<double>(difference) * difference;
        denominator += static_cast<double>(expected[index]) * expected[index];
    }
    const double relative = std::sqrt(numerator / std::max(denominator, 1e-30));
    std::cout << name << " max_abs=" << maximum << " rel_l2=" << relative << '\n';
    return std::isfinite(relative) && relative <= 0.02 && maximum <= 0.1F;
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-vae-parity MODEL_DIR FIXTURE_DIR auto|cpu|vulkan\n";
        return 2;
    }
    skintokens::runtime_options options; options.device = parse_device(argv[3]);
    auto backend = skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "skin-vae.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);
    const std::filesystem::path fixtures = argv[2];
    auto actual = skintokens::detail::run_skin_vae_fixture(**weights, (*backend)->value,
        read_values(fixtures / "skin-vae-points.f32"), read_values(fixtures / "skin-vae-normals.f32"));
    if (!actual) throw std::runtime_error(actual.error().message);
    bool passed = true;
    constexpr const char * names[]{"condition_embedding", "cond_projected", "cond_cross",
        "cond_self_01", "cond_self_02", "cond_norm", "cond_latents", "codes", "decoder_input",
        "decoder_self_01", "decoder_self_02", "decoder_self_03", "decoder_self_04", "decoder_self_05",
        "decoder_self_06", "decoder_self_07", "decoder_self_08", "decoder_self_09", "decoder_self_10",
        "decoder_cache", "decoder_cross", "output"};
    for (const char * name : names) {
        const auto found = actual->find(name);
        if (found == actual->end()) throw std::runtime_error(std::string{"missing snapshot: "} + name);
        passed = compare(name, found->second,
            read_values(fixtures / (std::string{"skin-vae-"} + name + ".f32"))) && passed;
    }
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << exception.what() << '\n';
    return 1;
}
