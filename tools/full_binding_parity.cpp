#include "binding.hpp"
#include "internal.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

template<class T>
std::vector<T> read_array(const std::filesystem::path & path, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::runtime_error("fixture array size overflows");
    const auto bytes = count * sizeof(T);
    std::error_code error;
    if (std::filesystem::file_size(path, error) != bytes || error)
        throw std::runtime_error("unexpected fixture size: " + path.string());
    std::vector<T> output(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("cannot read fixture: " + path.string());
    return output;
}

skintokens::device_kind parse_device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    throw std::runtime_error("device must be cpu or vulkan");
}

struct difference {
    double maximum = 0.0;
    double relative_l2 = 0.0;
};

difference compare(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison shape mismatch");
    double numerator = 0.0;
    double denominator = 0.0;
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double delta = static_cast<double>(actual[index]) - expected[index];
        maximum = std::max(maximum, std::abs(delta));
        numerator += delta * delta;
        denominator += static_cast<double>(expected[index]) * expected[index];
    }
    return {maximum, std::sqrt(numerator / std::max(denominator, 1.0e-30))};
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-full-binding-parity MODEL_DIR FIXTURE_DIR cpu|vulkan\n";
        return 2;
    }
    const std::filesystem::path fixture = argv[2];
    std::ifstream manifest_stream(fixture / "animation-vertices.json");
    if (!manifest_stream) throw std::runtime_error("cannot open fixture manifest");
    nlohmann::json manifest;
    manifest_stream >> manifest;
    const auto vertex_count = manifest.at("vertex_count").get<std::size_t>();
    const auto joint_count = manifest.at("joint_count").get<std::size_t>();
    constexpr std::size_t sampled_count = 54'000U;
    constexpr std::size_t dimensions = 3U;
    if (vertex_count == 0U || vertex_count > 20'000'000U || joint_count == 0U || joint_count > 256U)
        throw std::runtime_error("unsafe fixture dimensions");

    const auto flat_points = read_array<float>(fixture / "skin-points.f32", sampled_count * dimensions);
    const auto flat_normals = read_array<float>(fixture / "skin-normals.f32", sampled_count * dimensions);
    const auto queries_bytes = std::filesystem::file_size(fixture / "skin-queries.i32");
    if (queries_bytes == 0U || queries_bytes % sizeof(std::int32_t) != 0U)
        throw std::runtime_error("invalid condition query fixture");
    const auto queries = read_array<std::int32_t>(
        fixture / "skin-queries.i32", queries_bytes / sizeof(std::int32_t));
    const auto codes = read_array<std::int32_t>(fixture / "skin-codes.i32", joint_count * 4U);
    const auto expected_condition = read_array<float>(
        fixture / "reference-condition-latents.f32", queries.size() * 512U);
    const auto expected_sampled = read_array<float>(
        fixture / "reference-sampled-skin.f32", sampled_count * joint_count);
    const auto flat_normalized = read_array<float>(
        fixture / "normalized-vertices.f32", vertex_count * dimensions);
    const auto expected_joints = read_array<std::uint16_t>(
        fixture / "reference-joints.u16", vertex_count * 4U);
    const auto expected_weights = read_array<float>(
        fixture / "reference-weights.f32", vertex_count * 4U);

    std::vector<skintokens::vec3> points(sampled_count), normals(sampled_count), normalized(vertex_count);
    for (std::size_t index = 0; index < sampled_count; ++index) {
        points[index] = {flat_points[index * 3U], flat_points[index * 3U + 1U], flat_points[index * 3U + 2U]};
        normals[index] = {flat_normals[index * 3U], flat_normals[index * 3U + 1U], flat_normals[index * 3U + 2U]};
    }
    for (std::size_t index = 0; index < vertex_count; ++index)
        normalized[index] = {flat_normalized[index * 3U], flat_normalized[index * 3U + 1U], flat_normalized[index * 3U + 2U]};

    skintokens::runtime_options options;
    options.device = parse_device(argv[3]);
    auto backend = skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "skin-vae.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);

    auto condition = skintokens::detail::encode_skin_condition(
        **weights, (*backend)->value, points, normals, queries);
    if (!condition) throw std::runtime_error(condition.error().message);
    const auto condition_error = compare(*condition, expected_condition);
    std::cout << "condition max_abs=" << condition_error.maximum
              << " relative_l2=" << condition_error.relative_l2 << '\n';

    std::vector<std::vector<float>> dense(joint_count);
    std::vector<float> actual_sampled(sampled_count * joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint) {
        auto decoded = skintokens::detail::decode_skin_joint(
            **weights, (*backend)->value,
            std::span<const std::int32_t>{codes}.subspan(joint * 4U, 4U), *condition, points, normals);
        if (!decoded) throw std::runtime_error(decoded.error().message);
        dense[joint] = std::move(*decoded);
        for (std::size_t point = 0; point < sampled_count; ++point)
            actual_sampled[point * joint_count + joint] = dense[joint][point];
        std::cout << "decoded joint " << joint + 1U << '/' << joint_count << '\r' << std::flush;
    }
    std::cout << '\n';
    const auto sampled_error = compare(actual_sampled, expected_sampled);
    std::cout << "sampled_skin max_abs=" << sampled_error.maximum
              << " relative_l2=" << sampled_error.relative_l2 << '\n';

    skintokens::skeleton target;
    target.names.resize(joint_count);
    target.parents.resize(joint_count, -1);
    target.rest_positions.resize(joint_count);
    const auto binding = skintokens::detail::integrate_learned_binding(
        target, normalized, points, dense);
    std::vector<float> actual_dense(vertex_count * joint_count, 0.0F);
    std::vector<float> expected_dense(vertex_count * joint_count, 0.0F);
    std::size_t exact_vertices = 0U;
    std::size_t shared_slots = 0U;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        bool exact = true;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const auto actual_joint = binding.joints[vertex][slot];
            const auto expected_joint = expected_joints[vertex * 4U + slot];
            actual_dense[vertex * joint_count + actual_joint] += binding.weights[vertex][slot];
            expected_dense[vertex * joint_count + expected_joint] += expected_weights[vertex * 4U + slot];
        }
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            const bool actual = actual_dense[vertex * joint_count + joint] > 0.0F;
            const bool expected = expected_dense[vertex * joint_count + joint] > 0.0F;
            shared_slots += actual && expected ? 1U : 0U;
            exact = exact && actual == expected;
        }
        exact_vertices += exact ? 1U : 0U;
    }
    const auto binding_error = compare(actual_dense, expected_dense);
    const double exact_fraction = static_cast<double>(exact_vertices) / vertex_count;
    const double slot_fraction = static_cast<double>(shared_slots) / (vertex_count * 4U);
    std::cout << "binding max_abs=" << binding_error.maximum
              << " relative_l2=" << binding_error.relative_l2
              << " exact_top4_vertices=" << exact_fraction
              << " shared_top4_slots=" << slot_fraction << '\n';

    const bool cpu = options.device == skintokens::device_kind::cpu;
    const bool passed = cpu ?
        (condition_error.maximum <= 1.0e-5 && condition_error.relative_l2 <= 1.0e-5 &&
         sampled_error.maximum <= 2.0e-4 && sampled_error.relative_l2 <= 2.0e-5 &&
         binding_error.maximum <= 1.0e-2 && binding_error.relative_l2 <= 1.0e-3 &&
         exact_fraction >= 1.0 && slot_fraction >= 1.0) :
        (condition_error.maximum <= 1.0e-4 && condition_error.relative_l2 <= 3.0e-4 &&
         sampled_error.maximum <= 1.6e-2 && sampled_error.relative_l2 <= 1.2e-3 &&
         binding_error.maximum <= 0.26 && binding_error.relative_l2 <= 0.05 &&
         exact_fraction >= 0.99 && slot_fraction >= 0.998);
    std::cout << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << "full binding parity: " << exception.what() << '\n';
    return 2;
}
