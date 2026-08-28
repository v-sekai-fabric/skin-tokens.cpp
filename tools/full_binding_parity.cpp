#include "binding.hpp"
#include "internal.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
    const auto face_count = manifest.at("face_count").get<std::size_t>();
    const auto joint_count = manifest.at("joint_count").get<std::size_t>();
    const auto frame_count = manifest.at("frame_count").get<std::size_t>();
    const auto selected_frames = manifest.at("selected_frames").get<std::vector<std::size_t>>();
    constexpr std::size_t sampled_count = 54'000U;
    constexpr std::size_t dimensions = 3U;
    if (vertex_count == 0U || vertex_count > 20'000'000U || joint_count == 0U || joint_count > 256U)
        throw std::runtime_error("unsafe fixture dimensions");
    if (frame_count == 0U || frame_count > 100'000U || selected_frames.empty())
        throw std::runtime_error("unsafe animation fixture dimensions");
    for (const auto frame : selected_frames)
        if (frame >= frame_count) throw std::runtime_error("selected frame is out of range");

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
    const auto flat_normalized_joints = read_array<float>(
        fixture / "normalized-joints.f32", joint_count * dimensions);
    const auto faces = read_array<skintokens::triangle>(fixture / "faces.u32", face_count);
    const auto expected_surface = read_array<float>(
        fixture / "reference-surface-weights.f32", vertex_count * joint_count);
    const auto voxel_count = manifest.at("voxel_count").get<std::size_t>();
    const auto expected_voxels = read_array<std::int32_t>(
        fixture / "reference-voxel-coordinates.i32", voxel_count * 3U);
    const auto expected_seeds = read_array<std::uint32_t>(
        fixture / "reference-surface-seeds.u32", joint_count);
    const auto expected_joints = read_array<std::uint16_t>(
        fixture / "reference-joints.u16", vertex_count * 4U);
    const auto expected_weights = read_array<float>(
        fixture / "reference-weights.f32", vertex_count * 4U);
    const auto expected_final_dense = read_array<float>(
        fixture / "reference-dense-weights.f32", vertex_count * joint_count);
    const auto positions = read_array<float>(fixture / "positions.f32", vertex_count * 3U);
    const auto parents = read_array<std::int32_t>(fixture / "parents.i32", joint_count);
    const auto rest_positions = read_array<float>(
        fixture / "rest-positions.f32", joint_count * 3U);
    const auto root_translations = read_array<float>(
        fixture / "root-translations.f32", frame_count * 3U);
    const auto local_rotations = read_array<float>(
        fixture / "local-rotations.f32", frame_count * joint_count * 4U);
    const auto expected_animated = read_array<float>(
        fixture / "expected-vertices.f32", selected_frames.size() * vertex_count * 3U);

    std::vector<skintokens::vec3> points(sampled_count), normals(sampled_count), normalized(vertex_count),
        normalized_joints(joint_count);
    for (std::size_t index = 0; index < sampled_count; ++index) {
        points[index] = {flat_points[index * 3U], flat_points[index * 3U + 1U], flat_points[index * 3U + 2U]};
        normals[index] = {flat_normals[index * 3U], flat_normals[index * 3U + 1U], flat_normals[index * 3U + 2U]};
    }
    for (std::size_t index = 0; index < vertex_count; ++index)
        normalized[index] = {flat_normalized[index * 3U], flat_normalized[index * 3U + 1U], flat_normalized[index * 3U + 2U]};
    for (std::size_t index = 0; index < joint_count; ++index)
        normalized_joints[index] = {flat_normalized_joints[index * 3U], flat_normalized_joints[index * 3U + 1U],
                                     flat_normalized_joints[index * 3U + 2U]};

    skintokens::runtime_options options;
    options.device = parse_device(argv[3]);
    auto backend = skintokens::detail::make_backend(options);
    if (!backend) throw std::runtime_error(backend.error().message);
    auto weights = skintokens::detail::load_component(
        std::filesystem::path{argv[1]} / "skin-vae.gguf", (*backend)->value);
    if (!weights) throw std::runtime_error(weights.error().message);

    std::vector<std::vector<float>> dense(joint_count);
    std::vector<float> actual_sampled(sampled_count * joint_count);
    difference condition_error{};
    const bool surface_only = std::getenv("SKINTOKENS_PARITY_SURFACE_ONLY") != nullptr;
    if (surface_only) {
        actual_sampled = expected_sampled;
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            dense[joint].resize(sampled_count);
            for (std::size_t point = 0; point < sampled_count; ++point)
                dense[joint][point] = expected_sampled[point * joint_count + joint];
        }
        std::cout << "condition skipped (surface-only diagnostic)\n";
    } else {
        auto condition = skintokens::detail::encode_skin_condition(
            **weights, (*backend)->value, points, normals, queries);
        if (!condition) throw std::runtime_error(condition.error().message);
        condition_error = compare(*condition, expected_condition);
        std::cout << "condition max_abs=" << condition_error.maximum
                  << " relative_l2=" << condition_error.relative_l2 << '\n';
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
    }
    const auto sampled_error = compare(actual_sampled, expected_sampled);
    std::cout << "sampled_skin max_abs=" << sampled_error.maximum
              << " relative_l2=" << sampled_error.relative_l2 << '\n';

    skintokens::skeleton target;
    target.names.resize(joint_count);
    target.parents = parents;
    target.rest_positions.resize(joint_count);
    for (std::size_t joint=0; joint<joint_count; ++joint) {
        target.names[joint] = "joint_" + std::to_string(joint);
        target.rest_positions[joint] = {rest_positions[joint*3U], rest_positions[joint*3U+1U],
                                        rest_positions[joint*3U+2U]};
    }
    skintokens::detail::binding_trace binding_trace;
    const auto binding = skintokens::detail::integrate_postprocessed_binding(
        target, normalized, faces, normalized_joints, points, dense, &binding_trace);
    const auto surface_error = compare(binding_trace.surface_weights, expected_surface);
    std::cout << "surface_weights max_abs=" << surface_error.maximum
              << " relative_l2=" << surface_error.relative_l2 << '\n';
    if (std::filesystem::exists(fixture / "reference-mesh-surface-weights.f32")) {
        const auto mesh_surface = read_array<float>(
            fixture / "reference-mesh-surface-weights.f32", vertex_count * joint_count);
        const auto mesh_error = compare(binding_trace.surface_weights, mesh_surface);
        std::cout << "mesh_surface_diagnostic max_abs=" << mesh_error.maximum
                  << " relative_l2=" << mesh_error.relative_l2 << '\n';
    }
    using voxel = std::array<std::int32_t,3>;
    std::vector<voxel> actual_voxels(binding_trace.voxel_coordinates.size()/3U), reference_voxels(voxel_count);
    for (std::size_t index=0; index<actual_voxels.size(); ++index)
        actual_voxels[index]={binding_trace.voxel_coordinates[index*3U],binding_trace.voxel_coordinates[index*3U+1U],binding_trace.voxel_coordinates[index*3U+2U]};
    for (std::size_t index=0; index<reference_voxels.size(); ++index)
        reference_voxels[index]={expected_voxels[index*3U],expected_voxels[index*3U+1U],expected_voxels[index*3U+2U]};
    const bool voxel_order_exact=actual_voxels==reference_voxels;
    std::sort(actual_voxels.begin(),actual_voxels.end());
    std::sort(reference_voxels.begin(),reference_voxels.end());
    const bool voxel_exact=actual_voxels==reference_voxels;
    std::size_t seed_position_matches=0U;
    std::size_t mesh_seed_indices=0U,mesh_seed_index_matches=0U;
    const auto seed_position=[&](std::uint32_t seed, std::span<const std::int32_t> voxel_values) {
        if (seed<vertex_count) return normalized[seed];
        const auto offset=(static_cast<std::size_t>(seed)-vertex_count)*3U;
        return skintokens::vec3{static_cast<float>(voxel_values[offset]),static_cast<float>(voxel_values[offset+1U]),static_cast<float>(voxel_values[offset+2U])};
    };
    for (std::size_t joint=0; joint<joint_count; ++joint) {
        const auto actual=seed_position(binding_trace.surface_seeds[joint],binding_trace.voxel_coordinates);
        const auto expected=seed_position(expected_seeds[joint],expected_voxels);
        seed_position_matches += actual.x==expected.x && actual.y==expected.y && actual.z==expected.z ? 1U : 0U;
        if (expected_seeds[joint]<vertex_count) {
            ++mesh_seed_indices;
            mesh_seed_index_matches += binding_trace.surface_seeds[joint]==expected_seeds[joint] ? 1U : 0U;
        }
    }
    std::cout << "voxels actual=" << actual_voxels.size() << " expected=" << reference_voxels.size()
              << " exact_set=" << voxel_exact << " exact_order=" << voxel_order_exact
              << " seed_positions=" << seed_position_matches << '/' << joint_count
              << " mesh_seed_indices=" << mesh_seed_index_matches << '/' << mesh_seed_indices << '\n';
    const auto final_dense_error = compare(binding_trace.final_dense_weights, expected_final_dense);
    std::cout << "final_dense max_abs=" << final_dense_error.maximum
              << " relative_l2=" << final_dense_error.relative_l2 << '\n';
    std::vector<float> actual_dense(vertex_count * joint_count, 0.0F);
    std::vector<float> expected_dense(vertex_count * joint_count, 0.0F);
    std::size_t exact_vertices = 0U;
    std::size_t shared_slots = 0U;
    std::size_t first_mismatch = vertex_count;
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
        if (!exact && first_mismatch == vertex_count) first_mismatch = vertex;
    }
    const auto binding_error = compare(actual_dense, expected_dense);
    const double exact_fraction = static_cast<double>(exact_vertices) / vertex_count;
    const double slot_fraction = static_cast<double>(shared_slots) / (vertex_count * 4U);
    std::cout << "binding max_abs=" << binding_error.maximum
              << " relative_l2=" << binding_error.relative_l2
              << " exact_top4_vertices=" << exact_fraction
              << " shared_top4_slots=" << slot_fraction;
    if (first_mismatch != vertex_count) {
        std::cout << " first_mismatch=" << first_mismatch << " actual=[";
        for (std::size_t slot=0; slot<4U; ++slot)
            std::cout << (slot ? "," : "") << binding.joints[first_mismatch][slot]
                      << ':' << binding.weights[first_mismatch][slot];
        std::cout << "] expected=[";
        for (std::size_t slot=0; slot<4U; ++slot)
            std::cout << (slot ? "," : "") << expected_joints[first_mismatch*4U+slot]
                      << ':' << expected_weights[first_mismatch*4U+slot];
        std::cout << ']';
    }
    std::cout << '\n';

    // End-to-end acceptance: deform the original vertices with the binding
    // computed above and the exact captured animation tracks. This catches
    // joint-index, bind-pose and matrix-convention errors that a weight-only
    // comparison cannot see.
    skintokens::mesh geometry;
    geometry.vertices.resize(vertex_count);
    for (std::size_t vertex=0; vertex<vertex_count; ++vertex)
        geometry.vertices[vertex] = {positions[vertex*3U], positions[vertex*3U+1U],
                                     positions[vertex*3U+2U]};
    skintokens::motion animation;
    animation.frames = frame_count;
    animation.frames_per_second = manifest.value("frames_per_second", 30.0F);
    animation.rig = target;
    animation.root_translations.resize(frame_count);
    animation.local_rotations.resize(frame_count * joint_count);
    for (std::size_t frame=0; frame<frame_count; ++frame) {
        animation.root_translations[frame] = {root_translations[frame*3U], root_translations[frame*3U+1U],
                                               root_translations[frame*3U+2U]};
        for (std::size_t joint=0; joint<joint_count; ++joint) {
            const auto offset=(frame*joint_count+joint)*4U;
            animation.local_rotations[frame*joint_count+joint] = {
                local_rotations[offset], local_rotations[offset+1U],
                local_rotations[offset+2U], local_rotations[offset+3U]};
        }
    }
    std::vector<float> actual_animated;
    actual_animated.reserve(expected_animated.size());
    for (const auto frame : selected_frames) {
        auto deformed = skintokens::deform_vertices(geometry, binding, animation, frame);
        if (!deformed) throw std::runtime_error(deformed.error().message);
        for (const auto value : *deformed) {
            actual_animated.push_back(value.x);
            actual_animated.push_back(value.y);
            actual_animated.push_back(value.z);
        }
    }
    const auto animated_error = compare(actual_animated, expected_animated);
    std::cout << "animated_vertices max_abs=" << animated_error.maximum
              << " relative_l2=" << animated_error.relative_l2 << '\n';

    const bool cpu = options.device == skintokens::device_kind::cpu;
    const bool passed = cpu ?
        (condition_error.maximum <= 1.0e-5 && condition_error.relative_l2 <= 1.0e-5 &&
         sampled_error.maximum <= 2.0e-4 && sampled_error.relative_l2 <= 2.0e-5 &&
         surface_error.maximum <= 1.0e-4 && surface_error.relative_l2 <= 1.0e-4 &&
         final_dense_error.maximum <= 1.0e-4 && final_dense_error.relative_l2 <= 1.0e-4 &&
         binding_error.maximum <= 1.0e-4 && binding_error.relative_l2 <= 1.0e-4 &&
         animated_error.maximum <= 1.0e-4 && animated_error.relative_l2 <= 1.0e-5 &&
         exact_fraction >= 1.0 && slot_fraction >= 1.0) :
        (condition_error.maximum <= 1.0e-4 && condition_error.relative_l2 <= 3.0e-4 &&
         sampled_error.maximum <= 1.6e-2 && sampled_error.relative_l2 <= 1.2e-3 &&
         surface_error.maximum <= 1.0e-4 && surface_error.relative_l2 <= 1.0e-4 &&
         final_dense_error.maximum <= 2.0e-2 && final_dense_error.relative_l2 <= 2.0e-3 &&
         binding_error.maximum <= 0.26 && binding_error.relative_l2 <= 1.0e-2 &&
         animated_error.maximum <= 0.16 && animated_error.relative_l2 <= 1.0e-3 &&
         exact_fraction >= 0.995 && slot_fraction >= 0.999);
    std::cout << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << "full binding parity: " << exception.what() << '\n';
    return 2;
}
