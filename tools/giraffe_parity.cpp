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
#include <unordered_map>
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
    double numerator = 0.0, denominator = 0.0, maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double delta = static_cast<double>(actual[index]) - expected[index];
        maximum = std::max(maximum, std::abs(delta));
        numerator += delta * delta;
        denominator += static_cast<double>(expected[index]) * expected[index];
    }
    return {maximum, std::sqrt(numerator / std::max(denominator, 1.0e-30))};
}

struct binding_difference {
    difference dense;
    double exact_vertices = 0.0;
    double shared_slots = 0.0;
};

binding_difference compare_binding(
    const skintokens::skin & actual,
    std::span<const std::uint16_t> expected_joints,
    std::span<const float> expected_weights,
    std::size_t joint_count) {
    const auto vertex_count = actual.joints.size();
    if (actual.weights.size() != vertex_count || expected_joints.size() != vertex_count * 4U ||
        expected_weights.size() != vertex_count * 4U)
        throw std::runtime_error("binding comparison shape mismatch");
    std::vector<float> actual_dense(vertex_count * joint_count, 0.0F);
    std::vector<float> expected_dense(vertex_count * joint_count, 0.0F);
    std::size_t exact = 0U, shared = 0U;
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const auto actual_joint = actual.joints[vertex][slot];
            const auto expected_joint = expected_joints[vertex * 4U + slot];
            if (actual_joint >= joint_count || expected_joint >= joint_count)
                throw std::runtime_error("fixture binding joint is out of range");
            actual_dense[vertex * joint_count + actual_joint] += actual.weights[vertex][slot];
            expected_dense[vertex * joint_count + expected_joint] += expected_weights[vertex * 4U + slot];
        }
        bool vertex_exact = true;
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            const bool a = actual_dense[vertex * joint_count + joint] > 0.0F;
            const bool e = expected_dense[vertex * joint_count + joint] > 0.0F;
            shared += a && e ? 1U : 0U;
            vertex_exact = vertex_exact && a == e;
        }
        exact += vertex_exact ? 1U : 0U;
    }
    return {compare(actual_dense, expected_dense),
            static_cast<double>(exact) / static_cast<double>(vertex_count),
            static_cast<double>(shared) / static_cast<double>(vertex_count * 4U)};
}

void print_binding(std::string_view name, const binding_difference & value) {
    std::cout << name << " max_abs=" << value.dense.maximum
              << " relative_l2=" << value.dense.relative_l2
              << " exact_top4_vertices=" << value.exact_vertices
              << " shared_top4_slots=" << value.shared_slots << '\n';
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: skintokens-giraffe-parity MODEL_DIR FIXTURE_DIR cpu|vulkan\n";
        return 2;
    }
    const std::filesystem::path fixture = argv[2];
    std::ifstream manifest_stream(fixture / "manifest.json");
    if (!manifest_stream) throw std::runtime_error("cannot open giraffe fixture manifest");
    nlohmann::json manifest;
    manifest_stream >> manifest;
    const auto vertex_count = manifest.at("vertex_count").get<std::size_t>();
    const auto face_count = manifest.at("face_count").get<std::size_t>();
    const auto joint_count = manifest.at("joint_count").get<std::size_t>();
    const auto sample_count = manifest.at("sample_count").get<std::size_t>();
    const auto condition_tokens = manifest.at("condition_token_count").get<std::size_t>();
    const auto condition_width = manifest.at("condition_width").get<std::size_t>();
    const auto code_count = manifest.at("skin_code_count").get<std::size_t>();
    if (vertex_count == 0U || vertex_count > 20'000'000U || face_count == 0U ||
        joint_count == 0U || joint_count > 256U || sample_count != 54'000U ||
        condition_tokens == 0U || condition_tokens > 4096U || condition_width != 512U ||
        code_count != joint_count * 4U)
        throw std::runtime_error("unsafe or incompatible giraffe fixture dimensions");

    const auto flat_points = read_array<float>(fixture / "sampled-points.f32", sample_count * 3U);
    const auto flat_normals = read_array<float>(fixture / "sampled-normals.f32", sample_count * 3U);
    const auto condition = read_array<float>(
        fixture / "condition-latents.f32", condition_tokens * condition_width);
    const auto codes = read_array<std::int32_t>(fixture / "skin-codes.i32", code_count);
    const auto expected_sampled = read_array<float>(
        fixture / "f32-sampled-skin.f32", sample_count * joint_count);
    const auto released_bf16_sampled = read_array<float>(
        fixture / "sampled-skin.f32", sample_count * joint_count);
    const auto flat_vertices = read_array<float>(fixture / "vertices.f32", vertex_count * 3U);
    const auto precise_vertex_values = read_array<double>(fixture / "vertices.f64", vertex_count * 3U);
    const auto flat_joints = read_array<float>(fixture / "joints.f32", joint_count * 3U);
    const auto precise_joint_values = read_array<double>(fixture / "joints.f64", joint_count * 3U);
    const auto faces = read_array<skintokens::triangle>(fixture / "faces.u32", face_count);
    const auto parents = read_array<std::int32_t>(fixture / "parents.i32", joint_count);
    const auto expected_raw_dense = read_array<float>(
        fixture / "f32-raw-dense-weights.f32", vertex_count * joint_count);
    const auto expected_raw_joints = read_array<std::uint16_t>(
        fixture / "f32-raw-joints.u16", vertex_count * 4U);
    const auto expected_raw_weights = read_array<float>(
        fixture / "f32-raw-weights.f32", vertex_count * 4U);
    const auto released_bf16_raw_dense = read_array<float>(
        fixture / "raw-dense-weights.f32", vertex_count * joint_count);
    const auto released_bf16_raw_joints = read_array<std::uint16_t>(
        fixture / "raw-joints.u16", vertex_count * 4U);
    const auto released_bf16_raw_weights = read_array<float>(
        fixture / "raw-weights.f32", vertex_count * 4U);
    const auto expected_post_dense = read_array<float>(
        fixture / "f32-postprocessed-dense-weights.f32", vertex_count * joint_count);
    const auto expected_post_joints = read_array<std::uint16_t>(
        fixture / "f32-postprocessed-joints.u16", vertex_count * 4U);
    const auto expected_post_weights = read_array<float>(
        fixture / "f32-postprocessed-weights.f32", vertex_count * 4U);
    const auto voxel_count = manifest.at("voxel_count").get<std::size_t>();
    const auto expected_surface = read_array<float>(
        fixture / "f32-surface-weights.f32", vertex_count * joint_count);
    const auto expected_voxels = read_array<std::int32_t>(
        fixture / "f32-voxel-coordinates.i32", voxel_count * 3U);
    const auto expected_seeds = read_array<std::uint32_t>(
        fixture / "f32-surface-seeds.u32", joint_count);
    const auto expected_distance_f64 = read_array<double>(
        fixture / "f32-surface-distances.f64", joint_count * vertex_count);
    std::vector<float> expected_distances(expected_distance_f64.size());
    std::transform(expected_distance_f64.begin(), expected_distance_f64.end(),
                   expected_distances.begin(), [](double value) { return static_cast<float>(value); });
    const auto graph_nodes = manifest.at("graph_node_count").get<std::size_t>();
    const auto graph_edges = manifest.at("graph_edge_count").get<std::size_t>();
    const auto graph_indptr = read_array<std::int32_t>(fixture / "f32-graph-indptr.i32", graph_nodes + 1U);
    const auto graph_indices = read_array<std::int32_t>(fixture / "f32-graph-indices.i32", graph_edges);
    const auto graph_weights = read_array<double>(fixture / "f32-graph-weights.f64", graph_edges);

    std::vector<skintokens::vec3> points(sample_count), normals(sample_count),
        vertices(vertex_count), joints(joint_count);
    std::vector<skintokens::detail::precise_vec3> precise_vertices(vertex_count);
    std::vector<skintokens::detail::precise_vec3> precise_joints(joint_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        points[index] = {flat_points[index * 3U], flat_points[index * 3U + 1U], flat_points[index * 3U + 2U]};
        normals[index] = {flat_normals[index * 3U], flat_normals[index * 3U + 1U], flat_normals[index * 3U + 2U]};
    }
    for (std::size_t index = 0; index < vertex_count; ++index) {
        vertices[index] = {flat_vertices[index * 3U], flat_vertices[index * 3U + 1U], flat_vertices[index * 3U + 2U]};
        precise_vertices[index] = {precise_vertex_values[index * 3U], precise_vertex_values[index * 3U + 1U],
                                   precise_vertex_values[index * 3U + 2U]};
    }
    for (std::size_t index = 0; index < joint_count; ++index) {
        joints[index] = {flat_joints[index * 3U], flat_joints[index * 3U + 1U], flat_joints[index * 3U + 2U]};
        precise_joints[index] = {precise_joint_values[index * 3U], precise_joint_values[index * 3U + 1U],
                                 precise_joint_values[index * 3U + 2U]};
    }

    skintokens::runtime_options options;
    options.device = parse_device(argv[3]);
    std::vector<std::vector<float>> dense(joint_count);
    std::vector<float> actual_sampled(sample_count * joint_count);
    if (std::getenv("SKINTOKENS_GIRAFFE_SURFACE_ONLY") != nullptr) {
        actual_sampled = expected_sampled;
        for (std::size_t point = 0; point < sample_count; ++point)
            for (std::size_t joint = 0; joint < joint_count; ++joint) {
                if (dense[joint].empty()) dense[joint].resize(sample_count);
                dense[joint][point] = expected_sampled[point * joint_count + joint];
            }
        std::cout << "decoder skipped (surface-only diagnostic)\n";
    } else {
        auto backend = skintokens::detail::make_backend(options);
        if (!backend) throw std::runtime_error(backend.error().message);
        auto weights = skintokens::detail::load_component(
            std::filesystem::path{argv[1]} / "skin-vae.gguf", (*backend)->value);
        if (!weights) throw std::runtime_error(weights.error().message);
        for (std::size_t joint = 0; joint < joint_count; ++joint) {
            auto decoded = skintokens::detail::decode_skin_joint(
                **weights, (*backend)->value,
                std::span<const std::int32_t>{codes}.subspan(joint * 4U, 4U),
                condition, points, normals);
            if (!decoded) throw std::runtime_error(decoded.error().message);
            dense[joint] = std::move(*decoded);
            for (std::size_t point = 0; point < sample_count; ++point)
                actual_sampled[point * joint_count + joint] = dense[joint][point];
            std::cout << "decoded joint " << joint + 1U << '/' << joint_count << '\r' << std::flush;
        }
        std::cout << '\n';
    }
    const auto sampled_error = compare(actual_sampled, expected_sampled);
    std::cout << "sampled_skin max_abs=" << sampled_error.maximum
              << " relative_l2=" << sampled_error.relative_l2 << '\n';
    const auto released_bf16_error = compare(actual_sampled, released_bf16_sampled);
    std::cout << "released_bf16_sampled_diagnostic max_abs=" << released_bf16_error.maximum
              << " relative_l2=" << released_bf16_error.relative_l2 << '\n';

    skintokens::skeleton target;
    target.parents = parents;
    target.rest_positions = joints;
    target.names.resize(joint_count);
    for (std::size_t joint = 0; joint < joint_count; ++joint)
        target.names[joint] = "joint_" + std::to_string(joint);

    skintokens::detail::binding_trace raw_trace;
    const auto raw = skintokens::detail::integrate_learned_binding(
        target, vertices, points, dense, &raw_trace);
    const auto raw_dense_error = compare(raw_trace.final_dense_weights, expected_raw_dense);
    const auto raw_binding_error = compare_binding(
        raw, expected_raw_joints, expected_raw_weights, joint_count);
    std::cout << "raw_dense max_abs=" << raw_dense_error.maximum
              << " relative_l2=" << raw_dense_error.relative_l2 << '\n';
    print_binding("raw_binding", raw_binding_error);
    const auto released_bf16_raw_dense_error = compare(
        raw_trace.final_dense_weights, released_bf16_raw_dense);
    const auto released_bf16_raw_binding_error = compare_binding(
        raw, released_bf16_raw_joints, released_bf16_raw_weights, joint_count);
    std::cout << "released_bf16_raw_dense_diagnostic max_abs="
              << released_bf16_raw_dense_error.maximum << " relative_l2="
              << released_bf16_raw_dense_error.relative_l2 << '\n';
    print_binding("released_bf16_raw_binding_diagnostic", released_bf16_raw_binding_error);

    skintokens::detail::binding_trace post_trace;
    const auto post = skintokens::detail::integrate_postprocessed_binding_precise(
        target, vertices, precise_vertices, faces, joints, precise_joints, points, dense, &post_trace);
    const auto surface_error = compare(post_trace.surface_weights, expected_surface);
    std::cout << "surface_weights max_abs=" << surface_error.maximum
              << " relative_l2=" << surface_error.relative_l2 << '\n';
    const auto distance_error = compare(post_trace.surface_distances, expected_distances);
    std::size_t worst_distance=0U;
    for (std::size_t index=1; index<expected_distances.size(); ++index)
        if (std::abs(post_trace.surface_distances[index]-expected_distances[index]) >
            std::abs(post_trace.surface_distances[worst_distance]-expected_distances[worst_distance]))
            worst_distance=index;
    std::cout << "surface_distances max_abs=" << distance_error.maximum
              << " relative_l2=" << distance_error.relative_l2
              << " graph_edges=" << post_trace.surface_graph_edges << '/'
              << manifest.at("graph_edge_count").get<std::size_t>()
              << " max_distance=" << post_trace.maximum_surface_distance << '/'
              << manifest.at("maximum_surface_distance").get<double>()
              << " worst_joint=" << worst_distance/vertex_count
              << " worst_vertex=" << worst_distance%vertex_count
              << " actual=" << post_trace.surface_distances[worst_distance]
              << " expected=" << expected_distances[worst_distance] << '\n';
    std::unordered_map<std::uint64_t, double> reference_graph;
    for (std::size_t source=0; source<graph_nodes; ++source)
        for (std::int32_t offset=graph_indptr[source]; offset<graph_indptr[source+1U]; ++offset)
            reference_graph[(static_cast<std::uint64_t>(source)<<32U) |
                static_cast<std::uint32_t>(graph_indices[static_cast<std::size_t>(offset)])] =
                    graph_weights[static_cast<std::size_t>(offset)];
    std::unordered_map<std::uint64_t, float> actual_graph;
    for (std::size_t edge=0; edge<post_trace.surface_graph_sources.size(); ++edge)
        actual_graph[(static_cast<std::uint64_t>(post_trace.surface_graph_sources[edge])<<32U) |
            post_trace.surface_graph_targets[edge]] = post_trace.surface_graph_weights[edge];
    std::size_t missing_edges=0U, extra_edges=0U, printed=0U;
    double graph_weight_max=0.0;
    for (const auto [key, expected] : reference_graph) {
        const auto found=actual_graph.find(key);
        if (found==actual_graph.end()) {
            ++missing_edges;
            if (printed++<8U) std::cout << "missing_graph_edge " << (key>>32U) << "->"
                                       << static_cast<std::uint32_t>(key) << " weight=" << expected << '\n';
        } else graph_weight_max=std::max(graph_weight_max,std::abs(static_cast<double>(found->second)-expected));
    }
    for (const auto [key, unused] : actual_graph)
        extra_edges += reference_graph.contains(key) ? 0U : 1U;
    std::cout << "graph missing=" << missing_edges << " extra=" << extra_edges
              << " shared_weight_max_abs=" << graph_weight_max << '\n';
    using voxel = std::array<std::int32_t, 3>;
    std::vector<voxel> actual_voxels(post_trace.voxel_coordinates.size() / 3U);
    std::vector<voxel> reference_voxels(voxel_count);
    for (std::size_t index = 0; index < actual_voxels.size(); ++index)
        actual_voxels[index] = {post_trace.voxel_coordinates[index * 3U],
            post_trace.voxel_coordinates[index * 3U + 1U], post_trace.voxel_coordinates[index * 3U + 2U]};
    for (std::size_t index = 0; index < reference_voxels.size(); ++index)
        reference_voxels[index] = {expected_voxels[index * 3U], expected_voxels[index * 3U + 1U],
            expected_voxels[index * 3U + 2U]};
    const bool exact_order = actual_voxels == reference_voxels;
    std::sort(actual_voxels.begin(), actual_voxels.end());
    std::sort(reference_voxels.begin(), reference_voxels.end());
    const bool exact_set = actual_voxels == reference_voxels;
    std::size_t seed_matches = 0U;
    for (std::size_t joint = 0; joint < joint_count; ++joint)
        seed_matches += post_trace.surface_seeds[joint] == expected_seeds[joint] ? 1U : 0U;
    std::cout << "voxels actual=" << actual_voxels.size() << " expected=" << voxel_count
              << " exact_set=" << exact_set << " exact_order=" << exact_order
              << " seed_indices=" << seed_matches << '/' << joint_count << '\n';
    const auto post_dense_error = compare(post_trace.final_dense_weights, expected_post_dense);
    const auto post_binding_error = compare_binding(
        post, expected_post_joints, expected_post_weights, joint_count);
    std::cout << "postprocessed_dense max_abs=" << post_dense_error.maximum
              << " relative_l2=" << post_dense_error.relative_l2 << '\n';
    print_binding("postprocessed_binding", post_binding_error);

    const bool cpu = options.device == skintokens::device_kind::cpu;
    // Raw learned binding is the released demo's default and the acceptance
    // endpoint. voxel_skin remains visible above as an optional diagnostic;
    // its SciPy/Open3D tie behaviour is deliberately not allowed to hide a
    // decoder or raw final-weight regression.
    const bool passed = cpu ?
        (sampled_error.maximum <= 2.0e-4 && sampled_error.relative_l2 <= 3.0e-5 &&
         raw_dense_error.maximum <= 3.0e-4 && raw_dense_error.relative_l2 <= 1.0e-4 &&
         raw_binding_error.dense.relative_l2 <= 1.0e-3 && raw_binding_error.shared_slots >= 0.999) :
        (sampled_error.maximum <= 2.0e-2 && sampled_error.relative_l2 <= 2.0e-3 &&
         raw_dense_error.maximum <= 3.0e-2 && raw_dense_error.relative_l2 <= 3.0e-3 &&
         raw_binding_error.dense.relative_l2 <= 1.5e-2 && raw_binding_error.shared_slots >= 0.995);
    std::cout << "raw_default_result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << "giraffe parity: " << exception.what() << '\n';
    return 2;
}
