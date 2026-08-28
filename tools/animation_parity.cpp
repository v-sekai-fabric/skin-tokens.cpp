#include <skintokens/skintokens.hpp>

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
#include <string>
#include <vector>

namespace {

template<class T>
std::vector<T> read_array(const std::filesystem::path & path, std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::runtime_error("fixture array size overflows");
    const auto bytes = count * sizeof(T);
    std::error_code error;
    const auto actual = std::filesystem::file_size(path, error);
    if (error || actual != bytes)
        throw std::runtime_error("unexpected fixture size: " + path.string());
    std::vector<T> output(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(output.data()), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("cannot read fixture: " + path.string());
    return output;
}

std::size_t product(std::initializer_list<std::size_t> values) {
    std::size_t output = 1U;
    for (const auto value : values) {
        if (value != 0U && output > std::numeric_limits<std::size_t>::max() / value)
            throw std::runtime_error("fixture shape overflows");
        output *= value;
    }
    return output;
}

template<class T>
std::vector<T> required_array(const std::filesystem::path & directory, std::string_view name,
                              std::size_t count) {
    return read_array<T>(directory / name, count);
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 2) {
        std::cerr << "usage: skintokens-animation-parity FIXTURE_DIR\n";
        return 2;
    }
    const std::filesystem::path directory = argv[1];
    std::ifstream manifest_stream(directory / "animation-vertices.json");
    if (!manifest_stream) throw std::runtime_error("cannot open animation-vertices.json");
    nlohmann::json manifest;
    manifest_stream >> manifest;
    if (manifest.value("format", 0U) != 1U) throw std::runtime_error("unsupported fixture format");
    const auto vertex_count = manifest.at("vertex_count").get<std::size_t>();
    const auto joint_count = manifest.at("joint_count").get<std::size_t>();
    const auto frame_count = manifest.at("frame_count").get<std::size_t>();
    const auto selected = manifest.at("selected_frames").get<std::vector<std::size_t>>();
    if (vertex_count == 0U || vertex_count > 20'000'000U || joint_count == 0U || joint_count > 256U ||
        frame_count == 0U || frame_count > 100'000U || selected.empty() || selected.size() > 256U)
        throw std::runtime_error("fixture dimensions exceed safety limits");
    for (const auto frame : selected)
        if (frame >= frame_count) throw std::runtime_error("selected fixture frame is out of range");

    const auto positions = required_array<float>(directory, "positions.f32", product({vertex_count, 3U}));
    const auto joints = required_array<std::uint16_t>(directory, "joints.u16", product({vertex_count, 4U}));
    const auto weights = required_array<float>(directory, "weights.f32", product({vertex_count, 4U}));
    const auto parents = required_array<std::int32_t>(directory, "parents.i32", joint_count);
    const auto rest = required_array<float>(directory, "rest-positions.f32", product({joint_count, 3U}));
    const auto roots = required_array<float>(directory, "root-translations.f32", product({frame_count, 3U}));
    const auto rotations = required_array<float>(directory, "local-rotations.f32",
                                                  product({frame_count, joint_count, 4U}));
    const auto expected = required_array<float>(directory, "expected-vertices.f32",
                                                 product({selected.size(), vertex_count, 3U}));

    bool passed = true;
    if (std::filesystem::exists(directory / "reference-joints.u16") ||
        std::filesystem::exists(directory / "reference-weights.f32")) {
        const auto reference_joints = required_array<std::uint16_t>(
            directory, "reference-joints.u16", product({vertex_count, 4U}));
        const auto reference_weights = required_array<float>(
            directory, "reference-weights.f32", product({vertex_count, 4U}));
        double numerator = 0.0;
        double denominator = 0.0;
        double maximum = 0.0;
        std::size_t worst_vertex = 0U;
        std::size_t exact_vertices = 0U;
        std::size_t shared_slots = 0U;
        std::vector<double> actual_dense(joint_count);
        std::vector<double> reference_dense(joint_count);
        for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex) {
            std::fill(actual_dense.begin(), actual_dense.end(), 0.0);
            std::fill(reference_dense.begin(), reference_dense.end(), 0.0);
            for (std::size_t slot = 0U; slot < 4U; ++slot) {
                const auto actual_joint = joints[vertex * 4U + slot];
                const auto reference_joint = reference_joints[vertex * 4U + slot];
                if (actual_joint >= joint_count || reference_joint >= joint_count)
                    throw std::runtime_error("reference binding contains an invalid joint");
                actual_dense[actual_joint] += weights[vertex * 4U + slot];
                reference_dense[reference_joint] += reference_weights[vertex * 4U + slot];
            }
            for (std::size_t joint = 0U; joint < joint_count; ++joint) {
                const bool actual_selected = actual_dense[joint] > 0.0;
                const bool reference_selected = reference_dense[joint] > 0.0;
                shared_slots += actual_selected && reference_selected ? 1U : 0U;
                const double difference = actual_dense[joint] - reference_dense[joint];
                if (std::abs(difference) > maximum) {
                    maximum = std::abs(difference);
                    worst_vertex = vertex;
                }
                numerator += difference * difference;
                denominator += reference_dense[joint] * reference_dense[joint];
            }
            bool exact = true;
            for (std::size_t joint = 0U; joint < joint_count; ++joint)
                exact = exact && ((actual_dense[joint] > 0.0) == (reference_dense[joint] > 0.0));
            exact_vertices += exact ? 1U : 0U;
        }
        const double relative = std::sqrt(numerator / std::max(denominator, 1.0e-30));
        const double exact_fraction = static_cast<double>(exact_vertices) / vertex_count;
        const double slot_fraction = static_cast<double>(shared_slots) / (vertex_count * 4U);
        const double maximum_limit = manifest.value("binding_max_abs_tolerance", 0.05);
        const double relative_limit = manifest.value("binding_relative_l2_tolerance", 0.03);
        const double exact_limit = manifest.value("binding_exact_top4_tolerance", 0.98);
        const double slot_limit = manifest.value("binding_shared_top4_tolerance", 0.995);
        const bool binding_passed = maximum <= maximum_limit && relative <= relative_limit &&
                                    exact_fraction >= exact_limit && slot_fraction >= slot_limit;
        std::cout << "binding max_abs=" << maximum << " relative_l2=" << relative
                  << " exact_top4_vertices=" << exact_fraction
                  << " shared_top4_slots=" << slot_fraction
                  << " worst_vertex=" << worst_vertex << " result="
                  << (binding_passed ? "PASS" : "FAIL") << '\n';
        passed = binding_passed;
    }

    skintokens::mesh geometry;
    geometry.vertices.resize(vertex_count);
    skintokens::skin binding;
    binding.rig.names.resize(joint_count);
    binding.rig.parents = parents;
    binding.rig.rest_positions.resize(joint_count);
    binding.joints.resize(vertex_count);
    binding.weights.resize(vertex_count);
    skintokens::motion animation;
    animation.frames = frame_count;
    animation.frames_per_second = manifest.value("frames_per_second", 30.0F);
    animation.rig = binding.rig;
    animation.root_translations.resize(frame_count);
    animation.local_rotations.resize(product({frame_count, joint_count}));
    for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex) {
        geometry.vertices[vertex] = {positions[vertex * 3U], positions[vertex * 3U + 1U], positions[vertex * 3U + 2U]};
        for (std::size_t slot = 0U; slot < 4U; ++slot) {
            binding.joints[vertex][slot] = joints[vertex * 4U + slot];
            binding.weights[vertex][slot] = weights[vertex * 4U + slot];
        }
    }
    for (std::size_t joint = 0U; joint < joint_count; ++joint) {
        binding.rig.names[joint] = "joint_" + std::to_string(joint);
        binding.rig.rest_positions[joint] = {rest[joint * 3U], rest[joint * 3U + 1U], rest[joint * 3U + 2U]};
    }
    animation.rig = binding.rig;
    for (std::size_t frame = 0U; frame < frame_count; ++frame) {
        animation.root_translations[frame] = {roots[frame * 3U], roots[frame * 3U + 1U], roots[frame * 3U + 2U]};
        for (std::size_t joint = 0U; joint < joint_count; ++joint) {
            const auto index = (frame * joint_count + joint) * 4U;
            animation.local_rotations[frame * joint_count + joint] = {
                rotations[index], rotations[index + 1U], rotations[index + 2U], rotations[index + 3U]};
        }
    }

    const double max_abs_limit = manifest.value("max_abs_tolerance", 2.0e-5);
    const double relative_l2_limit = manifest.value("relative_l2_tolerance", 2.0e-6);
    const double frame_relative_l2_limit = manifest.value(
        "frame_relative_l2_tolerance", relative_l2_limit);
    double global_max = 0.0;
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t selected_index = 0U; selected_index < selected.size(); ++selected_index) {
        const auto frame = selected[selected_index];
        auto actual = skintokens::deform_vertices(geometry, binding, animation, frame);
        if (!actual) throw std::runtime_error(actual.error().message);
        double frame_max = 0.0;
        double frame_numerator = 0.0;
        double frame_denominator = 0.0;
        std::size_t worst_vertex = 0U;
        for (std::size_t vertex = 0U; vertex < vertex_count; ++vertex) {
            const float values[3]{(*actual)[vertex].x, (*actual)[vertex].y, (*actual)[vertex].z};
            for (std::size_t axis = 0U; axis < 3U; ++axis) {
                const auto expected_index = (selected_index * vertex_count + vertex) * 3U + axis;
                const double difference = static_cast<double>(values[axis]) - expected[expected_index];
                const double absolute = std::abs(difference);
                if (absolute > frame_max) { frame_max = absolute; worst_vertex = vertex; }
                frame_numerator += difference * difference;
                frame_denominator += static_cast<double>(expected[expected_index]) * expected[expected_index];
            }
        }
        const double relative = std::sqrt(frame_numerator / std::max(frame_denominator, 1.0e-30));
        std::cout << "frame=" << frame << " max_abs=" << frame_max << " relative_l2=" << relative
                  << " worst_vertex=" << worst_vertex << '\n';
        global_max = std::max(global_max, frame_max);
        numerator += frame_numerator;
        denominator += frame_denominator;
        passed = passed && frame_max <= max_abs_limit && relative <= frame_relative_l2_limit;
    }
    const double global_relative = std::sqrt(numerator / std::max(denominator, 1.0e-30));
    passed = passed && global_max <= max_abs_limit && global_relative <= relative_l2_limit;
    std::cout << "overall max_abs=" << global_max << " relative_l2=" << global_relative
              << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
} catch (const std::exception & exception) {
    std::cerr << "animation parity: " << exception.what() << '\n';
    return 2;
}
