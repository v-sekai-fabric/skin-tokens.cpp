#pragma once

#include <skintokens/skintokens.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace skintokens::detail {

struct binding_trace {
    std::vector<std::uint32_t> neighbor_indices;
    std::vector<float> interpolation_weights;
};

// Reproduce upstream Asset.from_data() followed by its Blender export: each
// source vertex receives inverse-distance interpolation from eight sampled
// points, then the greatest four learned joint weights are retained and
// normalized. This intentionally contains no geometric bone-distance gate.
skin integrate_learned_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace = nullptr);

} // namespace skintokens::detail
