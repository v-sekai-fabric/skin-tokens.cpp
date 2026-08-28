#include "binding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <queue>

namespace skintokens::detail {
namespace {

class point_tree {
    struct node {
        std::uint32_t point;
        std::int32_t left = -1;
        std::int32_t right = -1;
        std::uint8_t axis;
    };
    using candidate = std::pair<float, std::uint32_t>;

    std::span<const vec3> points_;
    std::vector<std::uint32_t> order_;
    std::vector<node> nodes_;

    static float coordinate(vec3 value, std::uint8_t axis) {
        return axis == 0U ? value.x : axis == 1U ? value.y : value.z;
    }

    std::int32_t build(std::size_t begin, std::size_t end, std::uint8_t axis) {
        if (begin == end) return -1;
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
            order_.begin() + static_cast<std::ptrdiff_t>(middle),
            order_.begin() + static_cast<std::ptrdiff_t>(end), [&](auto left, auto right) {
                return coordinate(points_[left], axis) < coordinate(points_[right], axis);
            });
        const auto result = static_cast<std::int32_t>(nodes_.size());
        nodes_.push_back({order_[middle], -1, -1, axis});
        const auto next = static_cast<std::uint8_t>((axis + 1U) % 3U);
        nodes_[static_cast<std::size_t>(result)].left = build(begin, middle, next);
        nodes_[static_cast<std::size_t>(result)].right = build(middle + 1U, end, next);
        return result;
    }

    void search(std::int32_t id, vec3 query, std::priority_queue<candidate> & best) const {
        if (id < 0) return;
        const auto & current = nodes_[static_cast<std::size_t>(id)];
        const auto point = points_[current.point];
        const float dx = query.x-point.x, dy = query.y-point.y, dz = query.z-point.z;
        const float distance = dx*dx + dy*dy + dz*dz;
        if (best.size() < 8U) best.emplace(distance, current.point);
        else if (distance < best.top().first) {
            best.pop();
            best.emplace(distance, current.point);
        }
        const float delta = coordinate(query, current.axis) - coordinate(point, current.axis);
        const auto near = delta < 0.0F ? current.left : current.right;
        const auto far = delta < 0.0F ? current.right : current.left;
        search(near, query, best);
        if (best.size() < 8U || delta * delta < best.top().first) search(far, query, best);
    }

public:
    explicit point_tree(std::span<const vec3> points) : points_(points), order_(points.size()) {
        std::iota(order_.begin(), order_.end(), 0U);
        nodes_.reserve(points.size());
        build(0U, points.size(), 0U);
    }

    std::array<candidate, 8> nearest(vec3 query) const {
        std::priority_queue<candidate> best;
        search(nodes_.empty() ? -1 : 0, query, best);
        std::array<candidate, 8> output{};
        for (std::size_t index = best.size(); index > 0U; --index) {
            output[index - 1U] = best.top();
            best.pop();
        }
        return output;
    }
};

} // namespace

skin integrate_learned_binding(
    const skeleton & target,
    std::span<const vec3> normalized_vertices,
    std::span<const vec3> sampled_points,
    std::span<const std::vector<float>> dense_joint_weights,
    binding_trace * trace) {
    skin output;
    output.rig = target;
    output.learned = true;
    output.joints.resize(normalized_vertices.size());
    output.weights.resize(normalized_vertices.size());
    if (trace != nullptr) {
        trace->neighbor_indices.resize(normalized_vertices.size() * 8U);
        trace->interpolation_weights.resize(normalized_vertices.size() * 8U);
    }

    const point_tree tree{sampled_points};
    struct binding_candidate { float learned_weight; std::uint16_t joint; };
    for (std::size_t vertex = 0; vertex < normalized_vertices.size(); ++vertex) {
        const auto neighbors = tree.nearest(normalized_vertices[vertex]);
        std::array<float, 8> interpolation{};
        float interpolation_sum = 0.0F;
        for (std::size_t index = 0; index < neighbors.size(); ++index) {
            interpolation[index] = 1.0F / (std::sqrt(neighbors[index].first) + 1e-8F);
            interpolation_sum += interpolation[index];
            if (trace != nullptr) {
                trace->neighbor_indices[vertex * 8U + index] = neighbors[index].second;
                trace->interpolation_weights[vertex * 8U + index] = interpolation[index];
            }
        }

        std::vector<binding_candidate> ranked;
        ranked.reserve(dense_joint_weights.size());
        for (std::size_t joint = 0; joint < dense_joint_weights.size(); ++joint) {
            float value = 0.0F;
            for (std::size_t index = 0; index < neighbors.size(); ++index)
                value += interpolation[index] * dense_joint_weights[joint][neighbors[index].second];
            ranked.push_back({value / interpolation_sum, static_cast<std::uint16_t>(joint)});
        }
        const auto keep = std::min<std::size_t>(4U, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(keep), ranked.end(),
            [](const auto & left, const auto & right) {
                if (left.learned_weight != right.learned_weight)
                    return left.learned_weight > right.learned_weight;
                return left.joint < right.joint;
            });
        float sum = 0.0F;
        for (std::size_t slot = 0; slot < 4U; ++slot) {
            const auto value = slot < keep ? ranked[slot] : binding_candidate{0.0F, 0U};
            output.weights[vertex][slot] = value.learned_weight;
            output.joints[vertex][slot] = value.joint;
            sum += value.learned_weight;
        }
        if (sum <= 1e-12F) {
            output.weights[vertex][0] = 1.0F;
            sum = 1.0F;
        }
        for (auto & value : output.weights[vertex]) value /= sum;
    }
    return output;
}

} // namespace skintokens::detail
