#include "binding.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    skintokens::skeleton rig;
    for (std::size_t joint = 0; joint < 6U; ++joint) {
        rig.names.push_back("joint_" + std::to_string(joint));
        rig.parents.push_back(joint == 0U ? -1 : static_cast<std::int32_t>(joint - 1U));
        // Deliberately place the strongest learned joints farthest away. A
        // geometry-ranked implementation therefore cannot pass this test.
        rig.rest_positions.push_back({static_cast<float>(joint), 0.0F, 0.0F});
    }
    const std::array<skintokens::vec3, 8> sampled{{
        {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, -1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, -1.0F}, {1.0F, 0.0F, 0.0F},
        {-1.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F},
    }};
    const std::array<skintokens::vec3, 1> vertices{{{0.1F, 0.1F, 0.1F}}};
    std::vector<std::vector<float>> dense(6U, std::vector<float>(sampled.size()));
    for (std::size_t joint = 0; joint < dense.size(); ++joint)
        std::fill(dense[joint].begin(), dense[joint].end(), 0.1F * static_cast<float>(joint + 1U));

    const auto binding = skintokens::detail::integrate_learned_binding(
        rig, vertices, sampled, dense);
    assert(binding.learned);
    assert(binding.joints.size() == 1U && binding.weights.size() == 1U);
    const std::array<std::uint16_t, 4> expected_joints{5U, 4U, 3U, 2U};
    assert(binding.joints[0] == expected_joints);
    const std::array<float, 4> expected_weights{6.0F/18.0F, 5.0F/18.0F,
                                                4.0F/18.0F, 3.0F/18.0F};
    for (std::size_t slot = 0; slot < 4U; ++slot)
        assert(std::abs(binding.weights[0][slot] - expected_weights[slot]) < 1.0e-6F);

    // The end-to-end upstream demo postprocess must keep the learned values
    // but prevent a remote high sigmoid channel from binding this vertex.
    // Build one connected strip and place joint zero at the queried end while
    // the raw learned ranking still favours distant joint five.
    const std::array<skintokens::vec3, 6> surface{{
        {0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F},
        {3.0F, 0.0F, 0.0F}, {4.0F, 0.0F, 0.0F}, {5.0F, 0.0F, 0.0F},
    }};
    const std::array<skintokens::triangle, 4> faces{{
        {{0U, 1U, 2U}}, {{1U, 2U, 3U}}, {{2U, 3U, 4U}}, {{3U, 4U, 5U}},
    }};
    std::array<skintokens::vec3, 6> normalized_joints{};
    for (std::size_t joint = 0; joint < normalized_joints.size(); ++joint)
        normalized_joints[joint] = surface[joint];
    std::array<std::vector<float>, 6> learned;
    for (std::size_t joint = 0; joint < learned.size(); ++joint)
        learned[joint].assign(sampled.size(), 0.1F * static_cast<float>(joint + 1U));
    const auto corrected = skintokens::detail::integrate_postprocessed_binding(
        rig, surface, faces,
        normalized_joints, sampled, learned);
    assert(corrected.joints[0][0] == 0U);
    assert(corrected.weights[0][0] > 0.999F);
}
