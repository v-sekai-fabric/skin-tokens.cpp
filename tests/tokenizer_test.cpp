#include "internal.hpp"

#include <cassert>

int main() {
    skintokens::mesh geometry;
    geometry.vertices = {{-1.0F, -1.0F, -1.0F}, {1.0F, 1.0F, 1.0F}};
    skintokens::skeleton rig;
    rig.names = {"root", "child_a", "child_b"};
    rig.parents = {-1, 0, 0};
    rig.rest_positions = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}, {0.5F, 0.0F, 0.0F}};
    auto prefix = skintokens::detail::tokenize_skeleton_prefix(geometry, rig);
    assert(prefix);
    assert(prefix->tokens.size() == 16U);
    assert(prefix->tokens[0] == 257); // BOS
    assert(prefix->tokens[1] == 266); // articulation
    assert(prefix->tokens[2] != 260); // no spurious spring marker
    assert(prefix->tokens[8] == 256); // branch before child_b
    assert(prefix->tokens.back() == 258); // skeleton EOS / skin switch
}
