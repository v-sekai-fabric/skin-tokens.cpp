#include <skintokens/skintokens.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

template<class T> void append(std::vector<std::byte> & output, const T & value) {
    const auto offset = output.size();
    output.resize(offset + sizeof(T));
    std::memcpy(output.data() + offset, &value, sizeof(T));
}

void write_import_fixture(const std::filesystem::path & path) {
    const float positions[] = {0,0,0, 1,0,0, 0,1,0};
    const std::uint16_t colors[] = {65535,0,0,65535, 0,65535,0,65535, 0,0,65535,65535};
    const std::uint16_t indices[] = {0,1,2};
    std::vector<std::byte> binary;
    for (float value : positions) append(binary, value);
    for (std::uint16_t value : colors) append(binary, value);
    for (std::uint16_t value : indices) append(binary, value);
    while (binary.size() % 4U) binary.push_back(std::byte{});
    std::string json = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"translation":[2,3,4],"scale":[-2,1,1]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"indices":2}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"normalized":true,"count":3,"type":"VEC4"},{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24},{"buffer":0,"byteOffset":60,"byteLength":6}],"buffers":[{"byteLength":68}]})";
    while (json.size() % 4U) json.push_back(' ');
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    const auto write_u32 = [&](std::uint32_t value) { stream.write(reinterpret_cast<const char *>(&value), 4); };
    write_u32(0x46546C67U); write_u32(2U);
    write_u32(static_cast<std::uint32_t>(12U + 8U + json.size() + 8U + binary.size()));
    write_u32(static_cast<std::uint32_t>(json.size())); write_u32(0x4E4F534AU);
    stream.write(json.data(), static_cast<std::streamsize>(json.size()));
    write_u32(static_cast<std::uint32_t>(binary.size())); write_u32(0x004E4942U);
    stream.write(reinterpret_cast<const char *>(binary.data()), static_cast<std::streamsize>(binary.size()));
}

void write_trellis_fixture(const std::filesystem::path & path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write("T2MESH03", 8);
    const std::uint32_t vertices = 3U, triangles = 1U;
    stream.write(reinterpret_cast<const char *>(&vertices), 4);
    stream.write(reinterpret_cast<const char *>(&triangles), 4);
    const float positions[] = {1,2,3, 4,5,6, 7,8,9};
    const float normals[] = {0,0,1, 0,0,1, 0,0,1};
    const float pbr[] = {1,0,0,.2F,.3F,1, 0,1,0,.4F,.5F,1, 0,0,1,.6F,.7F,1};
    const std::int32_t indices[] = {0,1,2};
    stream.write(reinterpret_cast<const char *>(positions), sizeof(positions));
    stream.write(reinterpret_cast<const char *>(normals), sizeof(normals));
    stream.write(reinterpret_cast<const char *>(pbr), sizeof(pbr));
    stream.write(reinterpret_cast<const char *>(indices), sizeof(indices));
}

void write_static_skeleton_fixture(const std::filesystem::path & path) {
    std::string json = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,2]}],"nodes":[{"name":"Hips","translation":[0,1,0],"children":[1]},{"name":"Spine1","translation":[0,0.4,0]},{"name":"MeshNode","mesh":0,"skin":0}],"meshes":[{"primitives":[]}],"skins":[{"skeleton":0,"joints":[1,0]}]})";
    while (json.size() % 4U) json.push_back(' ');
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    const auto write_u32 = [&](std::uint32_t value) { stream.write(reinterpret_cast<const char *>(&value), 4); };
    write_u32(0x46546C67U); write_u32(2U);
    write_u32(static_cast<std::uint32_t>(12U + 8U + json.size()));
    write_u32(static_cast<std::uint32_t>(json.size())); write_u32(0x4E4F534AU);
    stream.write(json.data(), static_cast<std::streamsize>(json.size()));
}

} // namespace

int main() {
    auto missing = skintokens::model::load("this-directory-does-not-exist");
    assert(!missing);

    skintokens::mesh invalid;
    auto glb = skintokens::load_glb_file("missing.glb");
    assert(!glb);

    skintokens::mesh geometry;
    geometry.vertices = {{-0.5F, 0.0F, 0.0F}, {0.5F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    geometry.normals = {{0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}};
    geometry.colors = {{1.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F, 1.0F},
                       {0.0F, 0.0F, 1.0F, 1.0F}};
    geometry.faces = {{{0U, 1U, 2U}}};
    skintokens::skin binding;
    binding.rig.names = {"root", "tip"};
    binding.rig.parents = {-1, 0};
    binding.rig.rest_positions = {{0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    binding.joints.assign(3U, {0U, 1U, 0U, 1U});
    binding.weights.assign(3U, {0.5F, 0.5F, 0.0F, 0.0F});
    skintokens::motion animation;
    animation.frames = 2U;
    animation.frames_per_second = 30.0F;
    animation.rig = binding.rig;
    animation.root_translations = {{0.0F, 0.0F, 0.0F}, {0.1F, 0.0F, 0.0F}};
    animation.local_rotations.assign(4U, {});

    // Exact CPU LBS evaluation: the tip rotates 90 degrees around Z while the
    // root translates one unit along X. These expected positions are derived
    // independently from the two joint transforms.
    skintokens::mesh lbs_geometry;
    lbs_geometry.vertices = {{0.0F, 0.0F, 0.0F}, {0.0F, 2.0F, 0.0F}, {0.0F, 2.0F, 0.0F}};
    skintokens::skin lbs_binding = binding;
    lbs_binding.joints = {{0U, 0U, 0U, 0U}, {1U, 0U, 0U, 0U}, {0U, 1U, 0U, 0U}};
    lbs_binding.weights = {{1.0F, 0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F, 0.0F},
                           {0.5F, 0.5F, 0.0F, 0.0F}};
    skintokens::motion lbs_animation = animation;
    lbs_animation.root_translations[1] = {1.0F, 0.0F, 0.0F};
    constexpr float half_sqrt_two = 0.7071067811865475F;
    lbs_animation.local_rotations[3] = {0.0F, 0.0F, half_sqrt_two, half_sqrt_two};
    auto rest_vertices = skintokens::deform_vertices(lbs_geometry, lbs_binding, lbs_animation, 0U);
    auto posed_vertices = skintokens::deform_vertices(lbs_geometry, lbs_binding, lbs_animation, 1U);
    assert(rest_vertices && posed_vertices);
    const auto close = [](float left, float right) { return std::abs(left - right) < 1.0e-5F; };
    assert(close((*rest_vertices)[2].x, 0.0F) && close((*rest_vertices)[2].y, 2.0F));
    assert(close((*posed_vertices)[0].x, 1.0F) && close((*posed_vertices)[0].y, 0.0F));
    assert(close((*posed_vertices)[1].x, 0.0F) && close((*posed_vertices)[1].y, 1.0F));
    assert(close((*posed_vertices)[2].x, 0.5F) && close((*posed_vertices)[2].y, 1.5F));
    assert(!skintokens::deform_vertices(lbs_geometry, lbs_binding, lbs_animation, 2U));
    // A fit may place and scale the rest skeleton, but glTF/Kimodo rotations
    // are already parent-local animation values. In particular, frame zero is
    // not a bind pose and must never be divided out of the complete track.
    animation.local_rotations[0] = {0.1F, -0.2F, 0.3F, 0.9F};
    animation.local_rotations[3] = {-0.3F, 0.1F, 0.2F, 0.9F};
    auto fitted = skintokens::fit_motion_to_mesh(geometry, animation);
    assert(fitted);
    assert(fitted->rig.rest_positions[0].y == 0.0F);
    assert(fitted->rig.rest_positions[1].y == 1.0F);
    assert(fitted->root_translations.front().x == fitted->rig.rest_positions.front().x);
    for (std::size_t index = 0; index < animation.local_rotations.size(); ++index) {
        assert(fitted->local_rotations[index].x == animation.local_rotations[index].x);
        assert(fitted->local_rotations[index].y == animation.local_rotations[index].y);
        assert(fitted->local_rotations[index].z == animation.local_rotations[index].z);
        assert(fitted->local_rotations[index].w == animation.local_rotations[index].w);
    }
    skintokens::motion soma_animation;
    soma_animation.frames = 1U;
    soma_animation.rig.names = {"Hips", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandMiddleEnd"};
    soma_animation.rig.parents = {-1, 0, 1, 2, 3};
    soma_animation.rig.rest_positions = {{0.0F, 0.0F, 0.0F}, {0.2F, 0.8F, 0.0F},
        {0.5F, 0.8F, 0.0F}, {0.8F, 0.8F, 0.0F}, {0.9F, 0.8F, 0.0F}};
    soma_animation.root_translations = {{0.0F, 0.0F, 0.0F}};
    soma_animation.local_rotations.assign(5U, {});
    auto fitted_soma = skintokens::fit_motion_to_mesh(geometry, soma_animation);
    assert(fitted_soma);
    assert(close(fitted_soma->rig.rest_positions[1].y, fitted_soma->rig.rest_positions[2].y));
    assert(close(fitted_soma->rig.rest_positions[2].y, fitted_soma->rig.rest_positions[3].y));
    const auto distance = [](skintokens::vec3 left, skintokens::vec3 right) {
        const float x = left.x - right.x, y = left.y - right.y, z = left.z - right.z;
        return std::sqrt(x*x + y*y + z*z);
    };
    const float global_upper = distance(fitted_soma->rig.rest_positions[1], fitted_soma->rig.rest_positions[2]);
    const float global_lower = distance(fitted_soma->rig.rest_positions[2], fitted_soma->rig.rest_positions[3]);
    const float global_hand = distance(fitted_soma->rig.rest_positions[3], fitted_soma->rig.rest_positions[4]);
    auto articulated_soma = skintokens::fit_motion_to_mesh(
        geometry, soma_animation, skintokens::skeleton_fit::articulated);
    assert(articulated_soma);
    assert(articulated_soma->rig.rest_positions[2].y < articulated_soma->rig.rest_positions[1].y);
    assert(articulated_soma->rig.rest_positions[3].y < articulated_soma->rig.rest_positions[2].y);
    assert(close(distance(articulated_soma->rig.rest_positions[1], articulated_soma->rig.rest_positions[2]),
                 global_upper));
    assert(close(distance(articulated_soma->rig.rest_positions[2], articulated_soma->rig.rest_positions[3]),
                 global_lower));
    assert(close(distance(articulated_soma->rig.rest_positions[3], articulated_soma->rig.rest_positions[4]),
                 global_hand));
    // Changing the articulated bind offsets must not double-apply that pose to
    // the animation. With identity source rotations, the rest-frame transfer
    // maps the articulated rig back onto the original global-fit trajectory.
    skintokens::mesh articulated_points;
    articulated_points.vertices = articulated_soma->rig.rest_positions;
    skintokens::skin articulated_binding;
    articulated_binding.rig = articulated_soma->rig;
    for (std::uint16_t joint_index = 0; joint_index < articulated_soma->rig.names.size(); ++joint_index) {
        articulated_binding.joints.push_back({joint_index, 0U, 0U, 0U});
        articulated_binding.weights.push_back({1.0F, 0.0F, 0.0F, 0.0F});
    }
    auto articulated_pose = skintokens::deform_vertices(
        articulated_points, articulated_binding, *articulated_soma, 0U);
    assert(articulated_pose && articulated_pose->size() == fitted_soma->rig.rest_positions.size());
    for (std::size_t joint_index = 0; joint_index < articulated_pose->size(); ++joint_index)
        assert(close(distance((*articulated_pose)[joint_index], fitted_soma->rig.rest_positions[joint_index]), 0.0F));
    // An arm chain that already lies in the mesh is kept rather than replaced
    // by a bounding-box guess. Covering the whole chain, shoulder included, is
    // what "already aligned" has to mean: the surface test samples along the
    // bones, so the fixture edge has to span them.
    skintokens::mesh aligned_arm_geometry = geometry;
    aligned_arm_geometry.normals.clear();
    aligned_arm_geometry.vertices.push_back(fitted_soma->rig.rest_positions[1]);
    aligned_arm_geometry.vertices.push_back(fitted_soma->rig.rest_positions[3]);
    aligned_arm_geometry.faces.push_back({0U, 3U, 4U});
    auto already_aligned = skintokens::fit_motion_to_mesh(
        aligned_arm_geometry, soma_animation, skintokens::skeleton_fit::articulated);
    assert(already_aligned);
    assert(close(already_aligned->rig.rest_positions[2].y, already_aligned->rig.rest_positions[1].y));
    assert(close(already_aligned->rig.rest_positions[3].y, already_aligned->rig.rest_positions[2].y));
    // Covering only the joints is not enough. A chord between two endpoints
    // that both sit on the surface can still leave the volume in between, so
    // sampling the bones rather than their endpoints must reject this one.
    skintokens::mesh endpoints_only_geometry = geometry;
    endpoints_only_geometry.normals.clear();
    endpoints_only_geometry.vertices.push_back(fitted_soma->rig.rest_positions[2]);
    endpoints_only_geometry.vertices.push_back(fitted_soma->rig.rest_positions[3]);
    endpoints_only_geometry.faces.push_back({0U, 3U, 4U});
    auto endpoints_only = skintokens::fit_motion_to_mesh(
        endpoints_only_geometry, soma_animation, skintokens::skeleton_fit::articulated);
    assert(endpoints_only);
    assert(endpoints_only->rig.rest_positions[2].y < endpoints_only->rig.rest_positions[1].y);
    // The clip's own first frame is a second length-exact pose of the same
    // skeleton. Where the supplied T-pose rest sits outside the mesh and that
    // first frame runs through it, articulated fitting adopts the first frame.
    // Frame zero then coincides with the bind pose, so the first exported frame
    // leaves the mesh undeformed instead of snapping it out of the T-pose.
    skintokens::mesh lowered_arm_geometry;
    lowered_arm_geometry.vertices = {{-0.1F, 0.0F, 0.0F}, {0.1F, 0.0F, 0.0F},
                                     {0.1F, 1.0F, 0.0F}, {-0.1F, 1.0F, 0.0F}};
    lowered_arm_geometry.faces = {{{0U, 1U, 2U}}, {{0U, 2U, 3U}}};
    skintokens::motion lowered_animation;
    lowered_animation.frames = 1U;
    lowered_animation.rig.names = {"Hips", "LeftArm", "LeftForeArm", "LeftHand"};
    lowered_animation.rig.parents = {-1, 0, 1, 2};
    lowered_animation.rig.rest_positions = {{0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F},
                                            {0.5F, 1.0F, 0.0F}, {1.0F, 1.0F, 0.0F}};
    lowered_animation.root_translations = {{0.0F, 0.0F, 0.0F}};
    lowered_animation.local_rotations.assign(4U, {});
    lowered_animation.local_rotations[1] = {0.0F, 0.0F, -half_sqrt_two, half_sqrt_two};
    auto kept_t_pose = skintokens::fit_motion_to_mesh(lowered_arm_geometry, lowered_animation);
    assert(kept_t_pose);
    assert(close(kept_t_pose->rig.rest_positions[1].y, kept_t_pose->rig.rest_positions[3].y));
    auto adopted = skintokens::fit_motion_to_mesh(
        lowered_arm_geometry, lowered_animation, skintokens::skeleton_fit::articulated);
    assert(adopted);
    assert(adopted->rig.rest_positions[2].y < adopted->rig.rest_positions[1].y);
    assert(adopted->rig.rest_positions[3].y < adopted->rig.rest_positions[2].y);
    assert(close(distance(adopted->rig.rest_positions[1], adopted->rig.rest_positions[2]), 0.5F));
    assert(close(distance(adopted->rig.rest_positions[2], adopted->rig.rest_positions[3]), 0.5F));
    skintokens::mesh adopted_points;
    adopted_points.vertices = adopted->rig.rest_positions;
    skintokens::skin adopted_binding;
    adopted_binding.rig = adopted->rig;
    for (std::uint16_t joint_index = 0; joint_index < adopted->rig.names.size(); ++joint_index) {
        adopted_binding.joints.push_back({joint_index, 0U, 0U, 0U});
        adopted_binding.weights.push_back({1.0F, 0.0F, 0.0F, 0.0F});
    }
    auto adopted_frame_zero = skintokens::deform_vertices(
        adopted_points, adopted_binding, *adopted, 0U);
    assert(adopted_frame_zero && adopted_frame_zero->size() == adopted->rig.rest_positions.size());
    for (std::size_t joint_index = 0; joint_index < adopted_frame_zero->size(); ++joint_index)
        assert(close(distance((*adopted_frame_zero)[joint_index],
                              adopted->rig.rest_positions[joint_index]), 0.0F));
    skintokens::skeleton generated_rig;
    generated_rig.names = {"bone_0", "bone_1", "bone_2"};
    generated_rig.parents = {-1, 0, 1};
    generated_rig.rest_positions = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.5F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    auto retargeted = skintokens::retarget_motion_to_rig(animation, generated_rig);
    assert(retargeted && retargeted->rig.names.size() == 3U && retargeted->frames == animation.frames);
    assert(retargeted->local_rotations.size() == animation.frames * 3U);
    assert(retargeted->root_translations.front().x == generated_rig.rest_positions.front().x);

    // The production A/B lane uses a semantic SOMA30 -> Mixamo52 map rather
    // than the generic proximity fallback. Exercise the exact published
    // Mixamo joint order and deterministic isolated-joint feedback loop.
    skintokens::motion soma30;
    soma30.frames = 2U;
    soma30.frames_per_second = 30.0F;
    soma30.rig.names = {"Hips","Spine1","Spine2","Chest","Neck1","Neck2","Head","Jaw","LeftEye","RightEye",
        "LeftShoulder","LeftArm","LeftForeArm","LeftHand","LeftHandThumbEnd","LeftHandMiddleEnd",
        "RightShoulder","RightArm","RightForeArm","RightHand","RightHandThumbEnd","RightHandMiddleEnd",
        "LeftLeg","LeftShin","LeftFoot","LeftToeBase","RightLeg","RightShin","RightFoot","RightToeBase"};
    soma30.rig.parents = {-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28};
    soma30.rig.rest_positions = {{0,1,0},{0,1.15F,0},{0,1.3F,0},{0,1.45F,0},{0,1.58F,0},{0,1.66F,0},{0,1.78F,0},
        {0,1.72F,.04F},{-.03F,1.8F,.08F},{.03F,1.8F,.08F},{-.12F,1.46F,0},{-.35F,1.44F,0},{-.58F,1.35F,0},
        {-.74F,1.25F,0},{-.84F,1.2F,.05F},{-.9F,1.2F,0},{.12F,1.46F,0},{.35F,1.44F,0},{.58F,1.35F,0},
        {.74F,1.25F,0},{.84F,1.2F,.05F},{.9F,1.2F,0},{-.12F,.92F,0},{-.13F,.5F,0},{-.13F,.12F,.02F},
        {-.13F,.03F,.18F},{.12F,.92F,0},{.13F,.5F,0},{.13F,.12F,.02F},{.13F,.03F,.18F}};
    soma30.root_translations = {soma30.rig.rest_positions.front(), soma30.rig.rest_positions.front()};
    soma30.local_rotations.assign(soma30.frames * soma30.rig.names.size(), {});
    soma30.local_rotations[soma30.rig.names.size() + 11U] = {0,0,half_sqrt_two,half_sqrt_two};
    auto mixamo52 = skintokens::make_mixamo52_rig(soma30.rig);
    assert(mixamo52 && mixamo52->names.size() == 52U);
    assert(mixamo52->names.front() == "mixamorig:Hips");
    assert(mixamo52->names[10] == "mixamorig:LeftHandThumb1");
    assert(mixamo52->names[25] == "mixamorig:RightShoulder");
    assert(mixamo52->names[29] == "mixamorig:RightHandIndex1");
    auto semantic = skintokens::retarget_soma30_to_mixamo52(soma30, *mixamo52);
    assert(semantic && semantic->rig.names.size() == 52U);
    assert(semantic->local_rotations[52U + 7U].z == half_sqrt_two);
    auto validation = skintokens::validate_soma30_to_mixamo52(soma30, *mixamo52);
    assert(validation && validation->isolated_cases >= 132U);
    assert(std::isfinite(validation->isolated_mean_position_error));
    assert(std::isfinite(validation->motion_mean_position_error));
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto output = std::filesystem::temp_directory_path() /
        ("skintokens-api-" + std::to_string(suffix) + ".glb");
    auto saved = skintokens::save_skinned_animation_glb_file(output, geometry, binding, animation);
    assert(saved);
    auto loaded = skintokens::load_glb_file(output);
    assert(loaded && loaded->vertices.size() == 3U && loaded->faces.size() == 1U && loaded->colors.size() == 3U);
    auto loaded_motion = skintokens::load_kimodo_glb_file(output);
    assert(loaded_motion && loaded_motion->rig.names.size() == binding.rig.names.size());
    assert(loaded_motion->rig.parents == binding.rig.parents);
    std::filesystem::remove(output);

    const auto source = std::filesystem::temp_directory_path() /
        ("skintokens-import-" + std::to_string(suffix) + ".glb");
    write_import_fixture(source);
    auto transformed = skintokens::load_glb_file(source);
    assert(transformed && transformed->vertices.size() == 3U && transformed->colors.size() == 3U);
    auto mesh_only_info = skintokens::inspect_glb_file(source);
    assert(mesh_only_info && mesh_only_info->has_mesh && !mesh_only_info->has_skeleton);
    assert(transformed->vertices[0].x == 2.0F && transformed->vertices[0].y == 3.0F && transformed->vertices[0].z == 4.0F);
    assert(transformed->vertices[1].x == 0.0F && transformed->vertices[2].y == 4.0F);
    assert(transformed->faces[0][0] == 0U && transformed->faces[0][1] == 2U && transformed->faces[0][2] == 1U);
    assert(transformed->colors[0].r == 1.0F && transformed->colors[1].g == 1.0F && transformed->colors[2].b == 1.0F);
    std::filesystem::remove(source);

    const auto static_skeleton_path = std::filesystem::temp_directory_path() /
        ("skintokens-static-skeleton-" + std::to_string(suffix) + ".glb");
    write_static_skeleton_fixture(static_skeleton_path);
    auto static_skeleton = skintokens::load_skeleton_glb_file(static_skeleton_path);
    assert(static_skeleton && static_skeleton->frames == 1U);
    assert(static_skeleton->rig.names.size() == 2U);
    assert(static_skeleton->rig.names[0] == "Hips" && static_skeleton->rig.parents[1] == 0);
    assert(close(static_skeleton->rig.rest_positions[1].y, 1.4F));
    assert(!skintokens::load_kimodo_glb_file(static_skeleton_path));
    auto static_info = skintokens::inspect_glb_file(static_skeleton_path);
    assert(static_info && static_info->has_mesh && static_info->has_skin && static_info->has_skeleton);
    assert(!static_info->has_animation && static_info->joint_count == 2U);
    std::filesystem::remove(static_skeleton_path);

    const auto trellis_path = std::filesystem::temp_directory_path() /
        ("skintokens-import-" + std::to_string(suffix) + ".t2mesh");
    write_trellis_fixture(trellis_path);
    auto trellis = skintokens::load_trellis_mesh_file(trellis_path);
    assert(trellis && trellis->vertices.size() == 3U && trellis->colors.size() == 3U);
    assert(trellis->vertices[0].x == 1.0F && trellis->vertices[0].y == 3.0F && trellis->vertices[0].z == -2.0F);
    assert(trellis->normals[0].y == 1.0F && trellis->normals[0].z == 0.0F);
    assert(trellis->colors[0].r == 1.0F && trellis->colors[1].g == 1.0F && trellis->colors[2].b == 1.0F);
    assert(trellis->metallic_factor > 0.39F && trellis->metallic_factor < 0.41F);
    std::filesystem::remove(trellis_path);
    return 0;
}
