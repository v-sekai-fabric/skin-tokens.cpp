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
    auto fitted = skintokens::fit_motion_to_mesh(geometry, animation);
    assert(fitted);
    assert(fitted->rig.rest_positions[0].y == 0.0F);
    assert(fitted->rig.rest_positions[1].y == 1.0F);
    assert(fitted->root_translations.front().x == fitted->rig.rest_positions.front().x);
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
    assert(fitted_soma->rig.rest_positions[2].y < fitted_soma->rig.rest_positions[1].y);
    assert(fitted_soma->rig.rest_positions[3].y < fitted_soma->rig.rest_positions[2].y);
    assert(close(fitted_soma->rig.rest_positions[4].y, fitted_soma->rig.rest_positions[3].y));
    assert(fitted_soma->rig.rest_positions[4].x > fitted_soma->rig.rest_positions[3].x);
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
    assert(transformed->vertices[0].x == 2.0F && transformed->vertices[0].y == 3.0F && transformed->vertices[0].z == 4.0F);
    assert(transformed->vertices[1].x == 0.0F && transformed->vertices[2].y == 4.0F);
    assert(transformed->faces[0][0] == 0U && transformed->faces[0][1] == 2U && transformed->faces[0][2] == 1U);
    assert(transformed->colors[0].r == 1.0F && transformed->colors[1].g == 1.0F && transformed->colors[2].b == 1.0F);
    std::filesystem::remove(source);

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
