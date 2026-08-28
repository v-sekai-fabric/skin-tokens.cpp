#include <skintokens/skintokens.hpp>

#include <iostream>
#include <filesystem>
#include <iomanip>
#include <string_view>

namespace {

void usage() {
    std::cerr << "usage:\n"
              << "  skintokens-cli inspect MODEL_DIR [--device auto|cpu|vulkan]\n"
              << "  skintokens-cli rig MODEL_DIR MESH.{glb,t2mesh} OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--postprocess]"
                 " [--beams N] [--temperature F] [--max-tokens N]\n"
              << "  skintokens-cli retarget-check SOMA30.glb [MIXAMO52.glb]\n"
              << "  skintokens-cli prepare-mixamo MESH.glb SOMA30.glb OUTPUT.glb\n"
              << "  skintokens-cli bind MODEL_DIR MESH.{glb,t2mesh} MOTION.glb OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--geometric|--supplied-skeleton] [--no-fit]"
                 " [--target-rig soma30|mixamo52]"
                 " [--postprocess]"
                 " [--beams N] [--temperature F] [--max-tokens N]\n";
}

skintokens::device_kind parse_device(std::string_view value) {
    if (value == "cpu") return skintokens::device_kind::cpu;
    if (value == "vulkan") return skintokens::device_kind::vulkan;
    return skintokens::device_kind::automatic;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    if (std::string_view{argv[1]} == "retarget-check") {
        auto animation = skintokens::load_kimodo_glb_file(argv[2]);
        if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
        auto mixamo = skintokens::make_mixamo52_rig(animation->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto report = skintokens::validate_soma30_to_mixamo52(*animation, *mixamo);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        std::cout << std::fixed << std::setprecision(7)
                  << "isolated cases: " << report->isolated_cases << '\n'
                  << "isolated mean normalized position error: " << report->isolated_mean_position_error << '\n'
                  << "isolated max normalized position error: " << report->isolated_max_position_error << '\n'
                  << "clip mean normalized position error: " << report->motion_mean_position_error << '\n'
                  << "clip max normalized position error: " << report->motion_max_position_error << '\n'
                  << "foot velocity relative error: " << report->foot_velocity_relative_error << '\n';
        if (argc >= 4) {
            auto exported = skintokens::load_kimodo_glb_file(argv[3]);
            if (!exported) { std::cerr << exported.error().message << '\n'; return 1; }
            auto roundtrip = skintokens::compare_soma30_to_mixamo52(*animation, *exported);
            if (!roundtrip) { std::cerr << roundtrip.error().message << '\n'; return 1; }
            std::cout << "exported clip mean normalized position error: "
                      << roundtrip->motion_mean_position_error << '\n'
                      << "exported clip max normalized position error: "
                      << roundtrip->motion_max_position_error << '\n'
                      << "exported foot velocity relative error: "
                      << roundtrip->foot_velocity_relative_error << '\n';
        }
        return 0;
    }
    if (std::string_view{argv[1]} == "prepare-mixamo") {
        if (argc != 5) { usage(); return 2; }
        auto geometry = skintokens::load_glb_file(argv[2]);
        auto source = skintokens::load_kimodo_glb_file(argv[3]);
        if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
        if (!source) { std::cerr << source.error().message << '\n'; return 1; }
        auto fitted = skintokens::fit_motion_to_mesh(*geometry, *source);
        if (!fitted) { std::cerr << fitted.error().message << '\n'; return 1; }
        auto mixamo = skintokens::make_mixamo52_rig(fitted->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto animation = skintokens::retarget_soma30_to_mixamo52(*fitted, *mixamo);
        if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
        if (geometry->vertices.size() < mixamo->names.size()) {
            std::cerr << "mesh has too few vertices for the covered-skeleton fixture\n";
            return 1;
        }
        skintokens::skin covered;
        covered.rig = *mixamo;
        covered.joints.assign(geometry->vertices.size(), {0U, 0U, 0U, 0U});
        covered.weights.assign(geometry->vertices.size(), {1.0F, 0.0F, 0.0F, 0.0F});
        // Upstream's predict transform trims unweighted leaves before
        // tokenization. Give every supplied joint a harmless 0.1% marker so
        // the comparison exercises the requested 52-joint topology.
        for (std::size_t joint = 1U; joint < mixamo->names.size(); ++joint) {
            covered.joints[joint] = {0U, static_cast<std::uint16_t>(joint), 0U, 0U};
            covered.weights[joint] = {0.999F, 0.001F, 0.0F, 0.0F};
        }
        auto saved = skintokens::save_skinned_animation_glb_file(
            argv[4], *geometry, covered, *animation);
        if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
        std::cout << "covered Mixamo52 comparison fixture written to " << argv[4] << '\n';
        return 0;
    }
    skintokens::runtime_options runtime;
    skintokens::generation_options generation;
    bool fit = true;
    bool supplied_skeleton = false;
    std::string_view target_rig = "soma30";
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::string_view{argv[i]} == "--device") runtime.device = parse_device(argv[++i]);
        else if (std::string_view{argv[i]} == "--geometric") generation.geometric_only = true;
        else if (std::string_view{argv[i]} == "--supplied-skeleton") supplied_skeleton = true;
        else if (std::string_view{argv[i]} == "--target-rig") {
            target_rig = argv[++i];
            if (target_rig != "soma30" && target_rig != "mixamo52") {
                std::cerr << "target rig must be soma30 or mixamo52\n";
                return 2;
            }
            supplied_skeleton = true;
        }
        else if (std::string_view{argv[i]} == "--no-fit") fit = false;
        else if (std::string_view{argv[i]} == "--raw-learned") generation.surface_postprocess = false;
        else if (std::string_view{argv[i]} == "--postprocess") generation.surface_postprocess = true;
        else if (std::string_view{argv[i]} == "--beams") generation.beams = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (std::string_view{argv[i]} == "--temperature") generation.temperature = std::stof(argv[++i]);
        else if (std::string_view{argv[i]} == "--max-tokens") generation.max_tokens = std::stoul(argv[++i]);
    }
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--geometric") generation.geometric_only = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--no-fit") fit = false;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--supplied-skeleton") supplied_skeleton = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--raw-learned") generation.surface_postprocess = false;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--postprocess") generation.surface_postprocess = true;
    auto model = skintokens::model::load(argv[2], runtime);
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    if (std::string_view{argv[1]} == "inspect") {
        std::cout << "SkinTokens 0.1 model bundle\nbackend: " << model->backend_name() << '\n';
        return 0;
    }
    if (std::string_view{argv[1]} == "rig") {
        if (argc < 5) { usage(); return 2; }
        const std::filesystem::path mesh_path = argv[3];
        auto geometry = mesh_path.extension() == ".t2mesh" ?
            skintokens::load_trellis_mesh_file(mesh_path) : skintokens::load_glb_file(mesh_path);
        if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
        auto binding = model->rig(*geometry, generation);
        if (!binding) { std::cerr << binding.error().message << '\n'; return 1; }

        // A rigged static GLB still needs one identity pose so ordinary glTF
        // viewers can evaluate the skin immediately. It can later receive
        // animation from any compatible retargeting workflow.
        skintokens::motion rest;
        rest.frames = 1U;
        rest.frames_per_second = 30.0F;
        rest.rig = binding->rig;
        rest.root_translations.assign(1U, {});
        rest.local_rotations.assign(rest.rig.names.size(), {});
        auto saved = skintokens::save_skinned_animation_glb_file(
            argv[4], *geometry, *binding, rest);
        if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
        std::cout << "learned skeleton and skin weights written to " << argv[4] << '\n';
        return 0;
    }
    if (std::string_view{argv[1]} != "bind" || argc < 6) {
        usage();
        return 2;
    }
    const std::filesystem::path mesh_path = argv[3];
    auto geometry = mesh_path.extension() == ".t2mesh" ?
        skintokens::load_trellis_mesh_file(mesh_path) : skintokens::load_glb_file(mesh_path);
    auto animation = skintokens::load_kimodo_glb_file(argv[4]);
    if (!geometry) { std::cerr << geometry.error().message << '\n'; return 1; }
    if (!animation) { std::cerr << animation.error().message << '\n'; return 1; }
    if (generation.geometric_only || supplied_skeleton) {
      if (fit) {
        auto fitted = skintokens::fit_motion_to_mesh(*geometry, *animation);
        if (!fitted) { std::cerr << fitted.error().message << '\n'; return 1; }
        animation = std::move(fitted);
      }
    }
    if (target_rig == "mixamo52") {
        auto mixamo = skintokens::make_mixamo52_rig(animation->rig);
        if (!mixamo) { std::cerr << mixamo.error().message << '\n'; return 1; }
        auto report = skintokens::validate_soma30_to_mixamo52(*animation, *mixamo);
        if (!report) { std::cerr << report.error().message << '\n'; return 1; }
        std::cerr << "SOMA30 -> Mixamo52: " << report->isolated_cases
                  << " isolated probes, mean/max " << report->isolated_mean_position_error
                  << '/' << report->isolated_max_position_error
                  << "; clip mean/max " << report->motion_mean_position_error
                  << '/' << report->motion_max_position_error
                  << "; foot velocity relative error " << report->foot_velocity_relative_error << '\n';
        auto retargeted = skintokens::retarget_soma30_to_mixamo52(*animation, *mixamo);
        if (!retargeted) { std::cerr << retargeted.error().message << '\n'; return 1; }
        animation = std::move(retargeted);
    }
    skintokens::result<skintokens::skin> binding = (generation.geometric_only || supplied_skeleton) ?
        model->bind(*geometry, animation->rig, generation) : model->rig(*geometry, generation);
    if (!binding) { std::cerr << binding.error().message << '\n'; return 1; }
    if (!generation.geometric_only && !supplied_skeleton) {
        auto retargeted = skintokens::retarget_motion_to_rig(*animation, binding->rig);
        if (!retargeted) { std::cerr << retargeted.error().message << '\n'; return 1; }
        animation = std::move(retargeted);
    }
    auto saved = skintokens::save_skinned_animation_glb_file(argv[5], *geometry, *binding, *animation);
    if (!saved) { std::cerr << saved.error().message << '\n'; return 1; }
    std::cout << (binding->learned ? "learned" : "geometric") << " binding written to " << argv[5] << '\n';
    return 0;
}
