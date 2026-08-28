#include <skintokens/skintokens.hpp>

#include <iostream>
#include <filesystem>
#include <string_view>

namespace {

void usage() {
    std::cerr << "usage:\n"
              << "  skintokens-cli inspect MODEL_DIR [--device auto|cpu|vulkan]\n"
              << "  skintokens-cli bind MODEL_DIR MESH.{glb,t2mesh} MOTION.glb OUTPUT.glb"
                 " [--device auto|cpu|vulkan] [--geometric|--supplied-skeleton] [--no-fit]"
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
    skintokens::runtime_options runtime;
    skintokens::generation_options generation;
    bool fit = true;
    bool supplied_skeleton = false;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::string_view{argv[i]} == "--device") runtime.device = parse_device(argv[++i]);
        else if (std::string_view{argv[i]} == "--geometric") generation.geometric_only = true;
        else if (std::string_view{argv[i]} == "--supplied-skeleton") supplied_skeleton = true;
        else if (std::string_view{argv[i]} == "--no-fit") fit = false;
        else if (std::string_view{argv[i]} == "--beams") generation.beams = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (std::string_view{argv[i]} == "--temperature") generation.temperature = std::stof(argv[++i]);
        else if (std::string_view{argv[i]} == "--max-tokens") generation.max_tokens = std::stoul(argv[++i]);
    }
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--geometric") generation.geometric_only = true;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--no-fit") fit = false;
    if (argc > 3 && std::string_view{argv[argc - 1]} == "--supplied-skeleton") supplied_skeleton = true;
    auto model = skintokens::model::load(argv[2], runtime);
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    if (std::string_view{argv[1]} == "inspect") {
        std::cout << "SkinTokens 0.1 model bundle\nbackend: " << model->backend_name() << '\n';
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
