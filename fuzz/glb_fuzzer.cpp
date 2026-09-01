#include <skintokens/skintokens.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    if (size > (2U << 20U)) return 0;
    static const auto path = std::filesystem::temp_directory_path() / "skintokens-glb-fuzzer.glb";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) return 0;
        output.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    }
    (void) skintokens::load_glb_file(path);
    (void) skintokens::load_kimodo_glb_file(path);
    (void) skintokens::load_skinned_glb_file(path);
    (void) skintokens::load_trellis_mesh_file(path);
    return 0;
}
