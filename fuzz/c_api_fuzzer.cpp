#include <skintokens/skintokens.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <process.h>
#define ST_GETPID _getpid
#else
#include <unistd.h>
#define ST_GETPID getpid
#endif

namespace {

const std::filesystem::path & input_path() {
    static const auto value = std::filesystem::temp_directory_path() /
        ("skintokens-c-api-fuzzer-" + std::to_string(ST_GETPID()) + ".glb");
    return value;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t * data, std::size_t size) {
    if (size > (2U << 20U)) return 0;
    const auto & path = input_path();
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) return 0;
        output.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
    }

    std::array<char, 65> error{};
    const std::size_t capacity = size == 0U ? 0U : static_cast<std::size_t>(data[0] % error.size());
    st_mesh_info mesh{};
    st_motion_info motion{};
    st_glb_info glb{};
    (void) st_inspect_mesh_file(path.string().c_str(), &mesh, error.data(), capacity);
    (void) st_inspect_motion_glb_file(path.string().c_str(), &motion, error.data(), capacity);
    (void) st_inspect_glb_file(path.string().c_str(), &glb, error.data(), capacity);

    // Exercise nullability and fixed-buffer behavior across the remaining C
    // surface. GGUF/model loading is deliberately outside this fuzz target.
    (void) st_inspect_mesh_file(path.string().c_str(), nullptr,
                                capacity == 0U ? nullptr : error.data(), capacity);
    (void) st_inspect_motion_glb_file(nullptr, &motion,
                                      capacity == 0U ? nullptr : error.data(), capacity);
    (void) st_bind_files(nullptr, path.string().c_str(), path.string().c_str(),
                         path.string().c_str(), nullptr, nullptr,
                         capacity == 0U ? nullptr : error.data(), capacity);
    (void) st_rig_file(nullptr, path.string().c_str(), path.string().c_str(), nullptr, nullptr,
                       capacity == 0U ? nullptr : error.data(), capacity);
    (void) st_skin_files(nullptr, path.string().c_str(), path.string().c_str(),
                         path.string().c_str(), 0, nullptr, nullptr,
                         capacity == 0U ? nullptr : error.data(), capacity);
    st_retarget_options *options = nullptr;
    (void) st_retarget_options_create(&options,
        capacity == 0U ? nullptr : error.data(), capacity);
    if (options != nullptr) {
        const float confidence = size > 1U ? static_cast<float>(data[1]) / 255.0F : 0.8F;
        (void) st_retarget_options_set_minimum_confidence(options, confidence,
            capacity == 0U ? nullptr : error.data(), capacity);
        (void) st_retarget_options_set_allow_flexible_hands(options, size > 2U ? data[2] & 1U : 1,
            capacity == 0U ? nullptr : error.data(), capacity);
        (void) st_retarget_options_set_finger_transfer(options,
            size > 3U ? data[3] % 2U : ST_FINGER_TRANSFER_NEUTRAL,
            capacity == 0U ? nullptr : error.data(), capacity);
        st_humanoid_match *match = nullptr;
        (void) st_humanoid_match_glb_file(path.string().c_str(), options, &match,
            capacity == 0U ? nullptr : error.data(), capacity);
        st_humanoid_match_free(match);
        st_retarget_options_free(options);
    }
    (void) st_model_backend_name(nullptr);
    (void) st_model_last_error(nullptr);
    (void) st_abi_version();
    st_model_free(nullptr);
    return 0;
}
