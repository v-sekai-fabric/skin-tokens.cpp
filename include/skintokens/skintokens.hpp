#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if defined(SKINTOKENS_SHARED)
#  if defined(_WIN32) && !defined(__MINGW32__)
#    if defined(SKINTOKENS_BUILD)
#      define SKINTOKENS_API __declspec(dllexport)
#    else
#      define SKINTOKENS_API __declspec(dllimport)
#    endif
#  else
#    define SKINTOKENS_API __attribute__((visibility("default")))
#  endif
#else
#  define SKINTOKENS_API
#endif

namespace skintokens {

inline constexpr std::string_view version = "0.1.0";

enum class error_code : std::uint8_t {
    invalid_argument,
    invalid_format,
    limit_exceeded,
    io,
    incompatible_model,
    backend_unavailable,
    allocation,
    compute,
};

struct error {
    error_code code = error_code::invalid_argument;
    std::string message;
};

template<class T> using result = std::expected<T, error>;

struct vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct quat {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float w = 1.0F;
};

struct color4 {
    float r = 1.0F;
    float g = 1.0F;
    float b = 1.0F;
    float a = 1.0F;
};

using triangle = std::array<std::uint32_t, 3>;

struct mesh {
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    // Linear RGBA vertex colours. Trellis' standard portable GLB export uses
    // COLOR_0, so retaining this stream keeps its generated appearance.
    std::vector<color4> colors;
    std::vector<triangle> faces;
    float metallic_factor = 0.0F;
    float roughness_factor = 0.72F;
    bool double_sided = false;
};

struct skeleton {
    std::vector<std::string> names;
    std::vector<std::int32_t> parents;
    std::vector<vec3> rest_positions;
};

struct motion {
    std::size_t frames = 0;
    float frames_per_second = 30.0F;
    skeleton rig;
    std::vector<vec3> root_translations;
    std::vector<quat> local_rotations; // [frame, joint]
};

struct skin {
    skeleton rig;
    // Four normalized influences per source vertex, ready for glTF JOINTS_0
    // and WEIGHTS_0. TokenRig's dense matrix is intentionally not retained.
    std::vector<std::array<std::uint16_t, 4>> joints;
    std::vector<std::array<float, 4>> weights;
    // False denotes the deterministic geometric integration baseline. It is
    // never presented as learned TokenRig output.
    bool learned = false;
};

enum class device_kind : std::uint8_t { automatic, cpu, vulkan };

struct runtime_options {
    device_kind device = device_kind::automatic;
    std::uint32_t threads = 0;
    std::filesystem::path backend_directory;
};

struct generation_options {
    std::uint64_t seed = 0;
    std::uint32_t top_k = 5;
    float top_p = 0.95F;
    float temperature = 1.0F;
    float repetition_penalty = 2.0F;
    std::uint32_t beams = 10;
    std::size_t max_tokens = 2048;
    // Diagnostic integration fallback; learned TokenRig/SkinVAE is the default.
    bool geometric_only = false;
};

class SKINTOKENS_API model final {
public:
    model(model &&) noexcept;
    model & operator=(model &&) noexcept;
    model(const model &) = delete;
    model & operator=(const model &) = delete;
    ~model();

    [[nodiscard]] static result<model> load(
        const std::filesystem::path & bundle,
        const runtime_options & options = {});

    [[nodiscard]] result<skin> rig(
        const mesh & source,
        const generation_options & options = {}) const;

    // Preferred Kimodo integration: the supplied skeleton is emitted as a
    // constrained TokenRig prefix, so inference generates skin weights only.
    [[nodiscard]] result<skin> bind(
        const mesh & source,
        const skeleton & target,
        const generation_options & options = {}) const;

    [[nodiscard]] std::string_view backend_name() const noexcept;

private:
    struct impl;
    explicit model(std::unique_ptr<impl>) noexcept;
    std::unique_ptr<impl> impl_;
};

[[nodiscard]] SKINTOKENS_API result<mesh> load_glb_file(
    const std::filesystem::path & path);

// Reads trellis2cpp's versioned persisted mesh format directly. Coordinates
// and PBR vertex colours are converted exactly as its portable GLB exporter.
[[nodiscard]] SKINTOKENS_API result<mesh> load_trellis_mesh_file(
    const std::filesystem::path & path);

[[nodiscard]] SKINTOKENS_API result<motion> load_kimodo_glb_file(
    const std::filesystem::path & path);

// Uniformly fits a motion rig to the mesh's vertical extent and centre. This
// is useful when a motion-only Kimodo skeleton and a separately generated
// Trellis character do not share an asset coordinate system.
[[nodiscard]] SKINTOKENS_API result<motion> fit_motion_to_mesh(
    const mesh & geometry, const motion & animation);

// Transfers a motion onto a mesh-specific SkinTokens rig by matching its
// generated rest-pose topology and normalized joint positions. Root travel is
// scaled to the target rig while local rotations remain frame-exact.
[[nodiscard]] SKINTOKENS_API result<motion> retarget_motion_to_rig(
    const motion & animation, const skeleton & target);

// Evaluates the same glTF linear-blend skinning transform emitted by
// save_skinned_animation_glb_file at an exact animation frame. This is useful
// for render-independent parity tests and offline vertex export.
[[nodiscard]] SKINTOKENS_API result<std::vector<vec3>> deform_vertices(
    const mesh & geometry,
    const skin & binding,
    const motion & animation,
    std::size_t frame);

[[nodiscard]] SKINTOKENS_API result<void> save_skinned_animation_glb_file(
    const std::filesystem::path & path,
    const mesh & geometry,
    const skin & binding,
    const motion & animation);

} // namespace skintokens
