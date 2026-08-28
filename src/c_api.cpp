#include <skintokens/skintokens.h>
#include <skintokens/skintokens.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

struct st_model {
    skintokens::model value;
    std::string backend;
    std::string last_error;
    explicit st_model(skintokens::model && model) : value(std::move(model)), backend(value.backend_name()) {}
};

namespace {
st_status status(skintokens::error_code code) {
    return static_cast<st_status>(static_cast<unsigned>(code) + 1U);
}
void copy_error(std::string_view message, char * output, std::size_t capacity) {
    if (output == nullptr || capacity == 0U) return;
    const auto count = std::min(capacity - 1U, message.size());
    std::memcpy(output, message.data(), count); output[count] = '\0';
}
st_status fail(st_model * model, st_status code, std::string message, char * output, std::size_t capacity) {
    if (model != nullptr) model->last_error = message;
    copy_error(message, output, capacity); return code;
}
skintokens::runtime_options runtime(const st_runtime_options * input) {
    skintokens::runtime_options output;
    if (input == nullptr) return output;
    output.device = input->device == ST_DEVICE_CPU ? skintokens::device_kind::cpu :
        input->device == ST_DEVICE_VULKAN ? skintokens::device_kind::vulkan : skintokens::device_kind::automatic;
    output.threads = input->threads;
    if (input->backend_directory != nullptr) output.backend_directory = input->backend_directory;
    return output;
}
skintokens::generation_options generation(const st_generation_options * input) {
    skintokens::generation_options output;
    if (input == nullptr) return output;
    output.seed = input->seed; output.top_k = input->top_k; output.top_p = input->top_p;
    output.temperature = input->temperature; output.repetition_penalty = input->repetition_penalty;
    output.beams = input->beams; output.max_tokens = input->max_tokens;
    output.geometric_only = input->geometric_only != 0;
    return output;
}
}

extern "C" st_runtime_options st_default_runtime_options(void) {
    return {ST_DEVICE_AUTO, 0U, nullptr};
}

extern "C" st_generation_options st_default_generation_options(void) {
    const skintokens::generation_options value;
    return {value.seed, value.top_k, value.top_p, value.temperature,
            value.repetition_penalty, value.beams, value.max_tokens, value.geometric_only ? 1 : 0};
}

extern "C" st_status st_model_load(const char * bundle, const st_runtime_options * options,
                                    st_model ** output, char * error, size_t error_capacity) try {
    if (output == nullptr)
        return fail(nullptr, ST_INVALID_ARGUMENT, "bundle and output are required", error, error_capacity);
    *output = nullptr;
    if (bundle == nullptr || bundle[0] == '\0' ||
        (options != nullptr && (options->device < ST_DEVICE_AUTO || options->device > ST_DEVICE_VULKAN)))
        return fail(nullptr, ST_INVALID_ARGUMENT, "bundle and valid runtime options are required", error, error_capacity);
    auto loaded = skintokens::model::load(bundle, runtime(options));
    if (!loaded) return fail(nullptr, status(loaded.error().code), loaded.error().message, error, error_capacity);
    auto handle = std::make_unique<st_model>(std::move(*loaded));
    *output = handle.release(); copy_error({}, error, error_capacity); return ST_OK;
} catch (const std::bad_alloc &) {
    return fail(nullptr, ST_ALLOCATION_FAILED, "allocation failed", error, error_capacity);
} catch (const std::exception & exception) {
    return fail(nullptr, ST_COMPUTE_FAILED, exception.what(), error, error_capacity);
} catch (...) {
    return fail(nullptr, ST_COMPUTE_FAILED, "unknown C++ exception", error, error_capacity);
}

extern "C" void st_model_free(st_model * value) { delete value; }
extern "C" const char * st_model_backend_name(const st_model * value) {
    return value == nullptr ? "" : value->backend.c_str();
}
extern "C" const char * st_model_last_error(const st_model * value) {
    return value == nullptr ? "invalid model handle" : value->last_error.c_str();
}

extern "C" st_status st_bind_glb_files(st_model * value, const char * mesh_path,
                                        const char * kimodo_motion_path, const char * output_path,
                                        const st_generation_options * options, int * learned,
                                        char * error, size_t error_capacity) {
    return st_bind_files(value, mesh_path, kimodo_motion_path, output_path,
                         options, learned, error, error_capacity);
}

extern "C" st_status st_bind_files(st_model * value, const char * mesh_path,
                                    const char * kimodo_motion_path, const char * output_path,
                                    const st_generation_options * options, int * learned,
                                    char * error, size_t error_capacity) try {
    if (value == nullptr || mesh_path == nullptr || kimodo_motion_path == nullptr || output_path == nullptr)
        return fail(value, ST_INVALID_ARGUMENT, "model, mesh, motion, and output paths are required", error, error_capacity);
    if (learned != nullptr) *learned = 0;
    const std::filesystem::path source_path = mesh_path;
    auto mesh = source_path.extension() == ".t2mesh" ?
        skintokens::load_trellis_mesh_file(source_path) : skintokens::load_glb_file(source_path);
    if (!mesh) return fail(value, status(mesh.error().code), mesh.error().message, error, error_capacity);
    auto motion = skintokens::load_kimodo_glb_file(kimodo_motion_path);
    if (!motion) return fail(value, status(motion.error().code), motion.error().message, error, error_capacity);
    const auto settings = generation(options);
    skintokens::result<skintokens::skin> binding;
    if (settings.geometric_only) {
        auto fitted = skintokens::fit_motion_to_mesh(*mesh, *motion);
        if (!fitted) return fail(value, status(fitted.error().code), fitted.error().message, error, error_capacity);
        motion = std::move(fitted);
        binding = value->value.bind(*mesh, motion->rig, settings);
    } else {
        binding = value->value.rig(*mesh, settings);
        if (binding) {
            auto retargeted = skintokens::retarget_motion_to_rig(*motion, binding->rig);
            if (!retargeted) return fail(value, status(retargeted.error().code), retargeted.error().message, error, error_capacity);
            motion = std::move(retargeted);
        }
    }
    if (!binding) return fail(value, status(binding.error().code), binding.error().message, error, error_capacity);
    auto saved = skintokens::save_skinned_animation_glb_file(output_path, *mesh, *binding, *motion);
    if (!saved) return fail(value, status(saved.error().code), saved.error().message, error, error_capacity);
    if (learned != nullptr) *learned = binding->learned ? 1 : 0;
    value->last_error.clear(); copy_error({}, error, error_capacity); return ST_OK;
} catch (const std::bad_alloc &) {
    return fail(value, ST_ALLOCATION_FAILED, "allocation failed", error, error_capacity);
} catch (const std::exception & exception) {
    return fail(value, ST_COMPUTE_FAILED, exception.what(), error, error_capacity);
} catch (...) {
    return fail(value, ST_COMPUTE_FAILED, "unknown C++ exception", error, error_capacity);
}
