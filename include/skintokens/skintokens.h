#ifndef SKINTOKENS_H
#define SKINTOKENS_H

#include <stddef.h>
#include <stdint.h>

#define ST_ABI_VERSION 1U

#if defined(_WIN32) && defined(SKINTOKENS_SHARED)
#  if defined(SKINTOKENS_BUILD)
#    define ST_API __declspec(dllexport)
#  else
#    define ST_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(SKINTOKENS_SHARED)
#  define ST_API __attribute__((visibility("default")))
#else
#  define ST_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_model st_model;

typedef enum st_status {
    ST_OK = 0,
    ST_INVALID_ARGUMENT,
    ST_INVALID_FORMAT,
    ST_LIMIT_EXCEEDED,
    ST_IO_ERROR,
    ST_INCOMPATIBLE_MODEL,
    ST_BACKEND_UNAVAILABLE,
    ST_ALLOCATION_FAILED,
    ST_COMPUTE_FAILED
} st_status;

typedef enum st_device { ST_DEVICE_AUTO = 0, ST_DEVICE_CPU = 1, ST_DEVICE_VULKAN = 2 } st_device;

typedef enum st_target_rig {
    ST_TARGET_GENERATED = 0,
    ST_TARGET_SOMA30 = 1,
    ST_TARGET_MIXAMO52 = 2
} st_target_rig;

typedef enum st_rig_kind {
    ST_RIG_UNKNOWN = 0,
    ST_RIG_SOMA30 = 1,
    ST_RIG_MIXAMO52 = 2
} st_rig_kind;

typedef struct st_runtime_options {
    st_device device;
    uint32_t threads;
    const char * backend_directory;
} st_runtime_options;

typedef struct st_generation_options {
    uint64_t seed;
    uint32_t top_k;
    float top_p;
    float temperature;
    float repetition_penalty;
    uint32_t beams;
    size_t max_tokens;
    int geometric_only;
    st_target_rig target_rig;
    int surface_postprocess;
} st_generation_options;

typedef struct st_mesh_info {
    size_t vertex_count;
    size_t triangle_count;
} st_mesh_info;

typedef struct st_motion_info {
    size_t frame_count;
    size_t joint_count;
    float frames_per_second;
} st_motion_info;

typedef struct st_glb_info {
    int has_mesh;
    int has_skin;
    int has_skeleton;
    int has_animation;
    size_t joint_count;
    size_t frame_count;
    float frames_per_second;
    st_rig_kind rig_kind;
} st_glb_info;

/* Return versioned library defaults. Prefer these over zero-initialising an
 * options structure so future defaults remain source-compatible. */
ST_API uint32_t st_abi_version(void);
ST_API st_runtime_options st_default_runtime_options(void);
ST_API st_generation_options st_default_generation_options(void);

ST_API st_status st_model_load(const char * bundle, const st_runtime_options * options,
                               st_model ** output, char * error, size_t error_capacity);
ST_API void st_model_free(st_model * value);
ST_API const char * st_model_backend_name(const st_model * value);
ST_API const char * st_model_last_error(const st_model * value);

/* Parse and validate an input without loading model weights. These functions
 * are useful for upload validation and are covered through the public C ABI by
 * the sanitizer/libFuzzer target. */
ST_API st_status st_inspect_mesh_file(const char * path, st_mesh_info * output,
                                      char * error, size_t error_capacity);
ST_API st_status st_inspect_motion_glb_file(const char * path, st_motion_info * output,
                                            char * error, size_t error_capacity);
ST_API st_status st_inspect_glb_file(const char * path, st_glb_info * output,
                                     char * error, size_t error_capacity);

/* Generate both a skeleton and learned skin weights for an unrigged mesh. */
ST_API st_status st_rig_file(st_model * value, const char * mesh_path,
                             const char * output_path,
                             const st_generation_options * options, int * learned,
                             char * error, size_t error_capacity);

/* Generate learned skin weights for a supplied armature. skeleton_path may
 * equal mesh_path when both are stored in one GLB. Static and animated
 * armatures are accepted. target_rig=SOMA30 preserves the supplied hierarchy;
 * target_rig=MIXAMO52 is accepted only for a detected SOMA30 hierarchy. */
ST_API st_status st_skin_files(st_model * value, const char * mesh_path,
                               const char * skeleton_path, const char * output_path,
                               int fit_skeleton_to_mesh,
                               const st_generation_options * options, int * learned,
                               char * error, size_t error_capacity);

ST_API st_status st_bind_glb_files(st_model * value, const char * mesh_path,
                                   const char * kimodo_motion_path, const char * output_path,
                                   const st_generation_options * options, int * learned,
                                   char * error, size_t error_capacity);

/* Preferred file API. Mesh input may be GLB or trellis2cpp T2MESH; motion and
 * output are GLB. Defaults bind the fitted SOMA30 hierarchy supplied by the
 * Kimodo motion. target_rig may instead request Mixamo52 or an unconstrained
 * generated rig; geometric_only bypasses learned skin-weight generation. */
ST_API st_status st_bind_files(st_model * value, const char * mesh_path,
                               const char * kimodo_motion_path, const char * output_path,
                               const st_generation_options * options, int * learned,
                               char * error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
