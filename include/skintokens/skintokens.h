#ifndef SKINTOKENS_H
#define SKINTOKENS_H

#include <stddef.h>
#include <stdint.h>

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
} st_generation_options;

/* Return versioned library defaults. Prefer these over zero-initialising an
 * options structure so future defaults remain source-compatible. */
ST_API st_runtime_options st_default_runtime_options(void);
ST_API st_generation_options st_default_generation_options(void);

ST_API st_status st_model_load(const char * bundle, const st_runtime_options * options,
                               st_model ** output, char * error, size_t error_capacity);
ST_API void st_model_free(st_model * value);
ST_API const char * st_model_backend_name(const st_model * value);
ST_API const char * st_model_last_error(const st_model * value);

ST_API st_status st_bind_glb_files(st_model * value, const char * mesh_path,
                                   const char * kimodo_motion_path, const char * output_path,
                                   const st_generation_options * options, int * learned,
                                   char * error, size_t error_capacity);

/* Preferred file API. Mesh input may be GLB or trellis2cpp T2MESH; motion and
 * output are GLB. The default generates a mesh-native TokenRig skeleton and
 * retargets the Kimodo motion. geometric_only binds the fitted Kimodo rig. */
ST_API st_status st_bind_files(st_model * value, const char * mesh_path,
                               const char * kimodo_motion_path, const char * output_path,
                               const st_generation_options * options, int * learned,
                               char * error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
