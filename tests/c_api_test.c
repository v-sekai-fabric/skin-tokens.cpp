#include <skintokens/skintokens.h>

#include <assert.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    assert(st_abi_version() == ST_ABI_VERSION);
    const st_runtime_options runtime = st_default_runtime_options();
    const st_generation_options generation = st_default_generation_options();
    assert(runtime.device == ST_DEVICE_AUTO);
    assert(generation.top_k == 5U);
    assert(generation.beams == 10U);
    assert(generation.max_tokens >= 4U);
    assert(generation.target_rig == ST_TARGET_SOMA30);
    assert(generation.surface_postprocess == 0);
    assert(ST_FIT_NONE == 0);
    assert(ST_FIT_GLOBAL_SIMILARITY == 1);
    assert(ST_FIT_ARTICULATED == 2);
    char error[12] = {0};
    st_model *model = (st_model *)(uintptr_t)1;
    const st_status result = st_model_load(NULL, NULL, &model, error, sizeof(error));
    assert(result == ST_INVALID_ARGUMENT);
    assert(model == NULL);
    assert(error[sizeof(error) - 1U] == '\0');
    assert(strlen(error) > 0U);
    assert(strcmp(st_model_backend_name(NULL), "") == 0);
    assert(strlen(st_model_last_error(NULL)) > 0U);
    assert(st_bind_files(NULL, NULL, NULL, NULL, NULL, NULL,
                         error, sizeof(error)) == ST_INVALID_ARGUMENT);
    st_mesh_info mesh = {7U, 9U};
    assert(st_inspect_mesh_file(NULL, &mesh, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(mesh.vertex_count == 0U && mesh.triangle_count == 0U);
    st_motion_info motion = {7U, 9U, 12.0F};
    assert(st_inspect_motion_glb_file(NULL, &motion, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(motion.frame_count == 0U && motion.joint_count == 0U && motion.frames_per_second == 0.0F);
    st_glb_info glb = {1, 1, 1, 1, 7U, 9U, 12.0F, ST_RIG_SOMA30};
    assert(st_inspect_glb_file(NULL, &glb, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(glb.has_mesh == 0 && glb.has_skeleton == 0 && glb.joint_count == 0U);
    assert(st_rig_file(NULL, NULL, NULL, NULL, NULL, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(st_skin_files(NULL, NULL, NULL, NULL, 0, NULL, NULL,
                         error, sizeof(error)) == ST_INVALID_ARGUMENT);
    st_model_free(NULL);
    return 0;
}
