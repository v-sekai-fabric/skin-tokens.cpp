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
    st_retarget_options *retarget = NULL;
    assert(st_retarget_options_create(&retarget, error, sizeof(error)) == ST_OK);
    assert(retarget != NULL);
    float confidence = 0.0F;
    int boolean = 0;
    st_finger_transfer fingers = 99U;
    assert(st_retarget_options_get_minimum_confidence(
        retarget, &confidence, error, sizeof(error)) == ST_OK);
    assert(confidence == 0.8F);
    assert(st_retarget_options_set_minimum_confidence(
        retarget, 0.65F, error, sizeof(error)) == ST_OK);
    assert(st_retarget_options_get_minimum_confidence(
        retarget, &confidence, error, sizeof(error)) == ST_OK && confidence == 0.65F);
    assert(st_retarget_options_set_minimum_confidence(
        retarget, 2.0F, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(st_retarget_options_get_allow_flexible_hands(
        retarget, &boolean, error, sizeof(error)) == ST_OK && boolean == 1);
    assert(st_retarget_options_set_allow_flexible_hands(
        retarget, 0, error, sizeof(error)) == ST_OK);
    assert(st_retarget_options_get_allow_flexible_hands(
        retarget, &boolean, error, sizeof(error)) == ST_OK && boolean == 0);
    assert(st_retarget_options_set_finger_transfer(
        retarget, ST_FINGER_TRANSFER_MAP_SOMA_ENDPOINTS, error, sizeof(error)) == ST_OK);
    assert(st_retarget_options_get_finger_transfer(
        retarget, &fingers, error, sizeof(error)) == ST_OK &&
        fingers == ST_FINGER_TRANSFER_MAP_SOMA_ENDPOINTS);
    assert(st_retarget_options_set_scale_root_motion(
        retarget, 0, error, sizeof(error)) == ST_OK);
    assert(st_retarget_options_get_scale_root_motion(
        retarget, &boolean, error, sizeof(error)) == ST_OK && boolean == 0);
    assert(st_humanoid_match_get_confidence(NULL, &confidence, error, sizeof(error)) ==
           ST_INVALID_ARGUMENT);
    uint64_t joint_count = 0U;
    st_semantic_role role = ST_ROLE_UNMAPPED;
    int64_t target_joint = -1;
    assert(st_humanoid_match_get_generated_joint_count(
        NULL, &joint_count, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(st_humanoid_match_get_generated_joint_role(
        NULL, 0U, &role, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(st_humanoid_match_get_soma30_target_joint(
        NULL, 0U, &target_joint, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    assert(st_retarget_soma30_glb_files(
        NULL, NULL, NULL, NULL, retarget, error, sizeof(error)) == ST_INVALID_ARGUMENT);
    st_retarget_options_free(retarget);
    st_retarget_options_free(NULL);
    st_humanoid_match_free(NULL);
    st_model_free(NULL);
    return 0;
}
