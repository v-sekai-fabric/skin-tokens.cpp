#include <skintokens/skintokens.h>

#include <cassert>
#include <cstring>

int main() {
    const st_runtime_options runtime = st_default_runtime_options();
    const st_generation_options generation = st_default_generation_options();
    assert(runtime.device == ST_DEVICE_AUTO);
    assert(generation.top_k == 5U);
    assert(generation.beams == 10U);
    assert(generation.max_tokens >= 4U);
    char error[12]{};
    st_model * model = reinterpret_cast<st_model *>(1);
    const auto result = st_model_load(nullptr, nullptr, &model, error, sizeof(error));
    assert(result == ST_INVALID_ARGUMENT);
    assert(model == nullptr);
    assert(error[sizeof(error) - 1U] == '\0');
    assert(std::strlen(error) > 0U);
    assert(std::strcmp(st_model_backend_name(nullptr), "") == 0);
    assert(std::strlen(st_model_last_error(nullptr)) > 0U);
    assert(st_bind_files(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         error, sizeof(error)) == ST_INVALID_ARGUMENT);
    st_model_free(nullptr);
}
