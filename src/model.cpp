#include "internal.hpp"
#include "binding.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>

namespace skintokens {

struct model::impl {
    std::unique_ptr<detail::backend_handle> backend;
    std::unique_ptr<detail::weight_component> mesh_weights;
    std::unique_ptr<detail::weight_component> tokenrig_weights;
    std::unique_ptr<detail::weight_component> skin_vae_weights;
    detail::bundle_metadata mesh_encoder;
    detail::bundle_metadata tokenrig;
    detail::bundle_metadata skin_vae;
};

namespace {

// SkinTokens is trained through Blender and therefore consumes Z-up geometry.
// Public mesh/motion I/O follows glTF/Three.js and is Y-up.  Keep that
// conversion at the model boundary so exported geometry is not rotated.
vec3 to_model_space(vec3 value) { return {value.x, -value.z, value.y}; }
vec3 from_model_space(vec3 value) { return {value.x, value.z, -value.y}; }

std::vector<vec3> to_model_space(std::span<const vec3> values) {
    std::vector<vec3> output;
    output.reserve(values.size());
    for (const auto value : values) output.push_back(to_model_space(value));
    return output;
}

skeleton to_model_space(const skeleton & source) {
    skeleton output = source;
    output.rest_positions = to_model_space(source.rest_positions);
    return output;
}

skeleton from_model_space(const skeleton & source) {
    skeleton output = source;
    output.rest_positions.reserve(source.rest_positions.size());
    output.rest_positions.clear();
    for (const auto value : source.rest_positions) output.rest_positions.push_back(from_model_space(value));
    return output;
}

quat multiply(quat a, quat b) {
    return {
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    };
}

quat conjugate(quat value) { return {-value.x, -value.y, -value.z, value.w}; }

quat normalized(quat value) {
    const float length = std::sqrt(value.x*value.x + value.y*value.y +
                                   value.z*value.z + value.w*value.w);
    if (length <= 1e-12F) return {};
    return {value.x/length, value.y/length, value.z/length, value.w/length};
}

result<void> validate_mesh(const mesh & source) {
    if (source.vertices.empty() || source.faces.empty())
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh must contain vertices and triangles"));
    if (!source.normals.empty() && source.normals.size() != source.vertices.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh normals must match vertex count"));
    for (const auto & value : source.vertices) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh contains non-finite positions"));
    }
    for (const auto & face : source.faces) {
        if (face[0] >= source.vertices.size() || face[1] >= source.vertices.size() || face[2] >= source.vertices.size())
            return std::unexpected(detail::fail(error_code::invalid_argument, "mesh triangle index is out of range"));
    }
    return {};
}

result<void> validate_skeleton(const skeleton & target) {
    const std::size_t count = target.names.size();
    if (count == 0 || count > 256U || target.parents.size() != count || target.rest_positions.size() != count)
        return std::unexpected(detail::fail(error_code::invalid_argument, "skeleton arrays must contain 1..256 matching joints"));
    std::size_t roots = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto parent = target.parents[i];
        if (parent < 0) ++roots;
        else if (static_cast<std::size_t>(parent) >= i)
            return std::unexpected(detail::fail(error_code::invalid_argument, "skeleton parents must precede children"));
    }
    if (roots != 1U) return std::unexpected(detail::fail(error_code::invalid_argument, "skeleton must contain exactly one root"));
    return {};
}

std::vector<vec3> vertex_normals(const mesh & source) {
    if (source.normals.size() == source.vertices.size()) return source.normals;
    std::vector<vec3> result(source.vertices.size());
    for (const auto & face : source.faces) {
        const auto a = source.vertices[face[0]];
        const auto b = source.vertices[face[1]];
        const auto c = source.vertices[face[2]];
        const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        const vec3 n{uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
        for (const auto index : face) {
            result[index].x += n.x; result[index].y += n.y; result[index].z += n.z;
        }
    }
    for (auto & value : result) {
        const float length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
        if (length > 1e-12F) { value.x /= length; value.y /= length; value.z /= length; }
        else value = {0.0F, 1.0F, 0.0F};
    }
    return result;
}

std::vector<std::int32_t> farthest_points(std::span<const vec3> points, std::size_t wanted) {
    wanted = std::min(wanted, points.size());
    std::vector<std::int32_t> output;
    output.reserve(wanted);
    std::vector<float> distance(points.size(), std::numeric_limits<float>::infinity());
    std::size_t farthest = 0;
    for (std::size_t sample = 0; sample < wanted; ++sample) {
        output.push_back(static_cast<std::int32_t>(farthest));
        const auto center = points[farthest];
        for (std::size_t index = 0; index < points.size(); ++index) {
            const float x = points[index].x - center.x;
            const float y = points[index].y - center.y;
            const float z = points[index].z - center.z;
            distance[index] = std::min(distance[index], x * x + y * y + z * z);
        }
        farthest = static_cast<std::size_t>(std::distance(distance.begin(),
            std::max_element(distance.begin(), distance.end())));
    }
    return output;
}

struct sampled_cloud { std::vector<vec3> points; std::vector<vec3> normals; };

result<sampled_cloud> sample_mesh_surface(const mesh & source, std::span<const vec3> normalized,
                                          std::span<const vec3> vertex_normal, std::uint64_t seed) {
    constexpr std::size_t total_samples = 54000U;
    // The released predict config declares 16,384 vertex samples, but
    // SamplerMix.sample() does not forward that member to
    // sample_vertex_groups(). Match the executable upstream path: all 54K
    // samples are area-weighted surface points.
    constexpr std::size_t vertex_samples = 0U;
    sampled_cloud output;
    output.points.reserve(total_samples); output.normals.reserve(total_samples);
    std::mt19937_64 random{seed};
    std::vector<std::size_t> order(normalized.size());
    std::iota(order.begin(), order.end(), 0U);
    std::shuffle(order.begin(), order.end(), random);
    const std::size_t direct = std::min(vertex_samples, order.size());
    for (std::size_t i = 0; i < direct; ++i) {
        output.points.push_back(normalized[order[i]]);
        output.normals.push_back(vertex_normal[order[i]]);
    }
    std::vector<double> areas(source.faces.size());
    std::vector<vec3> face_normals(source.faces.size());
    for (std::size_t i = 0; i < source.faces.size(); ++i) {
        const auto & face = source.faces[i];
        const vec3 left{normalized[face[1]].x - normalized[face[0]].x,
                        normalized[face[1]].y - normalized[face[0]].y,
                        normalized[face[1]].z - normalized[face[0]].z};
        const vec3 right{normalized[face[2]].x - normalized[face[0]].x,
                         normalized[face[2]].y - normalized[face[0]].y,
                         normalized[face[2]].z - normalized[face[0]].z};
        const vec3 value{left.y*right.z - left.z*right.y,
                         left.z*right.x - left.x*right.z,
                         left.x*right.y - left.y*right.x};
        const float length = std::sqrt(value.x*value.x + value.y*value.y + value.z*value.z);
        areas[i] = length;
        face_normals[i] = length > 1e-12F ? vec3{value.x/length, value.y/length, value.z/length} :
                                             vec3{0.0F, 1.0F, 0.0F};
    }
    if (std::accumulate(areas.begin(), areas.end(), 0.0) <= 0.0)
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh has no non-degenerate surface"));
    std::discrete_distribution<std::size_t> face_distribution(areas.begin(), areas.end());
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    while (output.points.size() < total_samples) {
        const std::size_t face_index = face_distribution(random);
        const auto & face = source.faces[face_index];
        float u = unit(random), v = unit(random);
        if (u + v > 1.0F) { u = 1.0F - u; v = 1.0F - v; }
        const auto a = normalized[face[0]], b = normalized[face[1]], c = normalized[face[2]];
        output.points.push_back({a.x + (b.x-a.x)*u + (c.x-a.x)*v,
                                 a.y + (b.y-a.y)*u + (c.y-a.y)*v,
                                 a.z + (b.z-a.z)*u + (c.z-a.z)*v});
        output.normals.push_back(face_normals[face_index]);
    }
    return output;
}

std::vector<std::int32_t> sampled_farthest_points(std::span<const vec3> points,
                                                   std::size_t candidates, std::size_t wanted,
                                                   std::uint64_t seed) {
    std::vector<std::size_t> order(points.size());
    std::iota(order.begin(), order.end(), 0U);
    std::mt19937_64 random{seed}; std::shuffle(order.begin(), order.end(), random);
    order.resize(std::min(candidates, order.size()));
    std::vector<vec3> subset; subset.reserve(order.size());
    for (const auto index : order) subset.push_back(points[index]);
    const auto local = farthest_points(subset, wanted);
    std::vector<std::int32_t> output; output.reserve(local.size());
    for (const auto index : local) output.push_back(static_cast<std::int32_t>(order[static_cast<std::size_t>(index)]));
    return output;
}

skin geometric_binding(const mesh & source, const skeleton & target) {
    skin output;
    output.rig = target;
    output.joints.resize(source.vertices.size());
    output.weights.resize(source.vertices.size());
    const auto squared_distance_to_bone = [&](vec3 point, std::size_t joint) {
        const auto end = target.rest_positions[joint];
        const auto parent = target.parents[joint];
        const auto start = parent < 0 ? end : target.rest_positions[static_cast<std::size_t>(parent)];
        const float dx = end.x - start.x, dy = end.y - start.y, dz = end.z - start.z;
        const float length2 = dx * dx + dy * dy + dz * dz;
        float t = length2 > 1e-12F ? ((point.x - start.x) * dx + (point.y - start.y) * dy +
            (point.z - start.z) * dz) / length2 : 0.0F;
        t = std::clamp(t, 0.0F, 1.0F);
        const float ex = point.x - (start.x + t * dx);
        const float ey = point.y - (start.y + t * dy);
        const float ez = point.z - (start.z + t * dz);
        return ex * ex + ey * ey + ez * ez;
    };
    for (std::size_t vertex = 0; vertex < source.vertices.size(); ++vertex) {
        std::array<std::pair<float, std::uint16_t>, 4> closest{};
        for (auto & entry : closest) entry = {std::numeric_limits<float>::infinity(), 0U};
        for (std::size_t joint = 0; joint < target.names.size(); ++joint) {
            const auto candidate = std::pair{squared_distance_to_bone(source.vertices[vertex], joint),
                                              static_cast<std::uint16_t>(joint)};
            if (candidate.first >= closest.back().first) continue;
            closest.back() = candidate;
            std::sort(closest.begin(), closest.end());
        }
        float sum = 0.0F;
        for (std::size_t slot = 0; slot < closest.size(); ++slot) {
            output.joints[vertex][slot] = closest[slot].second;
            output.weights[vertex][slot] = 1.0F / std::max(closest[slot].first, 1e-5F);
            sum += output.weights[vertex][slot];
        }
        for (auto & value : output.weights[vertex]) value /= sum;
    }
    output.learned = false;
    return output;
}

void dump_binding_trace_if_requested(
    const std::vector<std::vector<float>> & dense, const skin & output,
    std::span<const vec3> normalized_vertices,
    std::span<const std::uint32_t> neighbor_indices,
    std::span<const float> interpolation_weights) {
    const char * prefix = std::getenv("SKINTOKENS_DUMP_BINDING_TRACE_PREFIX");
    if (prefix == nullptr || *prefix == '\0') return;
    const std::filesystem::path base{prefix};
    const auto write = [](const std::filesystem::path & path, const void * data, std::size_t bytes) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create requested binding diagnostic dump");
        stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        if (!stream) throw std::runtime_error("cannot write requested binding diagnostic dump");
    };
    std::ofstream dense_stream(base.string() + ".dense.f32", std::ios::binary | std::ios::trunc);
    if (!dense_stream) throw std::runtime_error("cannot create requested dense binding dump");
    for (const auto & joint : dense)
        dense_stream.write(reinterpret_cast<const char *>(joint.data()),
                           static_cast<std::streamsize>(joint.size() * sizeof(float)));
    if (!dense_stream) throw std::runtime_error("cannot write requested dense binding dump");
    write(base.string() + ".joints.u16", output.joints.data(), output.joints.size() * sizeof(output.joints.front()));
    write(base.string() + ".weights.f32", output.weights.data(), output.weights.size() * sizeof(output.weights.front()));
    write(base.string() + ".normalized.f32", normalized_vertices.data(), normalized_vertices.size_bytes());
    write(base.string() + ".neighbors.u32", neighbor_indices.data(), neighbor_indices.size_bytes());
    write(base.string() + ".interpolation.f32", interpolation_weights.data(), interpolation_weights.size_bytes());
}

result<skin> decode_binding(const detail::weight_component & weights, ggml_backend_t backend,
                            const skeleton & target,
                            std::span<const std::int32_t> codes,
                            std::span<const float> vae_condition,
                            std::span<const vec3> sampled_points,
                            std::span<const vec3> sampled_normals,
                            std::span<const vec3> normalized_vertices) {
    if (codes.size() != target.names.size() * 4U)
        return std::unexpected(detail::fail(error_code::compute, "skin-code count does not match generated skeleton"));
    std::vector<std::vector<float>> dense(target.names.size());
    for (std::size_t joint = 0; joint < target.names.size(); ++joint) {
        auto decoded = detail::decode_skin_joint(weights, backend,
            std::span{codes.data() + joint * 4U, 4U}, vae_condition,
            sampled_points, sampled_normals);
        if (!decoded) return std::unexpected(decoded.error());
        dense[joint] = std::move(*decoded);
    }
    // Match Asset.from_data(): the released runtime decodes on its 54K
    // sampled cloud and interpolates each source vertex from eight nearest
    // samples. The exporter keeps the greatest four learned influences.
    const bool trace_binding = [] {
        const char * value = std::getenv("SKINTOKENS_DUMP_BINDING_TRACE_PREFIX");
        return value != nullptr && *value != '\0';
    }();
    detail::binding_trace trace;
    auto output = detail::integrate_learned_binding(target, normalized_vertices,
        sampled_points, dense, trace_binding ? &trace : nullptr);
    dump_binding_trace_if_requested(dense, output, normalized_vertices,
                                    trace.neighbor_indices, trace.interpolation_weights);
    return output;
}

void dump_mesh_condition_if_requested(std::span<const float> values) {
    const char * path = std::getenv("SKINTOKENS_DUMP_MESH_CONDITION");
    if (path == nullptr || *path == '\0') return;
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create requested mesh-condition dump");
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    if (!stream) throw std::runtime_error("cannot write requested mesh-condition dump");
}

template<class T>
void dump_binary(const std::filesystem::path & path, std::span<const T> values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create requested diagnostic dump");
    stream.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    if (!stream) throw std::runtime_error("cannot write requested diagnostic dump");
}

void dump_mesh_input_if_requested(const sampled_cloud & sampled,
                                  std::span<const std::int32_t> queries) {
    const char * prefix = std::getenv("SKINTOKENS_DUMP_MESH_INPUT_PREFIX");
    if (prefix == nullptr || *prefix == '\0') return;
    const std::filesystem::path base{prefix};
    dump_binary(base.string() + ".points.f32", std::span{sampled.points});
    dump_binary(base.string() + ".normals.f32", std::span{sampled.normals});
    dump_binary(base.string() + ".queries.i32", queries);
}

void dump_vae_trace_if_requested(const sampled_cloud & sampled,
                                 std::span<const std::int32_t> queries,
                                 std::span<const float> condition) {
    const char * prefix = std::getenv("SKINTOKENS_DUMP_VAE_TRACE_PREFIX");
    if (prefix == nullptr || *prefix == '\0') return;
    const std::filesystem::path base{prefix};
    dump_binary(base.string() + ".points.f32", std::span{sampled.points});
    dump_binary(base.string() + ".normals.f32", std::span{sampled.normals});
    dump_binary(base.string() + ".queries.i32", queries);
    dump_binary(base.string() + ".condition.f32", condition);
}

void dump_token_trace_if_requested(std::span<const std::int32_t> prefix,
                                   std::span<const std::int32_t> codes) {
    const char * base_value = std::getenv("SKINTOKENS_DUMP_TOKEN_TRACE_PREFIX");
    if (base_value == nullptr || *base_value == '\0') return;
    const std::filesystem::path base{base_value};
    dump_binary(base.string() + ".prefix.i32", prefix);
    dump_binary(base.string() + ".codes.i32", codes);
}

} // namespace

model::model(std::unique_ptr<impl> value) noexcept : impl_(std::move(value)) {}
model::model(model &&) noexcept = default;
model & model::operator=(model &&) noexcept = default;
model::~model() = default;

result<model> model::load(const std::filesystem::path & bundle, const runtime_options & options) {
    if (!std::filesystem::is_directory(bundle))
        return std::unexpected(detail::fail(error_code::io, "model bundle must be a directory"));
    auto output = std::make_unique<impl>();
    auto mesh_encoder = detail::inspect_gguf(bundle / "mesh-encoder.gguf");
    auto tokenrig = detail::inspect_gguf(bundle / "tokenrig.gguf");
    auto skin_vae = detail::inspect_gguf(bundle / "skin-vae.gguf");
    if (!mesh_encoder) return std::unexpected(mesh_encoder.error());
    if (!tokenrig) return std::unexpected(tokenrig.error());
    if (!skin_vae) return std::unexpected(skin_vae.error());
    if (mesh_encoder->component != "mesh-encoder" || tokenrig->component != "tokenrig" || skin_vae->component != "skin-vae")
        return std::unexpected(detail::fail(error_code::incompatible_model, "model bundle contains a component in the wrong file"));
    const auto same_identity = [&](const detail::bundle_metadata & value) {
        return value.upstream_revision == mesh_encoder->upstream_revision &&
               value.tokenrig_sha256 == mesh_encoder->tokenrig_sha256 &&
               value.skin_vae_sha256 == mesh_encoder->skin_vae_sha256;
    };
    if (!same_identity(*tokenrig) || !same_identity(*skin_vae))
        return std::unexpected(detail::fail(error_code::incompatible_model, "model components have mismatched source identities"));
    auto backend = detail::make_backend(options);
    if (!backend) return std::unexpected(backend.error());
    auto mesh_weights = detail::load_component(bundle / "mesh-encoder.gguf", (*backend)->value);
    if (!mesh_weights) return std::unexpected(mesh_weights.error());
    auto tokenrig_weights = detail::load_component(bundle / "tokenrig.gguf", (*backend)->value);
    if (!tokenrig_weights) return std::unexpected(tokenrig_weights.error());
    auto skin_vae_weights = detail::load_component(bundle / "skin-vae.gguf", (*backend)->value);
    if (!skin_vae_weights) return std::unexpected(skin_vae_weights.error());
    output->mesh_encoder = std::move(*mesh_encoder);
    output->tokenrig = std::move(*tokenrig);
    output->skin_vae = std::move(*skin_vae);
    output->backend = std::move(*backend);
    output->mesh_weights = std::move(*mesh_weights);
    output->tokenrig_weights = std::move(*tokenrig_weights);
    output->skin_vae_weights = std::move(*skin_vae_weights);
    return model{std::move(output)};
}

result<skin> model::rig(const mesh & source, const generation_options & options) const {
    auto valid = validate_mesh(source);
    if (!valid) return std::unexpected(valid.error());
    if (options.geometric_only)
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "geometric rigging requires a supplied skeleton"));
    const auto model_vertices = to_model_space(source.vertices);
    vec3 low = model_vertices.front(), high = low;
    for (const auto value : model_vertices) {
        low.x = std::min(low.x, value.x); low.y = std::min(low.y, value.y); low.z = std::min(low.z, value.z);
        high.x = std::max(high.x, value.x); high.y = std::max(high.y, value.y); high.z = std::max(high.z, value.z);
    }
    const vec3 center{(low.x + high.x) * 0.5F, (low.y + high.y) * 0.5F, (low.z + high.z) * 0.5F};
    const float scale = std::max({high.x-low.x, high.y-low.y, high.z-low.z}) * 0.5F;
    if (!std::isfinite(scale) || scale <= 1e-12F)
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh bounds are degenerate"));
    std::vector<vec3> normalized; normalized.reserve(model_vertices.size());
    for (const auto value : model_vertices)
        normalized.push_back({(value.x-center.x)/scale, (value.y-center.y)/scale, (value.z-center.z)/scale});
    auto normals = to_model_space(vertex_normals(source));
    auto sampled = sample_mesh_surface(source, normalized, normals, options.seed);
    if (!sampled) return std::unexpected(sampled.error());
    auto mesh_queries = sampled_farthest_points(sampled->points, 2048U, 512U, 0U);
    dump_mesh_input_if_requested(*sampled, mesh_queries);
    auto mesh_condition = detail::encode_mesh(*impl_->mesh_weights, impl_->backend->value,
        sampled->points, sampled->normals, mesh_queries);
    if (!mesh_condition) return std::unexpected(mesh_condition.error());
    dump_mesh_condition_if_requested(*mesh_condition);
    auto vae_queries = sampled_farthest_points(sampled->points, 1536U, 384U, 1U);
    auto vae_condition = detail::encode_skin_condition(*impl_->skin_vae_weights, impl_->backend->value,
        sampled->points, sampled->normals, vae_queries);
    if (!vae_condition) return std::unexpected(vae_condition.error());
    dump_vae_trace_if_requested(*sampled, vae_queries, *vae_condition);
    auto generated = detail::generate_rig_tokens(*impl_->tokenrig_weights, impl_->backend->value,
        *mesh_condition, options);
    if (!generated) return std::unexpected(generated.error());
    auto model_target = detail::detokenize_generated_skeleton(generated->skeleton_tokens, center, scale);
    if (!model_target) return std::unexpected(model_target.error());
    if (model_target->names.size() != generated->joint_count)
        return std::unexpected(detail::fail(error_code::compute, "generated skeleton parser disagrees with TokenRig"));
    auto target = from_model_space(*model_target);
    return decode_binding(*impl_->skin_vae_weights, impl_->backend->value, target,
        generated->skin_codes, *vae_condition, sampled->points, sampled->normals, normalized);
}

result<skin> model::bind(const mesh & source, const skeleton & target, const generation_options & options) const {
    auto valid_mesh = validate_mesh(source);
    if (!valid_mesh) return std::unexpected(valid_mesh.error());
    auto valid_rig = validate_skeleton(target);
    if (!valid_rig) return std::unexpected(valid_rig.error());
    mesh model_source = source;
    model_source.vertices = to_model_space(source.vertices);
    model_source.normals = to_model_space(vertex_normals(source));
    const auto model_target = to_model_space(target);
    auto prefix = detail::tokenize_skeleton_prefix(model_source, model_target);
    if (!prefix) return std::unexpected(prefix.error());
    if (options.geometric_only) return geometric_binding(source, target);
    auto normals = model_source.normals;
    auto sampled = sample_mesh_surface(source, prefix->normalized_vertices, normals, options.seed);
    if (!sampled) return std::unexpected(sampled.error());
    auto mesh_queries = sampled_farthest_points(sampled->points, 2048U, 512U, 0U);
    dump_mesh_input_if_requested(*sampled, mesh_queries);
    auto mesh_condition = detail::encode_mesh(*impl_->mesh_weights, impl_->backend->value,
        sampled->points, sampled->normals, mesh_queries);
    if (!mesh_condition) return std::unexpected(mesh_condition.error());
    dump_mesh_condition_if_requested(*mesh_condition);
    auto vae_queries = sampled_farthest_points(sampled->points, 1536U, 384U, 1U);
    auto vae_condition = detail::encode_skin_condition(*impl_->skin_vae_weights, impl_->backend->value,
        sampled->points, sampled->normals, vae_queries);
    if (!vae_condition) return std::unexpected(vae_condition.error());
    dump_vae_trace_if_requested(*sampled, vae_queries, *vae_condition);
    auto codes = detail::generate_skin_codes(*impl_->tokenrig_weights, impl_->backend->value,
        *mesh_condition, prefix->tokens, target.names.size(), options);
    if (!codes) return std::unexpected(codes.error());
    dump_token_trace_if_requested(prefix->tokens, *codes);
    return decode_binding(*impl_->skin_vae_weights, impl_->backend->value, target,
        *codes, *vae_condition, sampled->points, sampled->normals, prefix->normalized_vertices);
}

std::string_view model::backend_name() const noexcept {
    return impl_ && impl_->backend ? std::string_view{impl_->backend->name} : std::string_view{};
}

result<motion> fit_motion_to_mesh(const mesh & geometry, const motion & animation) {
    auto valid_mesh = validate_mesh(geometry);
    if (!valid_mesh) return std::unexpected(valid_mesh.error());
    auto valid_rig = validate_skeleton(animation.rig);
    if (!valid_rig) return std::unexpected(valid_rig.error());
    if (animation.frames == 0U || animation.root_translations.size() != animation.frames ||
        animation.local_rotations.size() != animation.frames * animation.rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "motion arrays are incomplete"));
    vec3 mesh_low = geometry.vertices.front(), mesh_high = mesh_low;
    vec3 rig_low = animation.rig.rest_positions.front(), rig_high = rig_low;
    const auto include = [](vec3 value, vec3 & low, vec3 & high) {
        low.x = std::min(low.x, value.x); low.y = std::min(low.y, value.y); low.z = std::min(low.z, value.z);
        high.x = std::max(high.x, value.x); high.y = std::max(high.y, value.y); high.z = std::max(high.z, value.z);
    };
    for (const auto & value : geometry.vertices) include(value, mesh_low, mesh_high);
    for (const auto & value : animation.rig.rest_positions) include(value, rig_low, rig_high);
    const float rig_height = rig_high.y - rig_low.y;
    const float mesh_height = mesh_high.y - mesh_low.y;
    if (rig_height <= 1e-8F || mesh_height <= 1e-8F)
        return std::unexpected(detail::fail(error_code::invalid_argument, "cannot fit degenerate mesh or skeleton bounds"));
    const float scale = mesh_height / rig_height;
    const vec3 mesh_center{(mesh_low.x + mesh_high.x) * 0.5F, 0.0F,
                           (mesh_low.z + mesh_high.z) * 0.5F};
    const vec3 rig_center{(rig_low.x + rig_high.x) * 0.5F, 0.0F,
                          (rig_low.z + rig_high.z) * 0.5F};
    const vec3 offset{mesh_center.x - rig_center.x * scale,
                      mesh_low.y - rig_low.y * scale,
                      mesh_center.z - rig_center.z * scale};
    const auto transform = [&](vec3 value) {
        return vec3{value.x * scale + offset.x, value.y * scale + offset.y,
                    value.z * scale + offset.z};
    };
    motion output = animation;
    for (auto & value : output.rig.rest_positions) value = transform(value);
    // Kimodo's SMPL-X22 bind skeleton is a T-pose, while image-generated
    // Trellis characters commonly arrive with their arms lowered. A supplied
    // SkinTokens skeleton is expected to already lie inside its mesh, so place
    // the two arm chains into a conservative relaxed pose before conditioning.
    // Canonical names cover SMPL-X and SOMA. G1's neutral skeleton already
    // has a lowered articulated arm chain and retains the uniform fit above.
    const auto joint = [&](std::string_view name) -> std::optional<std::size_t> {
        const auto found = std::find(output.rig.names.begin(), output.rig.names.end(), name);
        return found == output.rig.names.end() ? std::nullopt :
            std::optional<std::size_t>{static_cast<std::size_t>(found - output.rig.names.begin())};
    };
    const float mesh_width = mesh_high.x - mesh_low.x;
    const float mesh_depth_center = (mesh_low.z + mesh_high.z) * 0.5F;
    const auto lower_arm = [&](std::string_view shoulder_name, std::string_view elbow_name,
                               std::string_view wrist_name, float side) {
        const auto shoulder = joint(shoulder_name), elbow = joint(elbow_name), wrist = joint(wrist_name);
        if (!shoulder || !elbow || !wrist) return;
        const auto anchor = output.rig.rest_positions[*shoulder];
        output.rig.rest_positions[*elbow] = {
            mesh_center.x + side * mesh_width * 0.38F,
            anchor.y - mesh_height * 0.22F, mesh_depth_center};
        const auto old_wrist = output.rig.rest_positions[*wrist];
        const vec3 new_wrist{
            mesh_center.x + side * mesh_width * 0.43F,
            anchor.y - mesh_height * 0.43F, mesh_depth_center};
        output.rig.rest_positions[*wrist] = new_wrist;
        // SOMA includes finger-tip descendants. Keep their local offsets when
        // moving the wrist instead of stretching those terminal bones back to
        // their original absolute locations.
        const vec3 delta{new_wrist.x - old_wrist.x, new_wrist.y - old_wrist.y,
                         new_wrist.z - old_wrist.z};
        for (std::size_t candidate = *wrist + 1U; candidate < output.rig.parents.size(); ++candidate) {
            auto parent = output.rig.parents[candidate];
            while (parent >= 0 && static_cast<std::size_t>(parent) != *wrist)
                parent = output.rig.parents[static_cast<std::size_t>(parent)];
            if (parent >= 0) {
                auto & value = output.rig.rest_positions[candidate];
                value = {value.x + delta.x, value.y + delta.y, value.z + delta.z};
            }
        }
    };
    lower_arm("left_shoulder", "left_elbow", "left_wrist", 1.0F);
    lower_arm("right_shoulder", "right_elbow", "right_wrist", -1.0F);
    lower_arm("LeftArm", "LeftForeArm", "LeftHand", 1.0F);
    lower_arm("RightArm", "RightForeArm", "RightHand", -1.0F);
    // The imported mesh has no humanoid bind-pose metadata: its static pose is
    // the only pose to which the predicted weights can be bound.  Kimodo's
    // quaternions, however, are absolute transforms from the SMPL-X T-pose.
    // Convert them into hierarchy-correct global deltas from frame zero, then
    // back into local tracks.  This makes the first animation frame identical
    // to the fitted mesh bind pose and prevents an immediate T-pose-space warp.
    const std::size_t joint_count = output.rig.names.size();
    std::vector<quat> bind_global(joint_count), frame_global(joint_count), delta_global(joint_count);
    for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
        const auto parent = output.rig.parents[joint_index];
        const quat local = output.local_rotations[joint_index];
        bind_global[joint_index] = normalized(parent < 0 ? local :
            multiply(bind_global[static_cast<std::size_t>(parent)], local));
    }
    for (std::size_t frame = 0; frame < output.frames; ++frame) {
        for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
            const auto parent = output.rig.parents[joint_index];
            const quat local = output.local_rotations[frame*joint_count+joint_index];
            frame_global[joint_index] = normalized(parent < 0 ? local :
                multiply(frame_global[static_cast<std::size_t>(parent)], local));
            delta_global[joint_index] = normalized(multiply(frame_global[joint_index],
                                                            conjugate(bind_global[joint_index])));
        }
        for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index) {
            const auto parent = output.rig.parents[joint_index];
            output.local_rotations[frame*joint_count+joint_index] = normalized(parent < 0 ?
                delta_global[joint_index] : multiply(conjugate(delta_global[static_cast<std::size_t>(parent)]),
                                                     delta_global[joint_index]));
        }
    }
    const vec3 first_root = animation.root_translations.front();
    const vec3 fitted_root = output.rig.rest_positions.front();
    for (auto & value : output.root_translations) {
        value = {fitted_root.x + (value.x - first_root.x) * scale,
                 fitted_root.y + (value.y - first_root.y) * scale,
                 fitted_root.z + (value.z - first_root.z) * scale};
    }
    return output;
}

result<motion> retarget_motion_to_rig(const motion & animation, const skeleton & target) {
    auto valid_source = validate_skeleton(animation.rig);
    if (!valid_source) return std::unexpected(valid_source.error());
    auto valid_target = validate_skeleton(target);
    if (!valid_target) return std::unexpected(valid_target.error());
    if (animation.frames == 0U || animation.root_translations.size() != animation.frames ||
        animation.local_rotations.size() != animation.frames * animation.rig.names.size())
        return std::unexpected(detail::fail(error_code::invalid_argument, "motion arrays are incomplete"));
    const auto bounds = [](std::span<const vec3> points) {
        std::pair<vec3, vec3> output{points.front(), points.front()};
        for (const auto value : points) {
            output.first.x=std::min(output.first.x,value.x); output.first.y=std::min(output.first.y,value.y); output.first.z=std::min(output.first.z,value.z);
            output.second.x=std::max(output.second.x,value.x); output.second.y=std::max(output.second.y,value.y); output.second.z=std::max(output.second.z,value.z);
        }
        return output;
    };
    const auto source_bounds = bounds(animation.rig.rest_positions), target_bounds = bounds(target.rest_positions);
    const auto center_scale = [](const auto & value) {
        const vec3 center{(value.first.x+value.second.x)*0.5F,(value.first.y+value.second.y)*0.5F,
                          (value.first.z+value.second.z)*0.5F};
        const float scale=std::max({value.second.x-value.first.x,value.second.y-value.first.y,value.second.z-value.first.z});
        return std::pair{center, scale};
    };
    const auto [source_center, source_scale] = center_scale(source_bounds);
    const auto [target_center, target_scale] = center_scale(target_bounds);
    if (source_scale <= 1e-8F || target_scale <= 1e-8F)
        return std::unexpected(detail::fail(error_code::invalid_argument, "cannot retarget a degenerate skeleton"));
    const auto normalize = [](vec3 value, vec3 center, float scale) {
        return vec3{(value.x-center.x)/scale,(value.y-center.y)/scale,(value.z-center.z)/scale};
    };
    std::vector<vec3> source_position, target_position;
    for (const auto value : animation.rig.rest_positions) source_position.push_back(normalize(value,source_center,source_scale));
    for (const auto value : target.rest_positions) target_position.push_back(normalize(value,target_center,target_scale));
    const auto is_descendant = [&](std::size_t candidate, std::int32_t ancestor) {
        if (ancestor < 0) return true;
        std::int32_t cursor=static_cast<std::int32_t>(candidate);
        while (cursor >= 0) { if (cursor == ancestor) return true; cursor=animation.rig.parents[static_cast<std::size_t>(cursor)]; }
        return false;
    };
    std::vector<std::size_t> mapping(target.names.size(), 0U);
    for (std::size_t joint=0; joint<target.names.size(); ++joint) {
        if (target.parents[joint] < 0) { mapping[joint]=0U; continue; }
        const auto mapped_parent=mapping[static_cast<std::size_t>(target.parents[joint])];
        float best=std::numeric_limits<float>::infinity(); std::size_t best_index=mapped_parent;
        for (std::size_t candidate=0; candidate<source_position.size(); ++candidate) {
            if (!is_descendant(candidate, static_cast<std::int32_t>(mapped_parent))) continue;
            const float x=source_position[candidate].x-target_position[joint].x;
            const float y=source_position[candidate].y-target_position[joint].y;
            const float z=source_position[candidate].z-target_position[joint].z;
            float score=x*x+y*y+z*z;
            if ((source_position[candidate].x < -0.03F) != (target_position[joint].x < -0.03F) &&
                std::abs(target_position[joint].x)>0.08F) score += 1.0F;
            if (score < best) { best=score; best_index=candidate; }
        }
        mapping[joint]=best_index;
    }
    motion output;
    output.frames=animation.frames; output.frames_per_second=animation.frames_per_second; output.rig=target;
    output.local_rotations.resize(output.frames*target.names.size());
    for (std::size_t frame=0; frame<output.frames; ++frame)
        for (std::size_t joint=0; joint<target.names.size(); ++joint)
            output.local_rotations[frame*target.names.size()+joint]=
                animation.local_rotations[frame*animation.rig.names.size()+mapping[joint]];
    output.root_translations.resize(output.frames);
    const vec3 source_first=animation.root_translations.front(), target_root=target.rest_positions.front();
    const float travel_scale=target_scale/source_scale;
    for (std::size_t frame=0; frame<output.frames; ++frame) {
        const auto value=animation.root_translations[frame];
        output.root_translations[frame]={target_root.x+(value.x-source_first.x)*travel_scale,
                                         target_root.y+(value.y-source_first.y)*travel_scale,
                                         target_root.z+(value.z-source_first.z)*travel_scale};
    }
    return output;
}

} // namespace skintokens
