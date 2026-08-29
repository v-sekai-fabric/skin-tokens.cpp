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

struct sampled_cloud {
    std::vector<vec3> points;
    std::vector<vec3> normals;
    std::vector<std::uint32_t> faces;
    std::vector<std::array<double, 2>> barycentric;
};

// NumPy's legacy RandomState is used by the released preprocessing path.  It
// is not interchangeable with the C++ standard distributions: in particular,
// sample_surface draws every face first and only then draws all barycentric
// coordinates.  Keeping this tiny compatible lane also makes captured seeds
// meaningful across the Python and C++ implementations.
class numpy_random_state {
public:
    explicit numpy_random_state(std::uint32_t seed) : engine_(seed) {}

    double random_double() {
        const std::uint64_t a = static_cast<std::uint64_t>(engine_() >> 5U);
        const std::uint64_t b = static_cast<std::uint64_t>(engine_() >> 6U);
        return static_cast<double>(a * 67'108'864ULL + b) / 9'007'199'254'740'992.0;
    }

    void consume_permutation(std::size_t count) {
        if (count < 2U) return;
        for (std::size_t i = count - 1U; i != 0U; --i) (void) interval(i);
    }

private:
    std::size_t interval(std::size_t maximum) {
        std::uint32_t mask = static_cast<std::uint32_t>(maximum);
        mask |= mask >> 1U; mask |= mask >> 2U; mask |= mask >> 4U;
        mask |= mask >> 8U; mask |= mask >> 16U;
        std::uint32_t value;
        do value = static_cast<std::uint32_t>(engine_()) & mask; while (value > maximum);
        return value;
    }

    std::mt19937 engine_;
};

// NumPy 1.26 default_rng(0/1), used by both released farthest-point sampling
// paths.  PCG64 and Generator.choice follow NumPy's MIT/BSD-licensed reference
// algorithms; see NOTICE.  Only the two fixed inference seeds are accepted so
// the SeedSequence implementation does not become part of the runtime.
class numpy_pcg64 {
public:
    explicit numpy_pcg64(std::uint64_t seed) {
        if (seed == 0U) {
            state_ = make128(0x1aa1b5345996452dULL, 0x09585eb7a69561e3ULL);
            increment_ = make128(0x418ddadb3af71a82ULL, 0x588133bc447873a9ULL);
        } else if (seed == 1U) {
            state_ = make128(0x9c5b484bfedb756cULL, 0x2a6e7d6f320fbc7eULL);
            increment_ = make128(0x922af2da2645f895ULL, 0xa19857b95740937bULL);
        } else throw std::invalid_argument("unsupported fixed NumPy PCG64 seed");
    }

    std::vector<std::size_t> choice(std::size_t population, std::size_t count) {
        if (count > population) throw std::invalid_argument("NumPy choice exceeds population");
        std::vector<std::size_t> values(population);
        std::iota(values.begin(), values.end(), 0U);
        const std::size_t first = std::max(population - count, std::size_t{1});
        for (std::size_t i = population - 1U;; --i) {
            const std::size_t selected = bounded(i);
            std::swap(values[selected], values[i]);
            if (i == first) break;
        }
        return {values.end() - static_cast<std::ptrdiff_t>(count), values.end()};
    }

private:
#if defined(__SIZEOF_INT128__)
    __extension__ using uint128 = unsigned __int128;
#else
#error "The NumPy-compatible PCG64 inference sampler requires 128-bit integer support"
#endif

    static constexpr uint128 make128(std::uint64_t high, std::uint64_t low) {
        return (static_cast<uint128>(high) << 64U) | static_cast<uint128>(low);
    }

    std::uint64_t next64() {
        constexpr uint128 multiplier = make128(0x2360ed051fc65da4ULL, 0x4385df649fccf645ULL);
        state_ = state_ * multiplier + increment_;
        const auto high = static_cast<std::uint64_t>(state_ >> 64U);
        const auto low = static_cast<std::uint64_t>(state_);
        const auto value = high ^ low;
        const auto rotation = static_cast<unsigned>(high >> 58U);
        return (value >> rotation) | (value << ((0U - rotation) & 63U));
    }

    std::uint32_t next32() {
        if (has_uint32_) { has_uint32_ = false; return buffered_uint32_; }
        const auto value = next64();
        buffered_uint32_ = static_cast<std::uint32_t>(value >> 32U);
        has_uint32_ = true;
        return static_cast<std::uint32_t>(value);
    }

    std::size_t bounded(std::size_t inclusive_maximum) {
        const auto range = static_cast<std::uint32_t>(inclusive_maximum + 1U);
        std::uint64_t product = static_cast<std::uint64_t>(next32()) * range;
        std::uint32_t leftover = static_cast<std::uint32_t>(product);
        if (leftover < range) {
            const std::uint32_t threshold = static_cast<std::uint32_t>(0U - range) % range;
            while (leftover < threshold) {
                product = static_cast<std::uint64_t>(next32()) * range;
                leftover = static_cast<std::uint32_t>(product);
            }
        }
        return static_cast<std::size_t>(product >> 32U);
    }

    uint128 state_ = 0;
    uint128 increment_ = 0;
    bool has_uint32_ = false;
    std::uint32_t buffered_uint32_ = 0;
};

result<std::optional<std::vector<std::int32_t>>> forced_skin_codes(
    std::size_t joint_count) {
    const char * path = std::getenv("SKINTOKENS_FORCED_CODES");
    if (path == nullptr || *path == '\0') return std::optional<std::vector<std::int32_t>>{};
    const std::size_t expected = joint_count * 4U;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        return std::unexpected(detail::fail(error_code::io,
            "cannot open forced SkinVAE codes: " + std::string{path}));
    const auto bytes = input.tellg();
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) != expected * sizeof(std::int32_t))
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "forced SkinVAE code file must contain exactly four int32 values per joint"));
    input.seekg(0);
    std::vector<std::int32_t> codes(expected);
    input.read(reinterpret_cast<char *>(codes.data()), static_cast<std::streamsize>(expected * sizeof(std::int32_t)));
    if (!input)
        return std::unexpected(detail::fail(error_code::io, "cannot read forced SkinVAE codes"));
    for (const auto code : codes) {
        if (code < 0 || code > 32768)
            return std::unexpected(detail::fail(error_code::invalid_argument,
                "forced SkinVAE code is outside the checkpoint vocabulary"));
    }
    return std::optional<std::vector<std::int32_t>>{std::move(codes)};
}

result<sampled_cloud> sample_mesh_surface(const mesh & source, std::span<const vec3> normalized,
                                          std::span<const vec3> vertex_normal, std::uint64_t seed,
                                          std::span<const std::array<double, 3>> precise = {}) {
    constexpr std::size_t total_samples = 54000U;
    // The released predict config declares 16,384 vertex samples, but
    // SamplerMix.sample() does not forward that member to
    // sample_vertex_groups(). Match the executable upstream path: all 54K
    // samples are area-weighted surface points.
    constexpr std::size_t vertex_samples = 0U;
    (void) vertex_normal;
    if (!precise.empty() && precise.size() != normalized.size())
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "precise normalized positions must match the mesh"));
    const auto coordinate = [&](std::size_t vertex, std::size_t axis) {
        if (!precise.empty()) return precise[vertex][axis];
        const auto value = normalized[vertex];
        return static_cast<double>(axis == 0U ? value.x : axis == 1U ? value.y : value.z);
    };
    const auto blender_face = [](triangle face) {
        // BpyParser rebuilds each triangle from polygon.edge_keys, then picks
        // list(set(sorted(nodes)))[0] as its first corner.  For three small
        // non-negative integer keys, this is CPython's eight-slot small-set
        // insertion/iteration order.  Preserve winding and reproduce only the
        // resulting cyclic rotation.
        std::array<std::uint32_t, 3> sorted{face[0], face[1], face[2]};
        std::sort(sorted.begin(), sorted.end());
        std::array<std::optional<std::uint32_t>, 8> table{};
        for (const auto key : sorted) {
            std::uint64_t perturb = key;
            std::size_t slot = static_cast<std::size_t>(key & 7U);
            while (table[slot] && *table[slot] != key) {
                perturb >>= 5U;
                slot = (slot * 5U + 1U + static_cast<std::size_t>(perturb)) & 7U;
            }
            table[slot] = key;
        }
        const auto first = **std::find_if(table.begin(), table.end(), [](const auto & value) { return value.has_value(); });
        while (face[0] != first) std::rotate(face.begin(), face.begin() + 1, face.end());
        return face;
    };
    sampled_cloud output;
    output.points.reserve(total_samples); output.normals.reserve(total_samples);
    if (seed > std::numeric_limits<std::uint32_t>::max())
        return std::unexpected(detail::fail(error_code::invalid_argument,
            "sampling seed exceeds NumPy RandomState's supported range"));
    numpy_random_state random{static_cast<std::uint32_t>(seed)};
    // AugmentAffine consumes these two draws even though both inference-time
    // probabilities are zero, then sample_vertex_groups constructs a complete
    // permutation before selecting its configured zero direct vertices.
    (void) random.random_double();
    (void) random.random_double();
    random.consume_permutation(normalized.size());
    (void) vertex_samples;
    std::vector<double> areas(source.faces.size());
    std::vector<vec3> face_normals(source.faces.size());
    for (std::size_t i = 0; i < source.faces.size(); ++i) {
        const auto face = blender_face(source.faces[i]);
        const std::array<double, 3> left{
            coordinate(face[1], 0U) - coordinate(face[0], 0U),
            coordinate(face[1], 1U) - coordinate(face[0], 1U),
            coordinate(face[1], 2U) - coordinate(face[0], 2U)};
        const std::array<double, 3> right{
            coordinate(face[2], 0U) - coordinate(face[0], 0U),
            coordinate(face[2], 1U) - coordinate(face[0], 1U),
            coordinate(face[2], 2U) - coordinate(face[0], 2U)};
        const std::array<double, 3> value{
            left[1]*right[2] - left[2]*right[1],
            left[2]*right[0] - left[0]*right[2],
            left[0]*right[1] - left[1]*right[0]};
        const double length = std::sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2]);
        areas[i] = length;
        face_normals[i] = length > 1e-12 ? vec3{
            static_cast<float>(value[0]/length), static_cast<float>(value[1]/length),
            static_cast<float>(value[2]/length)} : vec3{0.0F, 1.0F, 0.0F};
    }
    std::partial_sum(areas.begin(), areas.end(), areas.begin());
    if (areas.back() <= 0.0)
        return std::unexpected(detail::fail(error_code::invalid_argument, "mesh has no non-degenerate surface"));
    std::vector<std::size_t> face_indices(total_samples);
    for (auto & face_index : face_indices) {
        const double pick = random.random_double() * areas.back();
        face_index = static_cast<std::size_t>(std::lower_bound(areas.begin(), areas.end(), pick) - areas.begin());
    }
    output.faces.reserve(total_samples);
    output.barycentric.reserve(total_samples);
    for (const auto face_index : face_indices) {
        const auto face = blender_face(source.faces[face_index]);
        double u = random.random_double(), v = random.random_double();
        if (u + v > 1.0) { u = 1.0 - u; v = 1.0 - v; }
        output.faces.push_back(static_cast<std::uint32_t>(face_index));
        output.barycentric.push_back({u, v});
        vec3 point{};
        float * output_axes[]{&point.x, &point.y, &point.z};
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const double a = coordinate(face[0], axis);
            const double b = coordinate(face[1], axis);
            const double c = coordinate(face[2], axis);
            *output_axes[axis] = static_cast<float>(a + (b-a)*u + (c-a)*v);
        }
        output.points.push_back(point);
        output.normals.push_back(face_normals[face_index]);
    }
    return output;
}

std::vector<std::int32_t> sampled_farthest_points(std::span<const vec3> points,
                                                   std::size_t candidates, std::size_t wanted,
                                                   std::uint64_t seed) {
    numpy_pcg64 random{seed};
    auto order = random.choice(points.size(), std::min(candidates, points.size()));
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
    std::span<const float> interpolation_weights,
    std::span<const float> surface_weights,
    std::span<const float> final_dense_weights) {
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
    if (!surface_weights.empty())
        write(base.string() + ".surface.f32", surface_weights.data(), surface_weights.size_bytes());
    if (!final_dense_weights.empty())
        write(base.string() + ".final-dense.f32", final_dense_weights.data(), final_dense_weights.size_bytes());
}

result<skin> decode_binding(const detail::weight_component & weights, ggml_backend_t backend,
                            const skeleton & target,
                            std::span<const std::int32_t> codes,
                            std::span<const float> vae_condition,
                            std::span<const vec3> sampled_points,
                            std::span<const vec3> sampled_normals,
                            std::span<const vec3> normalized_vertices,
                            std::span<const triangle> faces,
                            std::span<const vec3> normalized_joints,
                            bool surface_postprocess) {
    detail::profile_scope total_profile{"binding.decode.total"};
    if (codes.size() != target.names.size() * 4U)
        return std::unexpected(detail::fail(error_code::compute, "skin-code count does not match generated skeleton"));
    std::vector<std::vector<float>> dense(target.names.size());
    {
        detail::profile_scope decode_profile{"binding.decode.skin_vae"};
        for (std::size_t joint = 0; joint < target.names.size(); ++joint) {
            auto decoded = detail::decode_skin_joint(weights, backend,
                std::span{codes.data() + joint * 4U, 4U}, vae_condition,
                sampled_points, sampled_normals);
            if (!decoded) return std::unexpected(decoded.error());
            dense[joint] = std::move(*decoded);
        }
    }
    // Decode on the 54K cloud and interpolate to source vertices. The normal
    // upstream export retains those raw learned weights; voxel_skin is an
    // explicit optional heuristic before top-four selection.
    const bool trace_binding = [] {
        const char * value = std::getenv("SKINTOKENS_DUMP_BINDING_TRACE_PREFIX");
        return value != nullptr && *value != '\0';
    }();
    detail::binding_trace trace;
    auto output = detail::profiled("binding.decode.cpu_integration", [&] {
        return surface_postprocess ?
            detail::integrate_postprocessed_binding(target, normalized_vertices, faces,
                normalized_joints, sampled_points, dense, trace_binding ? &trace : nullptr) :
            detail::integrate_learned_binding(target, normalized_vertices,
                sampled_points, dense, trace_binding ? &trace : nullptr);
    });
    dump_binding_trace_if_requested(dense, output, normalized_vertices,
                                    trace.neighbor_indices, trace.interpolation_weights,
                                    trace.surface_weights, trace.final_dense_weights);
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
                                  std::span<const triangle> topology,
                                  std::span<const std::int32_t> queries) {
    const char * prefix = std::getenv("SKINTOKENS_DUMP_MESH_INPUT_PREFIX");
    if (prefix == nullptr || *prefix == '\0') return;
    const std::filesystem::path base{prefix};
    dump_binary(base.string() + ".points.f32", std::span{sampled.points});
    dump_binary(base.string() + ".normals.f32", std::span{sampled.normals});
    dump_binary(base.string() + ".faces.u32", std::span{sampled.faces});
    dump_binary(base.string() + ".barycentric.f64", std::span{sampled.barycentric});
    dump_binary(base.string() + ".topology.u32", topology);
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

void dump_token_prefix_if_requested(std::span<const std::int32_t> prefix) {
    const char * path = std::getenv("SKINTOKENS_DUMP_TOKEN_TRACE_PREFIX");
    if (path == nullptr || *path == '\0') return;
    dump_binary(std::string{path} + ".prefix.i32", prefix);
}

} // namespace

model::model(std::unique_ptr<impl> value) noexcept : impl_(std::move(value)) {}
model::model(model &&) noexcept = default;
model & model::operator=(model &&) noexcept = default;
model::~model() = default;

result<model> model::load(const std::filesystem::path & bundle, const runtime_options & options) {
    detail::profile_scope total_profile{"model.load.total"};
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
    detail::profile_scope total_profile{"model.rig.total"};
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
    auto sampled = detail::profiled("model.rig.sample_surface", [&] {
        return sample_mesh_surface(source, normalized, normals, options.seed);
    });
    if (!sampled) return std::unexpected(sampled.error());
    auto mesh_queries = detail::profiled("model.rig.mesh_fps", [&] {
        return sampled_farthest_points(sampled->points, 2048U, 512U, 0U);
    });
    dump_mesh_input_if_requested(*sampled, source.faces, mesh_queries);
    auto mesh_condition = detail::profiled("model.rig.mesh_encoder", [&] {
        return detail::encode_mesh(*impl_->mesh_weights, impl_->backend->value,
            sampled->points, sampled->normals, mesh_queries);
    });
    if (!mesh_condition) return std::unexpected(mesh_condition.error());
    dump_mesh_condition_if_requested(*mesh_condition);
    auto vae_queries = detail::profiled("model.rig.vae_fps", [&] {
        return sampled_farthest_points(sampled->points, 1536U, 384U, 1U);
    });
    auto vae_condition = detail::profiled("model.rig.vae_encoder", [&] {
        return detail::encode_skin_condition(*impl_->skin_vae_weights, impl_->backend->value,
            sampled->points, sampled->normals, vae_queries);
    });
    if (!vae_condition) return std::unexpected(vae_condition.error());
    dump_vae_trace_if_requested(*sampled, vae_queries, *vae_condition);
    auto generated = detail::profiled("model.rig.token_generation", [&] {
        return detail::generate_rig_tokens(*impl_->tokenrig_weights, impl_->backend->value,
            *mesh_condition, options);
    });
    if (!generated) return std::unexpected(generated.error());
    auto model_target = detail::detokenize_generated_skeleton(generated->skeleton_tokens, center, scale);
    if (!model_target) return std::unexpected(model_target.error());
    if (model_target->names.size() != generated->joint_count)
        return std::unexpected(detail::fail(error_code::compute, "generated skeleton parser disagrees with TokenRig"));
    auto target = from_model_space(*model_target);
    std::vector<vec3> normalized_joints;
    normalized_joints.reserve(model_target->rest_positions.size());
    for (const auto value : model_target->rest_positions)
        normalized_joints.push_back({(value.x-center.x)/scale, (value.y-center.y)/scale,
                                     (value.z-center.z)/scale});
    return decode_binding(*impl_->skin_vae_weights, impl_->backend->value, target,
        generated->skin_codes, *vae_condition, sampled->points, sampled->normals, normalized,
        source.faces, normalized_joints, options.surface_postprocess);
}

result<skin> model::bind(const mesh & source, const skeleton & target, const generation_options & options) const {
    detail::profile_scope total_profile{"model.bind.total"};
    auto valid_mesh = validate_mesh(source);
    if (!valid_mesh) return std::unexpected(valid_mesh.error());
    auto valid_rig = validate_skeleton(target);
    if (!valid_rig) return std::unexpected(valid_rig.error());
    mesh model_source = source;
    model_source.vertices = to_model_space(source.vertices);
    model_source.normals = to_model_space(vertex_normals(source));
    const auto model_target = to_model_space(target);
    auto prefix = detail::profiled("model.bind.tokenize_skeleton", [&] {
        return detail::tokenize_skeleton_prefix(model_source, model_target);
    });
    if (!prefix) return std::unexpected(prefix.error());
    dump_token_prefix_if_requested(prefix->tokens);
    if (options.geometric_only) return geometric_binding(source, target);
    auto normals = model_source.normals;
    auto sampled = detail::profiled("model.bind.sample_surface", [&] {
        return sample_mesh_surface(source, prefix->normalized_vertices, normals, options.seed,
            prefix->precise_normalized_vertices);
    });
    if (!sampled) return std::unexpected(sampled.error());
    auto mesh_queries = detail::profiled("model.bind.mesh_fps", [&] {
        return sampled_farthest_points(sampled->points, 2048U, 512U, 0U);
    });
    dump_mesh_input_if_requested(*sampled, source.faces, mesh_queries);
    auto mesh_condition = detail::profiled("model.bind.mesh_encoder", [&] {
        return detail::encode_mesh(*impl_->mesh_weights, impl_->backend->value,
            sampled->points, sampled->normals, mesh_queries);
    });
    if (!mesh_condition) return std::unexpected(mesh_condition.error());
    dump_mesh_condition_if_requested(*mesh_condition);
    auto vae_queries = detail::profiled("model.bind.vae_fps", [&] {
        return sampled_farthest_points(sampled->points, 1536U, 384U, 1U);
    });
    auto vae_condition = detail::profiled("model.bind.vae_encoder", [&] {
        return detail::encode_skin_condition(*impl_->skin_vae_weights, impl_->backend->value,
            sampled->points, sampled->normals, vae_queries);
    });
    if (!vae_condition) return std::unexpected(vae_condition.error());
    dump_vae_trace_if_requested(*sampled, vae_queries, *vae_condition);
    auto forced = forced_skin_codes(target.names.size());
    if (!forced) return std::unexpected(forced.error());
    std::vector<std::int32_t> codes;
    if (*forced) codes = std::move(**forced);
    else {
        auto generated = detail::profiled("model.bind.token_generation", [&] {
            return detail::generate_skin_codes(*impl_->tokenrig_weights, impl_->backend->value,
                *mesh_condition, prefix->tokens, target.names.size(), options);
        });
        if (!generated) return std::unexpected(generated.error());
        codes = std::move(*generated);
    }
    dump_token_trace_if_requested(prefix->tokens, codes);
    return decode_binding(*impl_->skin_vae_weights, impl_->backend->value, target,
        codes, *vae_condition, sampled->points, sampled->normals, prefix->normalized_vertices,
        source.faces, prefix->normalized_joints, options.surface_postprocess);
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

} // namespace skintokens
