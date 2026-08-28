#!/usr/bin/env python3
"""Replace an animation fixture's expected binding with an upstream SkinVAE run.

The C++ binding run must be made with SKINTOKENS_DUMP_VAE_TRACE_PREFIX and
SKINTOKENS_DUMP_TOKEN_TRACE_PREFIX. This script consumes those exact sampled
points, condition query indices and generated FSQ codes. It independently evaluates
the released safetensors SkinVAE, reproduces upstream's eight-neighbour
interpolation and learned-weight top-four export, then writes reference dense
and sparse influences plus final animated vertices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in rotation)
    length = x*x + y*y + z*z + w*w
    scale = 2.0 / length if length > 0 else 0.0
    output = np.array([
        [1-(y*y+z*z)*scale, (x*y-z*w)*scale, (x*z+y*w)*scale, translation[0]],
        [(x*y+z*w)*scale, 1-(x*x+z*z)*scale, (y*z-x*w)*scale, translation[1]],
        [(x*z-y*w)*scale, (y*z+x*w)*scale, 1-(x*x+y*y)*scale, translation[2]],
        [0, 0, 0, 1]], dtype=np.float64)
    return output


def animated_vertices(positions: np.ndarray, joints: np.ndarray, weights: np.ndarray,
                      parents: np.ndarray, rest: np.ndarray, roots: np.ndarray,
                      rotations: np.ndarray, frames: list[int]) -> np.ndarray:
    homogeneous = np.concatenate((positions.astype(np.float64), np.ones((len(positions), 1))), axis=1)
    output = np.zeros((len(frames), len(positions), 3), dtype="<f4")
    for selected, frame in enumerate(frames):
        globals_ = np.zeros((len(rest), 4, 4), dtype=np.float64)
        for joint, parent in enumerate(parents):
            translation = roots[frame] if parent < 0 else rest[joint] - rest[parent]
            local = matrix(rotations[frame, joint], translation)
            globals_[joint] = local if parent < 0 else globals_[parent] @ local
        result = np.zeros((len(positions), 3), dtype=np.float64)
        for slot in range(4):
            relative = positions.astype(np.float64) - rest[joints[:, slot]]
            points = np.concatenate((relative, np.ones((len(relative), 1))), axis=1)
            transformed = np.einsum("nij,nj->ni", globals_[joints[:, slot]], points)[:, :3]
            result += transformed * weights[:, slot, None]
        result /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-30)
        output[selected] = result.astype("<f4")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True,
                        help="verified TokenRig safetensors containing its frozen vae.model tensors")
    parser.add_argument("--fixture", type=Path, required=True,
                        help="fixture made by capture_animated_vertices.py")
    parser.add_argument("--vae-trace-prefix", type=Path, required=True)
    parser.add_argument("--token-trace-prefix", type=Path, required=True)
    parser.add_argument("--cpp-binding-trace-prefix", type=Path,
                        help="optional C++ dense/final binding trace for decoder-stage metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpp-backend", choices=("cpu", "vulkan"), default="vulkan",
                        help="backend that produced the C++ trace and animation GLB")
    parser.add_argument("--query-chunk", type=int, default=8192)
    parser.add_argument("--nearest-chunk", type=int, default=256)
    args = parser.parse_args()

    manifest_path = args.fixture / "animation-vertices.json"
    manifest = json.loads(manifest_path.read_text())
    vertices = manifest["vertex_count"]
    joints_count = manifest["joint_count"]
    frames_count = manifest["frame_count"]
    positions = np.fromfile(args.fixture / "positions.f32", "<f4").reshape(vertices, 3)
    np.fromfile(args.fixture / "joints.u16", "<u2").reshape(vertices, 4)
    np.fromfile(args.fixture / "weights.f32", "<f4").reshape(vertices, 4)
    parents = np.fromfile(args.fixture / "parents.i32", "<i4")
    rest = np.fromfile(args.fixture / "rest-positions.f32", "<f4").reshape(joints_count, 3)
    roots = np.fromfile(args.fixture / "root-translations.f32", "<f4").reshape(frames_count, 3)
    rotations = np.fromfile(args.fixture / "local-rotations.f32", "<f4").reshape(
        frames_count, joints_count, 4)
    if len(parents) != joints_count:
        raise ValueError("fixture skeleton is truncated")

    trace = str(args.vae_trace_prefix)
    sampled_points = np.fromfile(trace + ".points.f32", "<f4").reshape(-1, 3)
    sampled_normals = np.fromfile(trace + ".normals.f32", "<f4").reshape(-1, 3)
    cpp_condition = np.fromfile(trace + ".condition.f32", "<f4").reshape(-1, 512)
    condition_queries = np.fromfile(trace + ".queries.i32", "<i4").astype(np.int64)
    codes = np.fromfile(str(args.token_trace_prefix) + ".codes.i32", "<i4")
    if sampled_points.shape != sampled_normals.shape or len(sampled_points) != 54000:
        raise ValueError("expected the runtime's 54,000-point SkinVAE trace")
    if len(codes) != joints_count * 4:
        raise ValueError("skin code count does not match fixture skeleton")

    device = torch.device(args.device)
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        learned = {name: source.get_tensor(name).float().to(device)
                   for name in source.keys() if name.startswith("vae.model.")}

    def weight(name: str) -> torch.Tensor:
        return learned["vae.model." + name]

    def linear(value: torch.Tensor, stem: str) -> torch.Tensor:
        return F.linear(value, weight(stem + ".weight"), learned.get("vae.model." + stem + ".bias"))

    def norm(value: torch.Tensor, stem: str) -> torch.Tensor:
        return F.layer_norm(value, (value.shape[-1],), weight(stem + ".weight"),
                            weight(stem + ".bias"), 1e-5)

    def embedding(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        # Match FrequencyPositionalEmbedding.__init__ exactly. In particular,
        # upstream fills a CPU F32 phase buffer element by element from an I64
        # base tensor before moving the module to CUDA.
        frequency = (2.0 ** torch.arange(8, dtype=torch.float32)) * torch.pi
        phase = torch.arange(8, dtype=torch.float32)
        for index in range(8):
            step = (index + 1) / 8.0
            phase[index] = torch.pow(torch.tensor(8), 1.0 - step) + step
        phase *= torch.pi * 2.0
        frequency, phase = frequency.to(device), phase.to(device)
        angle = (points[..., None] * frequency).reshape(points.shape[0], -1)
        shifted = (points[..., None] * torch.pi * .5 + phase).reshape(points.shape[0], -1)
        return torch.cat((points, angle.sin() + shifted.sin(), angle.cos() + shifted.cos(), normals), -1)

    def attention(query_input: torch.Tensor, data_input: torch.Tensor, stem: str,
                  cross: bool) -> torch.Tensor:
        query = linear(query_input, stem + ".to_q")
        data = norm(data_input, stem + ".norm_cross") if cross else data_input
        key, value = linear(data, stem + ".to_k"), linear(data, stem + ".to_v")
        if cross:
            key, value = torch.cat((key, value), dim=-1).reshape(
                key.shape[0], 12, 128).split(64, dim=-1)
            query = query.reshape(query.shape[0], 12, 64)
        else:
            query, key, value = torch.cat((query, key, value), dim=-1).reshape(
                query.shape[0], 12, 192).split(64, dim=-1)
        def heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(1, item.shape[0], 12, 64).transpose(1, 2)
        result = F.scaled_dot_product_attention(heads(query), heads(key), heads(value))
        return linear(result.transpose(1, 2).reshape(query.shape[0], 768), stem + ".to_out.0")

    def block(state: torch.Tensor, stem: str, data: torch.Tensor | None = None) -> torch.Tensor:
        if data is None:
            value = norm(state, stem + ".norm1")
            state = state + attention(value, value, stem + ".attn1", False)
        else:
            state = state + attention(norm(state, stem + ".norm2"), data, stem + ".attn2", True)
        return state + linear(F.gelu(linear(norm(state, stem + ".norm3"), stem + ".ff.net.0.proj")),
                              stem + ".ff.net.2")

    point_tensor = torch.from_numpy(sampled_points).to(device)
    normal_tensor = torch.from_numpy(sampled_normals).to(device)
    dense = np.empty((len(sampled_points), joints_count), dtype="<f4")
    condition_stages: dict[str, np.ndarray] = {}
    basis = torch.tensor([1, 8, 64, 512, 4096], device=device, dtype=torch.int64)
    with torch.inference_mode():
        embedded_condition = embedding(point_tensor, normal_tensor)
        condition_stages["cond-embedding-head"] = embedded_condition[:64].float().cpu().numpy()
        projected_condition = linear(embedded_condition, "cond_encoder.proj_in")
        condition_stages["cond-projected-head"] = projected_condition[:64].float().cpu().numpy()
        condition_tensor = projected_condition[torch.from_numpy(condition_queries).to(device)]
        condition_stages["cond-selected"] = condition_tensor.float().cpu().numpy()
        condition_tensor = block(condition_tensor, "cond_encoder.blocks.0", projected_condition)
        condition_stages["cond-cross"] = condition_tensor.float().cpu().numpy()
        for layer in range(2):
            condition_tensor = block(condition_tensor, f"cond_encoder.blocks.{layer + 1}")
            condition_stages[f"cond-self-{layer + 1}"] = condition_tensor.float().cpu().numpy()
        condition_tensor = norm(condition_tensor, "cond_encoder.norm_out")
        condition_stages["cond-norm"] = condition_tensor.float().cpu().numpy()
        condition_tensor = linear(condition_tensor, "cond_quant")
        reference_condition = condition_tensor.float().cpu().numpy().astype("<f4")
        condition_difference = cpp_condition.astype(np.float64) - reference_condition.astype(np.float64)
        manifest["condition_max_abs"] = float(np.max(np.abs(condition_difference)))
        manifest["condition_relative_l2"] = float(
            np.linalg.norm(condition_difference.ravel()) /
            max(np.linalg.norm(reference_condition.astype(np.float64).ravel()), 1e-30))
        for joint in range(joints_count):
            indices = torch.from_numpy(codes[joint*4:(joint+1)*4].astype(np.int64)).to(device)
            fsq = (((indices[:, None] // basis) % 8).float() - 4.0) / 4.0
            decoded = linear(fsq, "FSQ.project_out")
            state = linear(torch.cat((decoded, condition_tensor), dim=0), "post_quant")
            for layer in range(10):
                state = block(state, f"decoder.blocks.{layer}")
            for begin in range(0, len(sampled_points), args.query_chunk):
                end = min(begin + args.query_chunk, len(sampled_points))
                query = linear(embedding(point_tensor[begin:end], normal_tensor[begin:end]),
                               "decoder.proj_query")
                query = block(query, "decoder.blocks.10", state)
                value = torch.sigmoid(linear(norm(query, "decoder.norm_out"), "decoder.proj_out"))
                dense[begin:end, joint] = value[:, 0].float().cpu().numpy()
            print(f"decoded joint {joint + 1}/{joints_count}", flush=True)

        # Rebuild tokenize_skeleton_prefix's normalized model-space vertices.
        model_vertices = positions[:, [0, 2, 1]].copy()
        model_vertices[:, 1] *= -1
        model_joints = rest[:, [0, 2, 1]].copy()
        model_joints[:, 1] *= -1
        low = np.minimum(model_vertices.min(axis=0), model_joints.min(axis=0))
        high = np.maximum(model_vertices.max(axis=0), model_joints.max(axis=0))
        center = (low + high) * .5
        scale = (high - low).max() * .5
        normalized = (model_vertices - center) / scale

        samples_gpu = torch.from_numpy(sampled_points).to(device)
        normalized_gpu = torch.from_numpy(normalized.astype("<f4")).to(device)
        nearest_indices = np.empty((vertices, 8), dtype=np.int64)
        nearest_distances = np.empty((vertices, 8), dtype=np.float32)
        for begin in range(0, vertices, args.nearest_chunk):
            end = min(begin + args.nearest_chunk, vertices)
            distances = torch.cdist(normalized_gpu[begin:end], samples_gpu)
            values, indices = torch.topk(distances, 8, largest=False, sorted=True)
            nearest_indices[begin:end] = indices.cpu().numpy()
            nearest_distances[begin:end] = values.cpu().numpy()

    interpolation = 1.0 / (nearest_distances + 1e-8)
    interpolation /= interpolation.sum(axis=1, keepdims=True)
    def integrate(dense_values: np.ndarray, indices: np.ndarray,
                  neighbor_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Asset.from_data() interpolates every learned joint channel from the
        # sampled cloud. The Blender exporter then chooses the greatest four
        # learned weights and normalizes only those four. Bone distance is not
        # part of either upstream operation.
        interpolated = np.einsum("vk,vkj->vj", neighbor_weights, dense_values[indices])
        top = np.argsort(-interpolated, axis=1, kind="stable")[:, :4]
        output_joints = top.astype("<u2")
        output_weights = np.take_along_axis(interpolated, top, axis=1).astype("<f4")
        sums = output_weights.sum(axis=1, keepdims=True)
        empty = sums[:, 0] <= 1e-12
        output_joints[empty, 0] = 0
        output_weights[empty, 0] = 1
        sums[empty, 0] = 1
        output_weights /= sums
        return interpolated.astype("<f4"), output_joints, output_weights

    reference_dense, reference_joints, reference_weights = integrate(
        dense, nearest_indices, interpolation)

    if args.cpp_binding_trace_prefix is not None:
        cpp_dense = np.fromfile(str(args.cpp_binding_trace_prefix) + ".dense.f32", "<f4").reshape(
            joints_count, len(sampled_points)).T
        difference = cpp_dense.astype(np.float64) - dense.astype(np.float64)
        manifest["decoder_dense_max_abs"] = float(np.max(np.abs(difference)))
        manifest["decoder_dense_relative_l2"] = float(
            np.linalg.norm(difference.ravel()) / max(np.linalg.norm(dense.astype(np.float64).ravel()), 1e-30))
        cpp_normalized = np.fromfile(str(args.cpp_binding_trace_prefix) + ".normalized.f32", "<f4").reshape(vertices, 3)
        cpp_neighbors = np.fromfile(str(args.cpp_binding_trace_prefix) + ".neighbors.u32", "<u4").reshape(vertices, 8)
        cpp_interpolation = np.fromfile(
            str(args.cpp_binding_trace_prefix) + ".interpolation.f32", "<f4").reshape(vertices, 8)
        cpp_interpolation /= cpp_interpolation.sum(axis=1, keepdims=True)
        manifest["normalized_vertices_max_abs"] = float(np.max(np.abs(cpp_normalized - normalized)))
        manifest["nearest_neighbor_slot_agreement"] = float(np.mean(cpp_neighbors == nearest_indices))
        _, cpp_py_joints, cpp_py_weights = integrate(
            cpp_dense, cpp_neighbors, cpp_interpolation)
        cpp_joints = np.fromfile(str(args.cpp_binding_trace_prefix) + ".joints.u16", "<u2").reshape(vertices, 4)
        cpp_weights = np.fromfile(str(args.cpp_binding_trace_prefix) + ".weights.f32", "<f4").reshape(vertices, 4)
        cpp_py_dense = np.zeros((vertices, joints_count), dtype=np.float32)
        cpp_final_dense = np.zeros_like(cpp_py_dense)
        reference_final_dense = np.zeros_like(cpp_py_dense)
        rows = np.arange(vertices)
        for slot in range(4):
            np.add.at(cpp_py_dense, (rows, cpp_py_joints[:, slot]), cpp_py_weights[:, slot])
            np.add.at(cpp_final_dense, (rows, cpp_joints[:, slot]), cpp_weights[:, slot])
            np.add.at(reference_final_dense, (rows, reference_joints[:, slot]), reference_weights[:, slot])
        post_difference = cpp_final_dense.astype(np.float64) - cpp_py_dense.astype(np.float64)
        manifest["cpp_postprocess_reproduction_max_abs"] = float(np.max(np.abs(post_difference)))
        manifest["cpp_postprocess_reproduction_relative_l2"] = float(
            np.linalg.norm(post_difference.ravel()) /
            max(np.linalg.norm(cpp_py_dense.astype(np.float64).ravel()), 1e-30))
        cpp_support = cpp_final_dense > 0
        reference_support = reference_final_dense > 0
        manifest["binding_top4_exact_vertex_fraction"] = float(np.mean(
            np.all(cpp_support == reference_support, axis=1)))
        manifest["binding_top4_slot_fraction"] = float(
            np.sum(cpp_support & reference_support) / (vertices * 4))

    reference_condition.tofile(args.fixture / "reference-condition-latents.f32")
    for name, values in condition_stages.items():
        values.astype("<f4").tofile(args.fixture / f"reference-{name}.f32")
    dense.tofile(args.fixture / "reference-sampled-skin.f32")
    sampled_points.astype("<f4").tofile(args.fixture / "skin-points.f32")
    sampled_normals.astype("<f4").tofile(args.fixture / "skin-normals.f32")
    condition_queries.astype("<i4").tofile(args.fixture / "skin-queries.i32")
    codes.astype("<i4").tofile(args.fixture / "skin-codes.i32")
    normalized.astype("<f4").tofile(args.fixture / "normalized-vertices.f32")
    nearest_indices.astype("<u4").tofile(args.fixture / "reference-neighbors.u32")
    interpolation.astype("<f4").tofile(args.fixture / "reference-interpolation.f32")
    reference_dense.tofile(args.fixture / "reference-dense-weights.f32")
    reference_joints.tofile(args.fixture / "reference-joints.u16")
    reference_weights.tofile(args.fixture / "reference-weights.f32")
    expected = animated_vertices(positions, reference_joints, reference_weights, parents, rest,
                                 roots, rotations, manifest["selected_frames"])
    expected.tofile(args.fixture / "expected-vertices.f32")
    manifest["reference"] = (
        "released SkinVAE safetensors in direct PyTorch F32 plus upstream "
        "Asset.from_data interpolation and Blender top-four export")
    manifest["cpp_backend"] = args.cpp_backend
    manifest["binding_selection"] = "top four interpolated learned weights"
    if args.cpp_backend == "cpu":
        manifest["binding_max_abs_tolerance"] = .01
        manifest["binding_relative_l2_tolerance"] = .001
        manifest["binding_exact_top4_tolerance"] = 1.0
        manifest["binding_shared_top4_tolerance"] = 1.0
        manifest["max_abs_tolerance"] = .01
        manifest["relative_l2_tolerance"] = .001
        manifest["frame_relative_l2_tolerance"] = .01
    else:
        # GGML Vulkan F32 reductions differ slightly from PyTorch CUDA. The
        # dense-network bounds are asserted by full_binding_parity; here the
        # important discrete contract is that at least 99.5% of selected bone
        # slots agree before checking normalized weights and animated vertices.
        manifest["binding_max_abs_tolerance"] = .26
        manifest["binding_relative_l2_tolerance"] = .05
        manifest["binding_exact_top4_tolerance"] = .99
        manifest["binding_shared_top4_tolerance"] = .998
        manifest["max_abs_tolerance"] = .16
        manifest["relative_l2_tolerance"] = .002
        manifest["frame_relative_l2_tolerance"] = .02
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
