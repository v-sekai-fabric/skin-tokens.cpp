#!/usr/bin/env python3
"""Capture final supplied-skeleton weights through the released upstream model.

Unlike the layer-oriented F32 capture, this loads the official Lightning
checkpoint and executes its SkinVAE condition encoder and decoder under the
same CUDA BF16 autocast used by TokenRig.generate(). Generated FSQ codes and
the sampled surface cloud are held fixed so the test isolates learned binding
and final Asset/export integration from framework-specific random sampling.

The ``float32`` diagnostic promotes the already-constructed VAE. TokenRig's
constructor first converts the module, including its nonpersistent PMPE
frequency/phase buffers, to BF16; promotion cannot recover their F32 values.
It is therefore useful for locating precision effects but is not the clean F32
reference produced by ``capture_binding_vertices.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from capture_binding_vertices import animated_vertices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--skin-checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--vae-trace-prefix", type=Path, required=True)
    parser.add_argument("--token-trace-prefix", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--safe-weights", type=Path,
                        help="optional extracted TokenRig safetensors for weight-identity diagnostics")
    args = parser.parse_args()

    # The released TokenRig constructor hard-codes FlashAttention2 for Qwen,
    # although this capture does not execute Qwen. Use Transformers eager
    # attention when that optional wheel is unavailable.
    original_from_config = AutoModelForCausalLM.from_config
    AutoModelForCausalLM.from_config = lambda config, **kwargs: original_from_config(
        config, attn_implementation="eager")
    from src.model.tokenrig import TokenRig, decode

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["hyper_parameters"]["model_config"]
    model_config["pretrained_vae"] = str(args.skin_checkpoint)
    model = TokenRig.load_from_system_checkpoint(
        checkpoint_path=str(args.checkpoint), model_config=model_config).to(args.device).eval()
    if args.precision == "float32":
        model.vae = model.vae.float()
    if args.safe_weights is not None:
        with safe_open(args.safe_weights, framework="pt", device="cpu") as source:
            expected_weight = source.get_tensor("vae.model.cond_encoder.proj_in.weight").float()
        actual_weight = model.vae.model.cond_encoder.proj_in.weight.detach().float().cpu()
        delta = actual_weight - expected_weight
        print("condition proj_in weight max_abs=", float(delta.abs().max()),
              "relative_l2=", float(torch.linalg.vector_norm(delta) /
                                     torch.linalg.vector_norm(expected_weight)))

    condition_stages: dict[str, torch.Tensor] = {}
    projected_calls: list[torch.Tensor] = []
    handles = [model.vae.model.cond_encoder.proj_in.register_forward_hook(
        lambda _module, _inputs, output: projected_calls.append(output.detach().float().cpu()))]
    for index, block in enumerate(model.vae.model.cond_encoder.blocks):
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output, i=index:
                condition_stages.__setitem__("cond-cross" if i == 0 else f"cond-self-{i}",
                                             output.detach().float().cpu())))
    handles.append(model.vae.model.cond_encoder.norm_out.register_forward_hook(
        lambda _module, _inputs, output:
            condition_stages.__setitem__("cond-norm", output.detach().float().cpu())))

    manifest_path = args.fixture / "animation-vertices.json"
    manifest = json.loads(manifest_path.read_text())
    vertices = manifest["vertex_count"]
    joints_count = manifest["joint_count"]
    frames_count = manifest["frame_count"]
    positions = np.fromfile(args.fixture / "positions.f32", "<f4").reshape(vertices, 3)
    parents = np.fromfile(args.fixture / "parents.i32", "<i4")
    rest = np.fromfile(args.fixture / "rest-positions.f32", "<f4").reshape(joints_count, 3)
    roots = np.fromfile(args.fixture / "root-translations.f32", "<f4").reshape(frames_count, 3)
    rotations = np.fromfile(args.fixture / "local-rotations.f32", "<f4").reshape(
        frames_count, joints_count, 4)

    trace = str(args.vae_trace_prefix)
    sampled_points = np.fromfile(trace + ".points.f32", "<f4").reshape(-1, 3)
    sampled_normals = np.fromfile(trace + ".normals.f32", "<f4").reshape(-1, 3)
    query_indices = np.fromfile(trace + ".queries.i32", "<i4")
    prefix = np.fromfile(str(args.token_trace_prefix) + ".prefix.i32", "<i4")
    codes = np.fromfile(str(args.token_trace_prefix) + ".codes.i32", "<i4")
    if (len(sampled_points) != 54000 or sampled_points.shape != sampled_normals.shape or
            len(query_indices) != model.tokens_skin_cond):
        raise ValueError("expected the runtime's 54,000-point sampled cloud")
    if len(codes) != joints_count * 4 or prefix[0] != 257 or prefix[-1] != 258:
        raise ValueError("invalid supplied-skeleton token trace")

    device = torch.device(args.device)
    condition = torch.from_numpy(np.concatenate((sampled_points, sampled_normals), axis=1)).to(device)
    queries = torch.from_numpy(query_indices.astype(np.int64)).to(device)
    input_ids = torch.from_numpy(np.concatenate((prefix, codes + model.tokenizer.vocab_size)).astype(np.int64)).to(device)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              enabled=args.precision == "bfloat16")
    with torch.inference_mode(), autocast:
        batched = condition.unsqueeze(0)
        condition_positions, condition_features = batched[..., :3], batched[..., 3:]
        embedded = torch.cat((model.vae.model.embedder(condition_positions), condition_features), dim=-1)
        condition_latents = model.vae.model.cond_quant(
            model.vae.model.cond_encoder(embedded[:, queries], embedded))
        result = decode(
            cond=condition, cond_latents=condition_latents[0], inputs_ids=input_ids,
            tokenizer=model.tokenizer, tokens_per_skin=model.tokens_per_skin, vae=model.vae)
    for handle in handles:
        handle.remove()
    if len(projected_calls) != 2:
        raise ValueError("unexpected condition projection call count")
    condition_stages["cond-selected"] = projected_calls[0]
    condition_stages["cond-projected-head"] = projected_calls[1][:, :64]
    condition_stages["cond-embedding-head"] = embedded[:, :64].detach().float().cpu()
    sampled_skin = result["skin_pred"].float().cpu().numpy().astype("<f4")
    if sampled_skin.shape != (len(sampled_points), joints_count):
        raise ValueError(f"unexpected upstream skin shape {sampled_skin.shape}")

    # Rebuild the exact upstream predict-time affine normalization from the
    # original glTF-space geometry and supplied rig.
    model_vertices = positions[:, [0, 2, 1]].copy()
    model_vertices[:, 1] *= -1
    model_joints = rest[:, [0, 2, 1]].copy()
    model_joints[:, 1] *= -1
    low = np.minimum(model_vertices.min(axis=0), model_joints.min(axis=0))
    high = np.maximum(model_vertices.max(axis=0), model_joints.max(axis=0))
    normalized = (model_vertices - (low + high) * .5) / ((high - low).max() * .5)

    distances, neighbors = cKDTree(sampled_points).query(normalized, k=8)
    interpolation = 1.0 / (distances + 1e-8)
    interpolation /= interpolation.sum(axis=1, keepdims=True)
    dense = np.einsum("nk,nkj->nj", interpolation, sampled_skin[neighbors]).astype("<f4")
    top = np.argsort(-dense, axis=1, kind="stable")[:, :4]
    weights = np.take_along_axis(dense, top, axis=1).astype("<f4")
    weights /= weights.sum(axis=1, keepdims=True)
    joints = top.astype("<u2")

    sampled_skin.tofile(args.fixture / "reference-sampled-skin.f32")
    condition_latents[0].float().cpu().numpy().astype("<f4").tofile(
        args.fixture / "reference-condition-latents.f32")
    for name, values in condition_stages.items():
        values.numpy().astype("<f4").tofile(args.fixture / f"reference-{name}.f32")
    neighbors.astype("<u4").tofile(args.fixture / "reference-neighbors.u32")
    interpolation.astype("<f4").tofile(args.fixture / "reference-interpolation.f32")
    dense.tofile(args.fixture / "reference-dense-weights.f32")
    joints.tofile(args.fixture / "reference-joints.u16")
    weights.tofile(args.fixture / "reference-weights.f32")
    animated_vertices(positions, joints, weights, parents, rest, roots, rotations,
                      manifest["selected_frames"]).tofile(args.fixture / "expected-vertices.f32")

    manifest["reference"] = (
        "official upstream TokenRig/SkinVAE CUDA " +
        ("BF16" if args.precision == "bfloat16" else "F32") +
        " supplied-skeleton decode")
    manifest["pmpe_buffer_construction"] = (
        "TokenRig constructor BF16; float32 mode promotes already-rounded buffers")
    manifest["binding_selection"] = "upstream top four interpolated learned weights"
    manifest["skeleton_prefix"] = prefix.tolist()
    manifest["skin_codes"] = codes.tolist()
    manifest["binding_max_abs_tolerance"] = .001
    manifest["binding_relative_l2_tolerance"] = .001
    manifest["max_abs_tolerance"] = .0005
    manifest["relative_l2_tolerance"] = .0001
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "reference", "vertex_count", "joint_count", "binding_selection")}, indent=2))


if __name__ == "__main__":
    main()
