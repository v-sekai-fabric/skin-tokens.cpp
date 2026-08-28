#!/usr/bin/env python3
"""Run an official SkinTokens example through the released inference path.

This is intentionally an end-to-end behavioural fixture.  It loads the input
with upstream's Blender parser, applies the checkpoint's predict transform,
runs TokenRig, and exports both the raw default result and the optional
``voxel_skin`` postprocessed result.  The checked-in flash-attention shim only
selects PyTorch SDPA in environments without a compatible flash-attn wheel; it
does not replace any learned operation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from transformers import AutoModelForCausalLM


def top_four(skin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Match BpyParser.export_asset: NumPy's default argsort, then normalize by
    # the retained group count rather than by every dense channel.
    joints = np.argsort(-skin, axis=1)[:, :4]
    weights = np.take_along_axis(skin, joints, axis=1)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-30)
    return joints.astype("<u2"), weights.astype("<f4")


def write_binding(directory: Path, stem: str, skin: np.ndarray) -> None:
    dense = skin.astype("<f4")
    joints, weights = top_four(dense)
    dense.tofile(directory / f"{stem}-dense-weights.f32")
    joints.tofile(directory / f"{stem}-joints.u16")
    weights.tofile(directory / f"{stem}-weights.f32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--skin-checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-skeleton", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--num-beams", type=int, default=10)
    args = parser.parse_args()

    root = args.upstream_root.resolve()
    sys.path.insert(0, str(root))
    # The released constructor requests FlashAttention2 unconditionally.  The
    # project already uses this eager fallback for trusted layer captures when
    # a matching optional flash-attn wheel is unavailable.
    original_from_config = AutoModelForCausalLM.from_config
    AutoModelForCausalLM.from_config = lambda config, **_kwargs: original_from_config(
        config, attn_implementation="eager")

    from src.data.dataset import RigDataset
    from src.data.transform import Transform
    import src.data.vertex_group as vertex_group_module
    from src.data.vertex_group import voxel_skin
    from src.model.spec import ModelInput
    from src.model.tokenrig import TokenRig, decode
    import src.model.tokenrig as tokenrig_module
    from src.rig_package.info.asset import Asset
    from src.rig_package.parser.bpy import BpyParser
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizeInput

    # A partially-created local model directory must not shadow the official
    # identifier embedded in the checkpoint. Upstream tests only Path.exists()
    # before passing it to Transformers, which makes an empty interrupted
    # download fail instead of falling back to the Hub configuration.
    if (tokenrig_module.LLM_LOCAL_DIR.exists() and
            not (tokenrig_module.LLM_LOCAL_DIR / "config.json").is_file()):
        tokenrig_module.LLM_LOCAL_DIR = Path("/__missing_qwen_config__")

    random.seed(args.seed)
    np.random.seed(args.seed)
    original_default_rng = np.random.default_rng
    # SkinFSQCVAEModel._sample_features asks for default_rng(None), bypassing
    # np.random.seed(). Pin only that otherwise-unseeded fixture lane so
    # repeated captures with --seed are byte-reproducible.
    np.random.default_rng = lambda seed=None: original_default_rng(
        args.seed if seed is None else seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = copy.deepcopy(checkpoint["hyper_parameters"]["model_config"])
    model_config["pretrained_vae"] = str(args.skin_checkpoint.resolve())
    model = TokenRig.load_from_system_checkpoint(
        checkpoint_path=str(args.checkpoint.resolve()), model_config=model_config).to(args.device).eval()
    mesh_conditions: list[torch.Tensor] = []
    hook = model.output_proj.register_forward_hook(
        lambda _module, _inputs, value: mesh_conditions.append(value.detach().float().cpu()))
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])

    asset = BpyParser.load(str(args.input.resolve()))
    asset.cls = "articulation"
    asset.path = str(args.input.resolve())
    transform.apply(asset=asset)
    tokens = None
    if args.use_skeleton:
        if asset.parents is None or asset.joint_names is None:
            raise ValueError("--use-skeleton input has no Blender-readable armature")
        tokens = tokenizer.tokenize(TokenizeInput(
            joints=asset.joints, parents=asset.parents, cls=asset.cls,
            joint_names=asset.joint_names))

    dataset = RigDataset(data=[], transform=transform, process_fn=model._process_fn,
                         tokenizer=tokenizer)
    batch = dataset.collate_fn([ModelInput(asset=asset, tokens=tokens)])
    batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    batch["generate_kwargs"] = {
        "max_length": args.max_length,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": 1,
        "num_beams": args.num_beams,
        "do_sample": True,
    }
    skeleton_tokens = None
    if args.use_skeleton:
        mask = batch["skeleton_mask"][0] == 1
        skeleton_tokens = [batch["skeleton_tokens"][0][mask].cpu().numpy()]

    with torch.inference_mode():
        result = model.predict_step(
            batch, skeleton_tokens=skeleton_tokens, make_asset=True)["results"][0]
    hook.remove()
    if result.asset is None or result.output_ids is None or result.skin_pred is None:
        raise RuntimeError("upstream inference returned an incomplete result")
    if result.cond_latents is None or len(mesh_conditions) != 1:
        raise RuntimeError("upstream inference did not expose both conditioning tensors")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized = result.asset
    normalized.vertices.astype("<f4").tofile(output / "vertices.f32")
    normalized.vertices.astype("<f8").tofile(output / "vertices.f64")
    normalized.faces.astype("<u4").tofile(output / "faces.u32")
    asset.sampled_vertices.astype("<f4").tofile(output / "sampled-points.f32")
    asset.sampled_normals.astype("<f4").tofile(output / "sampled-normals.f32")
    output_ids = result.output_ids.detach().cpu().numpy().astype("<i4")
    output_ids.tofile(output / "tokens.i32")
    switch = np.flatnonzero(output_ids == tokenizer.eos)
    if len(switch) != 1:
        raise RuntimeError(f"expected one skeleton/skin switch token, found {len(switch)}")
    skeleton_tokens_out = output_ids[:switch[0] + 1]
    skin_codes = output_ids[switch[0] + 1:] - tokenizer.vocab_size
    if len(skin_codes) != len(normalized.parents) * model.tokens_per_skin:
        raise RuntimeError("generated skin-code count does not match generated skeleton")
    skeleton_tokens_out.astype("<i4").tofile(output / "skeleton-tokens.i32")
    skin_codes.astype("<i4").tofile(output / "skin-codes.i32")
    result.cond_latents.detach().float().cpu().numpy().astype("<f4").tofile(
        output / "condition-latents.f32")
    mesh_conditions[0].numpy().astype("<f4").tofile(output / "mesh-condition.f32")
    result.skin_pred.detach().float().cpu().numpy().astype("<f4").tofile(
        output / "sampled-skin.f32")
    np.asarray(normalized.parents, dtype="<i4").tofile(output / "parents.i32")
    normalized.joints.astype("<f4").tofile(output / "joints.f32")
    normalized.joints.astype("<f8").tofile(output / "joints.f64")
    write_binding(output, "raw", normalized.skin)
    BpyParser.export(normalized, str(output / "raw.glb"), group_per_vertex=4)

    postprocessed = normalized.copy()
    voxel = postprocessed.voxel(resolution=196)
    postprocessed.skin *= voxel_skin(
        grid=0, grid_coords=voxel.coords, joints=postprocessed.joints,
        vertices=postprocessed.vertices, faces=postprocessed.faces,
        mode="square", voxel_size=voxel.voxel_size)
    postprocessed.normalize_skin()
    write_binding(output, "postprocessed", postprocessed.skin)
    BpyParser.export(postprocessed, str(output / "postprocessed.glb"), group_per_vertex=4)

    # The production C++ milestone is explicitly F32. The released TokenRig
    # constructor converts the VAE (including nonpersistent PMPE buffers) to
    # BF16, so promotion alone is not a clean F32 reference. Restore those two
    # analytically-defined buffers after promoting the learned BF16 checkpoint
    # values, then decode the exact same upstream condition and generated IDs.
    model.vae.float()
    frequencies = (2.0 ** torch.arange(8, dtype=torch.float32, device=args.device)) * torch.pi
    phase = torch.arange(8, dtype=torch.float32, device=args.device)
    for index in range(8):
        step = (index + 1) / 8.0
        phase[index] = torch.pow(torch.tensor(8, device=args.device), 1.0 - step) + step
    phase *= torch.pi * 2.0
    model.vae.model.embedder.frequencies = frequencies
    model.vae.model.embedder.phase = phase
    with torch.inference_mode():
        f32_result = decode(
            cond=result.cond.float(), cond_latents=result.cond_latents.float(),
            inputs_ids=result.output_ids, tokenizer=tokenizer,
            tokens_per_skin=model.tokens_per_skin, vae=model.vae)
    if f32_result["skin_pred"] is None:
        raise RuntimeError("clean F32 decode returned no skin predictions")
    f32_sampled_skin = f32_result["skin_pred"].detach().float().cpu().numpy()
    f32_sampled_skin.astype("<f4").tofile(output / "f32-sampled-skin.f32")
    detokenized = f32_result["detokenize_output"]
    f32_asset = Asset.from_data(
        vertices=asset.vertices, faces=asset.faces,
        sampled_vertices=asset.sampled_vertices, sampled_skin=f32_sampled_skin,
        joints=detokenized.joints, parents=np.asarray(detokenized.parents),
        cls=asset.cls, path=asset.path)
    write_binding(output, "f32-raw", f32_asset.skin)
    f32_postprocessed = f32_asset.copy()
    f32_voxel = f32_postprocessed.voxel(resolution=196)
    surface_trace: dict[str, object] = {}
    original_shortest_path = vertex_group_module.shortest_path
    def capture_shortest_path(graph, *shortest_args, **shortest_kwargs):
        graph = graph.tocsr()
        surface_trace["graph"] = graph.copy()
        value = original_shortest_path(graph, *shortest_args, **shortest_kwargs)
        surface_trace["raw_distances"] = value.copy()
        return value
    vertex_group_module.shortest_path = capture_shortest_path
    try:
        f32_surface = voxel_skin(
            grid=0, grid_coords=f32_voxel.coords, joints=f32_postprocessed.joints,
            vertices=f32_postprocessed.vertices, faces=f32_postprocessed.faces,
            mode="square", voxel_size=f32_voxel.voxel_size)
    finally:
        vertex_group_module.shortest_path = original_shortest_path
    f32_postprocessed.skin *= f32_surface
    f32_postprocessed.normalize_skin()
    write_binding(output, "f32-postprocessed", f32_postprocessed.skin)
    f32_surface.astype("<f4").tofile(output / "f32-surface-weights.f32")
    f32_voxel.coords.astype("<i4").tofile(output / "f32-voxel-coordinates.i32")
    _, f32_seeds = cKDTree(np.concatenate(
        (f32_postprocessed.vertices, f32_voxel.coords), axis=0)).query(
            f32_postprocessed.joints)
    np.asarray(f32_seeds, dtype="<u4").tofile(output / "f32-surface-seeds.u32")
    graph = surface_trace["graph"]
    graph.indptr.astype("<i4").tofile(output / "f32-graph-indptr.i32")
    graph.indices.astype("<i4").tofile(output / "f32-graph-indices.i32")
    graph.data.astype("<f8").tofile(output / "f32-graph-weights.f64")
    surface_distances = surface_trace["raw_distances"][:, :len(f32_postprocessed.vertices)]
    unreachable = np.isinf(surface_distances).all(axis=0)
    fallback_distances, fallback_joints = cKDTree(f32_postprocessed.joints).query(
        f32_postprocessed.vertices[unreachable], min(len(f32_postprocessed.joints), 3))
    unreachable_vertices = np.flatnonzero(unreachable)
    surface_distances[
        fallback_joints,
        np.repeat(unreachable_vertices, min(len(f32_postprocessed.joints), 3)).reshape(-1, 3),
    ] = fallback_distances
    maximum_surface_distance = float(np.max(surface_distances[np.isfinite(surface_distances)]))
    surface_distances = np.nan_to_num(
        surface_distances, nan=maximum_surface_distance,
        posinf=maximum_surface_distance, neginf=maximum_surface_distance)
    surface_distances = np.maximum(surface_distances, 1.0e-6)
    surface_distances.astype("<f8").tofile(output / "f32-surface-distances.f64")

    manifest = {
        "format": 1,
        "reference": "official SkinTokens predict path with PyTorch eager attention fallback",
        "upstream_revision": "273b691d35989d71cd17ff2895fdc735097b92d1",
        "source": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "use_skeleton": args.use_skeleton,
        "seed": args.seed,
        "unseeded_default_rng_override": args.seed,
        "vertex_count": int(len(normalized.vertices)),
        "face_count": int(len(normalized.faces)),
        "joint_count": int(len(normalized.parents)),
        "sample_count": int(len(asset.sampled_vertices)),
        "token_count": int(result.output_ids.numel()),
        "skeleton_token_count": int(len(skeleton_tokens_out)),
        "skin_code_count": int(len(skin_codes)),
        "condition_token_count": int(result.cond_latents.shape[0]),
        "condition_width": int(result.cond_latents.shape[1]),
        "mesh_condition_token_count": int(mesh_conditions[0].shape[-2]),
        "mesh_condition_width": int(mesh_conditions[0].shape[-1]),
        "f32_reference": (
            "official decoder with checkpoint values promoted to F32 and "
            "analytically reconstructed F32 PMPE buffers"),
        "voxel_count": int(len(f32_voxel.coords)),
        "graph_node_count": int(graph.shape[0]),
        "graph_edge_count": int(graph.nnz),
        "maximum_surface_distance": maximum_surface_distance,
        "parameters": batch["generate_kwargs"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
