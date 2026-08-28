#!/usr/bin/env python3
"""Capture a deterministic F32 SkinTokens condition/decode trajectory.

This is a direct expression of the released architecture over its verified
safetensors intermediate. It deliberately avoids importing upstream model
classes so the fixture remains usable in the small trusted reference image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(name).float() for name in source.keys()
                   if name.startswith("vae.model.")}

    def w(name: str) -> torch.Tensor:
        return weights["vae.model." + name]

    def linear(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.linear(x, w(stem + ".weight"), weights.get("vae.model." + stem + ".bias"))

    def norm(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), w(stem + ".weight"), w(stem + ".bias"), 1e-5)

    def embedded(x: torch.Tensor) -> torch.Tensor:
        position, features = x[..., :3], x[..., 3:]
        frequency = (2.0 ** torch.arange(8, dtype=torch.float32)) * torch.pi
        phase = torch.arange(8, dtype=torch.float32)
        for index in range(8):
            step = (index + 1) / 8.0
            phase[index] = torch.pow(torch.tensor(8), 1.0 - step) + step
        phase *= torch.pi * 2.0
        frequency, phase = frequency.to(position.device), phase.to(position.device)
        angle = (position[..., None] * frequency).reshape(*position.shape[:-1], -1)
        shifted = (position[..., None] * torch.pi * 0.5 + phase).reshape(*position.shape[:-1], -1)
        position_embedding = torch.cat((position, angle.sin() + shifted.sin(),
                                        angle.cos() + shifted.cos()), dim=-1)
        return torch.cat((position_embedding, features), dim=-1)

    def attention(query_input: torch.Tensor, data_input: torch.Tensor, stem: str,
                  cross: bool) -> torch.Tensor:
        heads, dim = 12, 64
        query = linear(query_input, stem + ".to_q")
        data = norm(data_input, stem + ".norm_cross") if cross else data_input
        key = linear(data, stem + ".to_k")
        value = linear(data, stem + ".to_v")
        if cross:
            key, value = torch.cat((key, value), dim=-1).view(
                1, key.shape[1], heads, dim * 2).split(dim, dim=-1)
            query = query.view(1, query.shape[1], heads, dim)
        else:
            query, key, value = torch.cat((query, key, value), dim=-1).view(
                1, query.shape[1], heads, dim * 3).split(dim, dim=-1)
        probability = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", query, key) / np.sqrt(dim), dim=-1)
        result = torch.einsum("bhqk,bkhd->bqhd", probability, value).reshape(1, query.shape[1], heads * dim)
        return linear(result, stem + ".to_out.0")

    def block(state: torch.Tensor, stem: str, data: torch.Tensor | None = None) -> torch.Tensor:
        if data is None:
            state = state + attention(norm(state, stem + ".norm1"),
                                      norm(state, stem + ".norm1"), stem + ".attn1", False)
        else:
            state = state + attention(norm(state, stem + ".norm2"), data, stem + ".attn2", True)
        ff = linear(F.gelu(linear(norm(state, stem + ".norm3"), stem + ".ff.net.0.proj")),
                    stem + ".ff.net.2")
        return state + ff

    generator = torch.Generator().manual_seed(20260826)
    points = torch.randn((1, 64, 3), generator=generator) * 0.35
    normals = F.normalize(torch.randn((1, 64, 3), generator=generator), dim=-1)
    condition = torch.cat((points, normals), dim=-1)
    condition_embedding = embedded(condition)
    query_indices = torch.tensor([0, 7, 13, 21, 32, 44, 55, 63])
    state = linear(condition_embedding[:, query_indices], "cond_encoder.proj_in")
    data = linear(condition_embedding, "cond_encoder.proj_in")
    captured = {"points": points, "normals": normals, "condition_embedding": condition_embedding,
                "cond_projected": data}
    state = block(state, "cond_encoder.blocks.0", data)
    captured["cond_cross"] = state
    for layer in range(2):
        state = block(state, f"cond_encoder.blocks.{layer + 1}")
        captured[f"cond_self_{layer + 1:02d}"] = state
    state = norm(state, "cond_encoder.norm_out")
    captured["cond_norm"] = state
    cond_latents = linear(state, "cond_quant")
    captured["cond_latents"] = cond_latents

    indices = torch.tensor([[0, 1, 12345, 32767]], dtype=torch.int64)
    basis = torch.tensor([1, 8, 64, 512, 4096], dtype=torch.int64)
    codes = (((indices[..., None] // basis) % 8).float() - 4.0) / 4.0
    codes = linear(codes, "FSQ.project_out")
    captured["codes"] = codes
    state = linear(torch.cat((codes, cond_latents), dim=1), "post_quant")
    captured["decoder_input"] = state
    for layer in range(10):
        state = block(state, f"decoder.blocks.{layer}")
        captured[f"decoder_self_{layer + 1:02d}"] = state
    captured["decoder_cache"] = state
    query = linear(condition_embedding, "decoder.proj_query")
    query = block(query, "decoder.blocks.10", state)
    captured["decoder_cross"] = query
    logits = torch.sigmoid(linear(norm(query, "decoder.norm_out"), "decoder.proj_out"))
    captured["output"] = logits

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.detach().numpy().astype("<f4") for name, value in captured.items()}
    np.savez(args.output / "skin-vae.npz", **arrays)
    for name, value in arrays.items():
        value.tofile(args.output / f"skin-vae-{name}.f32")
    manifest = {"reference": "released SkinFSQCVAE, direct PyTorch F32",
                "query_indices": query_indices.tolist(), "fsq_indices": indices[0].tolist(),
                "tensors": {name: list(value.shape) for name, value in arrays.items()}}
    (args.output / "skin-vae.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
