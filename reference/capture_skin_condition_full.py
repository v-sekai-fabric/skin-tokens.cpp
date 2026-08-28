#!/usr/bin/env python3
"""Evaluate the released SkinVAE condition encoder on a captured full cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(name).float().to(args.device)
                   for name in source.keys() if name.startswith("vae.model.")}

    def w(name: str) -> torch.Tensor: return weights["vae.model." + name]
    def linear(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.linear(x, w(stem + ".weight"), weights.get("vae.model." + stem + ".bias"))
    def norm(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), w(stem + ".weight"), w(stem + ".bias"), 1e-5)
    def embedded(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        frequency = (2.0 ** torch.arange(8, dtype=torch.float32)) * torch.pi
        phase = torch.arange(8, dtype=torch.float32)
        for index in range(8):
            step = (index + 1) / 8.0
            phase[index] = torch.pow(torch.tensor(8), 1.0 - step) + step
        phase *= torch.pi * 2.0
        frequency, phase = frequency.to(args.device), phase.to(args.device)
        angle = (points[..., None] * frequency).reshape(1, points.shape[1], -1)
        shifted = (points[..., None] * torch.pi * .5 + phase).reshape(1, points.shape[1], -1)
        return torch.cat((points, angle.sin() + shifted.sin(), angle.cos() + shifted.cos(), normals), -1)
    def attention(qin: torch.Tensor, din: torch.Tensor, stem: str, cross: bool) -> torch.Tensor:
        q = linear(qin, stem + ".to_q")
        data = norm(din, stem + ".norm_cross") if cross else din
        k, v = linear(data, stem + ".to_k"), linear(data, stem + ".to_v")
        if cross:
            k, v = torch.cat((k, v), dim=-1).view(
                1, k.shape[1], 12, 128).split(64, dim=-1)
        else:
            q, k, v = torch.cat((q, k, v), dim=-1).view(
                1, q.shape[1], 12, 192).split(64, dim=-1)
        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(1, x.shape[1], 12, 64).transpose(1, 2)
        result = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        return linear(result.transpose(1, 2).reshape(1, q.shape[1], 768), stem + ".to_out.0")
    def block(state: torch.Tensor, stem: str, data: torch.Tensor | None = None) -> torch.Tensor:
        if data is None:
            value = norm(state, stem + ".norm1")
            state = state + attention(value, value, stem + ".attn1", False)
        else:
            state = state + attention(norm(state, stem + ".norm2"), data, stem + ".attn2", True)
        return state + linear(F.gelu(linear(norm(state, stem + ".norm3"), stem + ".ff.net.0.proj")),
                              stem + ".ff.net.2")

    prefix = str(args.input_prefix)
    points = torch.from_numpy(np.fromfile(prefix + ".points.f32", "<f4").reshape(1, -1, 3)).to(args.device)
    normals = torch.from_numpy(np.fromfile(prefix + ".normals.f32", "<f4").reshape(1, -1, 3)).to(args.device)
    queries = torch.from_numpy(np.fromfile(prefix + ".queries.i32", "<i4").astype(np.int64)).to(args.device)
    values = embedded(points, normals)
    data = linear(values, "cond_encoder.proj_in")
    state = linear(values[:, queries], "cond_encoder.proj_in")
    state = block(state, "cond_encoder.blocks.0", data)
    for layer in range(2): state = block(state, f"cond_encoder.blocks.{layer + 1}")
    state = linear(norm(state, "cond_encoder.norm_out"), "cond_quant")
    output = state.cpu().numpy().astype("<f4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    print(f"points={points.shape[1]} queries={queries.numel()} output={output.shape}")


if __name__ == "__main__": main()
