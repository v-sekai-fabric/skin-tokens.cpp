#!/usr/bin/env python3
"""Evaluate the released mesh encoder on a C++-captured full point cloud."""

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
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        weights = {name: source.get_tensor(name).to(args.device, dtype=dtype)
                   for name in source.keys()
                   if name.startswith(("mesh_encoder.", "output_proj."))}

    def w(name: str) -> torch.Tensor:
        return weights[name]

    def linear(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.linear(x, w(stem + ".weight"), weights.get(stem + ".bias"))

    def norm(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), w(stem + ".weight"), w(stem + ".bias"), 1e-5)

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, query, width = q.shape
        data, heads, dim = k.shape[1], 8, width // 8
        q = q.view(batch, query, heads, dim).transpose(1, 2)
        k = k.view(batch, data, heads, dim).transpose(1, 2)
        v = v.view(batch, data, heads, dim).transpose(1, 2)
        value = F.scaled_dot_product_attention(q, k, v)
        return value.transpose(1, 2).contiguous().view(batch, query, width)

    def mlp(x: torch.Tensor, stem: str) -> torch.Tensor:
        return linear(F.gelu(linear(x, stem + ".c_fc")), stem + ".c_proj")

    prefix = str(args.input_prefix)
    points_np = np.fromfile(prefix + ".points.f32", dtype="<f4").reshape(1, -1, 3)
    normals_np = np.fromfile(prefix + ".normals.f32", dtype="<f4").reshape(1, -1, 3)
    queries = torch.from_numpy(np.fromfile(prefix + ".queries.i32", dtype="<i4").astype(np.int64)).to(args.device)
    points = torch.from_numpy(points_np).to(args.device, dtype=dtype)
    normals = torch.from_numpy(normals_np).to(args.device, dtype=dtype)
    frequencies = 2.0 ** torch.arange(8, device=args.device, dtype=dtype)
    encoded = (points[..., None] * frequencies).view(1, points.shape[1], -1)
    embedded = torch.cat((points, encoded.sin(), encoded.cos(), normals), dim=-1)

    base = "mesh_encoder.encoder"
    projected = linear(embedded, base + ".input_proj")
    state = projected[:, queries]
    stem = base + ".cross_attn"
    qn, dn = norm(state, stem + ".ln_1"), norm(projected, stem + ".ln_2")
    q = linear(qn, stem + ".attn.c_q")
    kv = linear(dn, stem + ".attn.c_kv")
    k, v = kv.chunk(2, dim=-1)
    state = state + linear(attention(q, k, v), stem + ".attn.c_proj")
    state = state + mlp(norm(state, stem + ".ln_3"), stem + ".mlp")
    for index in range(8):
        stem = f"{base}.self_attn.resblocks.{index}"
        qkv = linear(norm(state, stem + ".ln_1"), stem + ".attn.c_qkv")
        q, k, v = qkv.chunk(3, dim=-1)
        state = state + linear(attention(q, k, v), stem + ".attn.c_proj")
        state = state + mlp(norm(state, stem + ".ln_2"), stem + ".mlp")
    state = norm(state, base + ".ln_post")
    state = linear(state, "output_proj.0")
    state = F.rms_norm(state, (896,), w("output_proj.1.weight"), eps=torch.finfo(dtype).eps)
    output = state.float().cpu().numpy().astype("<f4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    print(f"points={points.shape[1]} queries={queries.numel()} output={output.shape}")


if __name__ == "__main__":
    main()
