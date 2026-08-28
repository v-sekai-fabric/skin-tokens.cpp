#!/usr/bin/env python3
"""Capture a small deterministic F32 Michelangelo encoder trajectory."""

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
                   if name.startswith(("mesh_encoder.", "output_proj."))}
    def w(name: str) -> torch.Tensor:
        return weights[name]
    def linear(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.linear(x, w(stem + ".weight"), weights.get(stem + ".bias"))
    def norm(x: torch.Tensor, stem: str) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), w(stem + ".weight"), w(stem + ".bias"), 1e-5)
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int = 8) -> torch.Tensor:
        batch, query, _, dim = q.shape
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        p = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / np.sqrt(dim), dim=-1)
        return torch.matmul(p, v).transpose(1, 2).contiguous().view(batch, query, heads * dim)
    def mlp(x: torch.Tensor, stem: str) -> torch.Tensor:
        return linear(F.gelu(linear(x, stem + ".c_fc")), stem + ".c_proj")

    generator = torch.Generator().manual_seed(20260826)
    points = torch.randn((1, 64, 3), generator=generator) * 0.35
    normals = F.normalize(torch.randn((1, 64, 3), generator=generator), dim=-1)
    frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
    encoded = (points[..., None] * frequencies).view(1, 64, -1)
    embedded = torch.cat((points, encoded.sin(), encoded.cos(), normals), dim=-1)
    base = "mesh_encoder.encoder"
    projected = linear(embedded, base + ".input_proj")
    query_indices = torch.tensor([0, 7, 13, 21, 32, 44, 55, 63])
    query = projected[:, query_indices]

    stem = base + ".cross_attn"
    qn, dn = norm(query, stem + ".ln_1"), norm(projected, stem + ".ln_2")
    q = linear(qn, stem + ".attn.c_q").view(1, len(query_indices), 8, -1)
    kv = linear(dn, stem + ".attn.c_kv").view(1, projected.shape[1], 8, -1); k, v = kv.chunk(2, dim=-1)
    query = query + linear(attention(q, k, v), stem + ".attn.c_proj")
    query = query + mlp(norm(query, stem + ".ln_3"), stem + ".mlp")
    captured = {"points": points, "normals": normals, "embedded": embedded,
                "projected": projected, "cross": query}
    for index in range(8):
        stem = f"{base}.self_attn.resblocks.{index}"
        qkv = linear(norm(query, stem + ".ln_1"), stem + ".attn.c_qkv")
        q, k, v = qkv.view(1, len(query_indices), 8, -1).chunk(3, dim=-1)
        query = query + linear(attention(q, k, v), stem + ".attn.c_proj")
        query = query + mlp(norm(query, stem + ".ln_2"), stem + ".mlp")
        captured[f"self_{index + 1:02d}"] = query
    query = norm(query, base + ".ln_post")
    captured["ln_post"] = query
    query = linear(query, "output_proj.0")
    # torch.nn.RMSNorm(eps=None) uses the input dtype's machine epsilon.
    query = F.rms_norm(query, (896,), w("output_proj.1.weight"), eps=torch.finfo(query.dtype).eps)
    captured["output"] = query

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.detach().numpy().astype("<f4") for name, value in captured.items()}
    np.savez(args.output / "mesh-encoder.npz", **arrays)
    for name, value in arrays.items(): value.tofile(args.output / f"mesh-encoder-{name}.f32")
    manifest = {"reference": "released Michelangelo encoder, direct PyTorch F32",
                "query_indices": query_indices.tolist(),
                "tensors": {name: list(value.shape) for name, value in arrays.items()}}
    (args.output / "mesh-encoder.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
