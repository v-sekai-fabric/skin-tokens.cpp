#!/usr/bin/env python3
"""Capture constrained greedy TokenRig skin-code generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mesh-embedding", type=Path, required=True)
    parser.add_argument("--prefix", type=Path,
                        help="little-endian int32 supplied-skeleton prefix; defaults to the small fixture")
    parser.add_argument("--steps", type=int,
                        help="number of constrained skin-code tokens; defaults to eight")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = Qwen3Config(vocab_size=33036, hidden_size=896, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=3192, rms_norm_eps=1e-6,
        rope_theta=1_000_000.0, attention_bias=False, use_cache=False,
        tie_word_embeddings=False)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        state = {name.removeprefix("transformer."): source.get_tensor(name).float()
                 for name in source.keys() if name.startswith("transformer.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"model.rotary_emb.inv_freq"} or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(args.device).eval()
    mesh = torch.from_numpy(np.fromfile(args.mesh_embedding, dtype="<f4").reshape(1, -1, 896)).to(args.device)
    prefix_values = (np.fromfile(args.prefix, dtype="<i4") if args.prefix is not None else
                     np.asarray([257, 263, 128, 128, 128, 128, 196, 128, 258], dtype="<i4"))
    if len(prefix_values) < 3 or prefix_values[0] != 257 or prefix_values[-1] != 258:
        raise ValueError("invalid TokenizerPart skeleton prefix")
    prefix = torch.from_numpy(prefix_values.astype(np.int64)[None]).to(args.device)
    steps = args.steps if args.steps is not None else 8
    if steps <= 0 or steps % 4:
        raise ValueError("constrained generation steps must be a positive multiple of four")
    generated: list[int] = []
    last_logits = []
    logits_trajectory = []
    with torch.inference_mode():
        for _ in range(steps):
            token_ids = torch.cat((prefix, torch.tensor([generated], dtype=torch.long, device=args.device)), dim=1)
            token_embedding = model.model.embed_tokens(token_ids)
            logits = model(inputs_embeds=torch.cat((mesh, token_embedding), dim=1)).logits[0, -1]
            token = int(torch.argmax(logits[267:33035]).item()) + 267
            generated.append(token)
            last_logits.append(float(logits[token]))
            logits_trajectory.append(logits.float().cpu().numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    np.asarray(generated, dtype="<i4").tofile(args.output / "token-generation-codes.i32")
    np.asarray(logits_trajectory, dtype="<f4").tofile(args.output / "token-generation-logits.f32")
    prefix_values.astype("<i4").tofile(args.output / "token-generation-prefix.i32")
    manifest = {"reference": "Transformers 4.57.1 Qwen3 greedy constrained generation",
                "prefix": prefix[0].tolist(), "mesh_tokens": mesh.shape[1],
                "tokens": generated, "codes": [value - 267 for value in generated],
                "joint_count": steps // 4,
                "selected_logits": last_logits}
    (args.output / "token-generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
