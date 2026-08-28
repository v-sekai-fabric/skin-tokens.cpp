#!/usr/bin/env python3
"""Capture deterministic Qwen3 layer boundaries from the safe checkpoint.

This runs only in the trusted reference image. The normal C++ build and tests
never import PyTorch or Transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()

    config = Qwen3Config(
        vocab_size=33036,
        hidden_size=896,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=3192,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    reference_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = Qwen3ForCausalLM(config).to(dtype=reference_dtype)
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        state = {name.removeprefix("transformer."): source.get_tensor(name)
                 for name in source.keys() if name.startswith("transformer.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"model.rotary_emb.inv_freq"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device=args.device, dtype=reference_dtype).eval()

    # Valid skeleton-prefix token IDs with enough length to exercise the causal
    # mask and non-zero RoPE positions.
    ids = torch.tensor([[257, 263, 128, 128, 128, 128, 196, 128, 258]],
                       dtype=torch.long, device=args.device)
    captured: dict[str, np.ndarray] = {"ids": ids.cpu().numpy()}
    hooks = []
    layer = model.model.layers[0]
    modules = {
        "attn_norm": layer.input_layernorm,
        "q_linear": layer.self_attn.q_proj,
        "k_linear": layer.self_attn.k_proj,
        "v_linear": layer.self_attn.v_proj,
        "q_norm": layer.self_attn.q_norm,
        "k_norm": layer.self_attn.k_norm,
        "o_linear": layer.self_attn.o_proj,
        "ffn_norm": layer.post_attention_layernorm,
        "gate_linear": layer.mlp.gate_proj,
        "up_linear": layer.mlp.up_proj,
        "down_linear": layer.mlp.down_proj,
        "layer_output": layer,
        "final_norm": model.model.norm,
    }
    for index, block in enumerate(model.model.layers, start=1):
        modules[f"hidden_{index:02d}"] = block

    def save(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().cpu().numpy()
        return hook

    for name, module in modules.items():
        hooks.append(module.register_forward_hook(save(name)))
    def save_input(name: str):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().float().cpu().numpy()
        return hook
    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(save_input("attention")))
    with torch.inference_mode():
        captured["input"] = model.model.embed_tokens(ids).float().cpu().numpy()
        result = model(ids, output_hidden_states=True, use_cache=False)
        captured["logits"] = result.logits[:, -1].float().cpu().numpy()
    for hook in hooks:
        hook.remove()

    # Re-express the small attention prefix with the pinned Transformers
    # helpers so RoPE, scores, and softmax can be compared independently.
    with torch.inference_mode():
        h = model.model.embed_tokens(ids)
        norm = layer.input_layernorm(h)
        q = layer.self_attn.q_norm(layer.self_attn.q_proj(norm).view(1, ids.shape[1], heads := 16, 128)).transpose(1, 2)
        k = layer.self_attn.k_norm(layer.self_attn.k_proj(norm).view(1, ids.shape[1], 8, 128)).transpose(1, 2)
        v = layer.self_attn.v_proj(norm).view(1, ids.shape[1], 8, 128).transpose(1, 2)
        position_ids = torch.arange(ids.shape[1], device=args.device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        captured["q_rope"] = q.transpose(1, 2).float().cpu().numpy()
        captured["k_rope"] = k.transpose(1, 2).float().cpu().numpy()
        k = repeat_kv(k, heads // 8)
        v = repeat_kv(v, heads // 8)
        scores = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(128.0)
        captured["scores"] = scores.float().cpu().numpy()
        causal = torch.triu(torch.full_like(scores, float("-inf")), diagonal=1)
        scores = scores + causal
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        captured["probabilities"] = probabilities.float().cpu().numpy()
        manual_attention = torch.matmul(probabilities, v).transpose(1, 2).contiguous().view(1, ids.shape[1], -1)
        if not torch.allclose(manual_attention.float(), torch.from_numpy(captured["attention"]), atol=1e-5, rtol=1e-5):
            raise RuntimeError("manual attention boundary disagrees with Qwen3 eager attention")

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "qwen-layer0.npz", **captured)
    for name, value in captured.items():
        if name != "ids":
            np.asarray(value, dtype="<f4").tofile(args.output / f"qwen-layer0-{name}.f32")
    manifest = {
        "reference": "transformers 4.57.1 Qwen3 eager attention",
        "checkpoint": str(args.weights),
        "device": args.device,
        "dtype": args.dtype,
        "sequence": ids[0].tolist(),
        "tensors": {name: list(value.shape) for name, value in captured.items()},
    }
    (args.output / "qwen-layer0.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
