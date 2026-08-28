#!/usr/bin/env python3
"""Capture upstream TokenRig's unconstrained skeleton/skin token sequence.

This runs only in the pinned reference container. It deliberately mirrors the
released VocabSwitchingLogitsProcessor, including its boundary arithmetic, so
the C++ port does not infer generation semantics from the decoder alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import LogitsProcessor, LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM


class TokenizerPart:
    bos = 257
    eos = 258
    pad = 259
    vocab_size = 267

    @staticmethod
    def next_possible_token(ids: np.ndarray) -> list[int]:
        state = "expect_bos"
        for token in ids:
            if state == "expect_bos":
                assert token == 257
                state = "expect_cls_or_part_or_joint"
            elif state == "expect_cls_or_part_or_joint":
                state = "expect_joint_2" if token < 256 else ("expect_part_or_joint" if token >= 263 else "expect_joint")
            elif state == "expect_part_or_joint":
                state = "expect_joint_2" if token < 256 else "expect_part_or_joint"
            elif state == "expect_joint_2": state = "expect_joint_3"
            elif state == "expect_joint_3": state = "expect_branch_or_part_or_joint"
            elif state == "expect_branch_or_part_or_joint":
                state = "expect_joint" if token >= 256 else "expect_joint_2"
            elif state == "expect_joint": state = "expect_joint_2"
        coords = list(range(256))
        parts = [260, 261, 262]
        return {
            "expect_bos": [257],
            "expect_cls_or_part_or_joint": [263, 264, 265, 266] + parts + coords,
            "expect_part_or_joint": parts + coords + [258],
            "expect_joint_2": coords,
            "expect_joint_3": coords,
            "expect_branch_or_part_or_joint": coords + parts + [256, 258],
            "expect_joint": coords,
        }[state]

    @staticmethod
    def bones_in_sequence(ids: np.ndarray) -> int:
        state, count, branch = "expect_bos", 0, False
        for token in ids:
            if state == "expect_bos": state = "expect_cls_or_part_or_joint"
            elif state == "expect_cls_or_part_or_joint":
                state = "expect_joint_2" if token < 256 else ("expect_part_or_joint" if token >= 263 else "expect_joint")
            elif state == "expect_part_or_joint": state = "expect_joint_2" if token < 256 else "expect_part_or_joint"
            elif state == "expect_joint_2": state = "expect_joint_3"
            elif state == "expect_joint_3":
                if not branch: count += 1
                branch = False; state = "expect_branch_or_part_or_joint"
            elif state == "expect_branch_or_part_or_joint":
                if token == 256: state, branch = "expect_joint", True
                elif token < 256: state = "expect_joint_2"
                else: state = "expect_joint"
            elif state == "expect_joint": state = "expect_joint_2"
            if token == 258: break
        return count


class SwitchProcessor(LogitsProcessor):
    def __init__(self, init: torch.Tensor): self.init = init
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for batch, generated in enumerate(input_ids):
            mask = torch.full_like(scores[batch], float("-inf"))
            sequence = torch.cat([self.init, generated])
            length = len(sequence)
            if 258 in sequence:
                mask[258:] = 0
                where = torch.where(sequence == 258)[0][:1]
                joints = TokenizerPart.bones_in_sequence(sequence.detach().cpu().numpy())
                if (length - where) == joints * 4:
                    mask[:] = float("-inf"); mask[33035] = 0
                else: mask[33035] = float("-inf")
            else:
                mask[TokenizerPart.next_possible_token(sequence.detach().cpu().numpy())] = 0
            scores[batch] += mask
        return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mesh-embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--beams", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", type=Path,
                        help="optional complete skeleton-token prefix for constrained skin generation")
    args = parser.parse_args()
    config = Qwen3Config(vocab_size=33036, hidden_size=896, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=3192, rms_norm_eps=1e-6,
        rope_theta=1_000_000.0, attention_bias=False, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(config)
    with safe_open(args.weights, framework="pt", device="cpu") as source:
        state = {name.removeprefix("transformer."): source.get_tensor(name)
                 for name in source.keys() if name.startswith("transformer.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) - {"model.rotary_emb.inv_freq"} or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = model.to(args.device, dtype=dtype).eval()
    mesh = torch.from_numpy(np.fromfile(args.mesh_embedding, dtype="<f4").reshape(1, -1, 896)).to(args.device, dtype)
    start_values = ([257, 266] if args.prefix is None else
                    np.fromfile(args.prefix, dtype="<i4").astype(np.int64).tolist())
    start = torch.tensor(start_values, dtype=torch.long, device=args.device)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        inputs = torch.cat([mesh, model.model.embed_tokens(start[None])], dim=1)
        initial_logits = model(inputs_embeds=inputs).logits[0, -1].float().cpu().numpy().astype("<f4")
        result = model.generate(inputs_embeds=inputs,
            max_length=args.max_length, top_k=args.top_k, top_p=args.top_p,
            temperature=args.temperature, repetition_penalty=args.repetition_penalty,
            num_beams=args.beams, num_return_sequences=1,
            do_sample=True, bos_token_id=257, eos_token_id=33035, pad_token_id=259,
            logits_processor=LogitsProcessorList([SwitchProcessor(start)]))[0]
    full = torch.cat([start, result]).cpu().numpy().astype("<i4")
    switch = int(np.where(full == 258)[0][0])
    summary = {"tokens": full.tolist(), "switch_index": switch,
               "joint_count": TokenizerPart.bones_in_sequence(full),
               "tokens_after_switch": len(full) - switch - 1}
    args.output.mkdir(parents=True, exist_ok=True)
    full.tofile(args.output / "unconstrained-tokens.i32")
    initial_logits.tofile(args.output / "unconstrained-initial-logits.f32")
    (args.output / "unconstrained.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "tokens"}, indent=2))


if __name__ == "__main__": main()
