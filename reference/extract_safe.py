#!/usr/bin/env python3
"""One-time trusted extraction of official Lightning checkpoints.

Run only in the disposable reference container after SHA-256 verification.
The resulting safetensors and JSON are the sole inputs accepted by the normal
converter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file


def plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def extract(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    # These are pinned official inputs inside an unprivileged, disposable
    # container. Never move this unsafe load into the converter or host tools.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Lightning state dictionary")
    state = checkpoint["state_dict"]
    tensors = {
        # Break tied-storage aliases (Qwen ties lm_head to token embeddings).
        # GGUF records both named tensors explicitly and the normal converter
        # must never depend on pickle/model-class knowledge to reconstruct it.
        str(name): tensor.detach().cpu().contiguous().clone()
        for name, tensor in state.items() if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise ValueError(f"{path} contains no tensors")
    return tensors, plain(checkpoint.get("hyper_parameters", {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenrig", type=Path, required=True)
    parser.add_argument("--skin-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenrig, tokenrig_config = extract(args.tokenrig)
    skin_vae, skin_vae_config = extract(args.skin_vae)
    save_file(tokenrig, args.output / "tokenrig.safetensors")
    save_file(skin_vae, args.output / "skin-vae.safetensors")
    config = {"tokenrig": tokenrig_config, "skin_vae": skin_vae_config}
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"extracted {len(tokenrig)} TokenRig and {len(skin_vae)} SkinVAE tensors")


if __name__ == "__main__":
    main()
