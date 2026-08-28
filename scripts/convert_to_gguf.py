#!/usr/bin/env python3
"""Convert verified SkinTokens safetensors intermediates to GGUF v3.

This program deliberately accepts no PyTorch checkpoint format. The official
Lightning `.ckpt` files are pickle containers and must first be extracted in
the pinned, disposable reference container. Conversion itself therefore never
executes checkpoint-controlled Python objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
F32 = 0
F16 = 1
UINT32 = 4
STRING = 8

UPSTREAM_REVISION = "273b691d35989d71cd17ff2895fdc735097b92d1"


def encoded(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def kv_string(key: str, value: str) -> bytes:
    return encoded(key) + struct.pack("<I", STRING) + encoded(value)


def kv_u32(key: str, value: int) -> bytes:
    return encoded(key) + struct.pack("<II", UINT32, value)


def aligned(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def compact_name(name: str) -> str:
    replacements = (
        ("transformer.model.layers.", "llm.l."),
        ("transformer.model.embed_tokens.", "llm.tok."),
        ("transformer.model.norm.", "llm.norm."),
        ("transformer.lm_head.", "llm.out."),
        ("mesh_encoder.encoder.", "mesh.enc."),
        ("mesh_encoder.", "mesh."),
        ("output_proj.", "mesh.out."),
        ("vae.model.", "vae."),
        ("vae.up_perceiver.", "vae.up."),
        ("vae.down_perceiver.", "vae.down."),
        ("self_attn.", "attn."),
        ("cross_attn.", "xattn."),
        ("input_layernorm.", "an."),
        ("post_attention_layernorm.", "fn."),
        ("q_proj.", "q."), ("k_proj.", "k."), ("v_proj.", "v."),
        ("o_proj.", "o."), ("gate_proj.", "g."), ("up_proj.", "u."),
        ("down_proj.", "d."), ("weight", "w"), ("bias", "b"),
    )
    output = name
    for old, new in replacements:
        output = output.replace(old, new)
    if len(output.encode("utf-8")) >= 64:
        suffix = hashlib.sha256(name.encode()).hexdigest()[:12]
        output = output.encode("utf-8")[:48].decode("utf-8", "ignore") + "." + suffix
    return output


def component_for(name: str) -> str | None:
    if name.startswith(("mesh_encoder.", "output_proj.")):
        return "mesh-encoder"
    if name.startswith("transformer."):
        return "tokenrig"
    if name.startswith("vae."):
        return "skin-vae"
    return None


def choose_type(name: str, shape: tuple[int, ...], ftype: int) -> int:
    if ftype == 0:
        return F32
    sensitive = any(part in name for part in ("norm", "embed", "query", "q_vec", "FSQ"))
    return F16 if len(shape) >= 2 and not sensitive else F32


@dataclass
class Tensor:
    name: str
    value: object
    kind: int
    dimensions: tuple[int, ...]
    offset: int

    @property
    def data(self) -> bytes:
        import torch
        dtype = torch.float32 if self.kind == F32 else torch.float16
        return self.value.to(dtype=dtype).contiguous().numpy().tobytes(order="C")


def load_tensors(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    # PyTorch is used only as the safetensors numeric backend. Unlike a .ckpt
    # load this executes no file-controlled objects and handles BF16 exactly.
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in source.keys():
            value = source.get_tensor(name)
            if not value.is_floating_point():
                raise ValueError(f"unsupported tensor dtype for {name}: {value.dtype}")
            result[name] = value
    if not result:
        raise ValueError(f"{path} contains no tensors")
    return result


def write_component(path: Path, component: str, values: list[tuple[str, object]],
                    config_json: str, tokenrig_hash: str, vae_hash: str, ftype: int) -> None:
    tensors: list[Tensor] = []
    offset = 0
    names: set[str] = set()
    for original, value in sorted(values):
        name = compact_name(original)
        if name in names:
            raise ValueError(f"compacted tensor name collision: {name}")
        names.add(name)
        shape = tuple(int(v) for v in value.shape) or (1,)
        kind = choose_type(original, shape, ftype)
        tensor = Tensor(name, value, kind, tuple(reversed(shape)), offset)
        tensors.append(tensor)
        offset = aligned(offset + len(tensor.data))

    metadata = [
        kv_string("general.architecture", "skintokens"),
        kv_string("general.name", "SkinTokens TokenRig"),
        kv_u32("general.alignment", ALIGNMENT),
        kv_u32("general.file_type", ftype),
        kv_u32("skintokens.format_version", 1),
        kv_string("skintokens.component", component),
        kv_string("skintokens.upstream_revision", UPSTREAM_REVISION),
        kv_string("skintokens.tokenrig_sha256", tokenrig_hash),
        kv_string("skintokens.skin_vae_sha256", vae_hash),
        kv_string("skintokens.config_json", config_json),
    ]
    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
    for item in metadata:
        header += item
    for tensor in tensors:
        header += encoded(tensor.name)
        header += struct.pack("<I", len(tensor.dimensions))
        header += struct.pack("<" + "Q" * len(tensor.dimensions), *tensor.dimensions)
        header += struct.pack("<IQ", tensor.kind, tensor.offset)
    header += bytes(aligned(len(header)) - len(header))

    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(header)
        position = 0
        for tensor in tensors:
            if tensor.offset < position:
                raise AssertionError("overlapping tensor offsets")
            stream.write(bytes(tensor.offset - position))
            payload = tensor.data
            stream.write(payload)
            position = tensor.offset + len(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    print(f"wrote {path} ({len(tensors)} tensors, {path.stat().st_size / (1 << 20):.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenrig", type=Path, required=True,
                        help="verified TokenRig safetensors intermediate")
    parser.add_argument("--skin-vae", type=Path, required=True,
                        help="verified SkinVAE safetensors intermediate")
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON emitted by the trusted extractor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ftype", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_json = json.dumps(config, separators=(",", ":"), sort_keys=True)
    tokenrig = load_tensors(args.tokenrig)
    vae = load_tensors(args.skin_vae)
    # The TokenRig state includes its frozen VAE. Prefer those exact tensors;
    # retain the standalone VAE only as the verified construction/source lane.
    combined = dict(tokenrig)
    if not any(name.startswith("vae.") for name in combined):
        combined.update({"vae." + name: value for name, value in vae.items()})

    grouped: dict[str, list[tuple[str, object]]] = {
        "mesh-encoder": [], "tokenrig": [], "skin-vae": []
    }
    ignored: list[str] = []
    for name, value in combined.items():
        component = component_for(name)
        if component is None:
            ignored.append(name)
        else:
            grouped[component].append((name, value))
    for component, values in grouped.items():
        if not values:
            raise ValueError(f"no tensors found for required component {component}")
    if ignored:
        raise ValueError("unclassified TokenRig tensors: " + ", ".join(sorted(ignored)[:20]))

    tokenrig_hash = sha256(args.tokenrig)
    vae_hash = sha256(args.skin_vae)
    for component, values in grouped.items():
        write_component(args.output / f"{component}.gguf", component, values,
                        config_json, tokenrig_hash, vae_hash, args.ftype)


if __name__ == "__main__":
    main()
