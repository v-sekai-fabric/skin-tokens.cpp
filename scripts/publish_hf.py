#!/usr/bin/env python3
"""Validate and optionally publish the SkinTokens GGUF bundles.

The default is a local dry run: it performs no network requests and prints the
exact repository, paths, sizes, and SHA-256 values. Repository creation and
uploads require both --upload and --confirm-upstream-license.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "LocalAI-io/SkinTokens-GGUF"
SOURCE_REPOSITORY = "VAST-AI/SkinTokens"
SOURCE_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
SOURCE_CODE_REVISION = "273b691d35989d71cd17ff2895fdc735097b92d1"
SOURCE_HASHES = {
    "tokenrig": "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
    "skin_vae": "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
}
COMPONENTS = ("mesh-encoder.gguf", "tokenrig.gguf", "skin-vae.gguf")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def collect(models: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for precision, directory in (("F16", "skintokens-f16"), ("F32", "skintokens-f32")):
        source = models / directory
        for name in COMPONENTS:
            path = source / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty GGUF: {path}")
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError(f"invalid GGUF magic: {path}")
            files.append((path, f"{precision}/{name}"))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument("--upload", action="store_true",
                        help="create/update the Hugging Face model repository")
    parser.add_argument("--confirm-upstream-license", action="store_true",
                        help="required with --upload; confirms authority to redistribute the MIT weights")
    args = parser.parse_args()

    card = ROOT / "scripts/hf/SkinTokens-GGUF/README.md"
    license_file = ROOT / "scripts/hf/SkinTokens-GGUF/LICENSE"
    try:
        if not card.is_file() or not license_file.is_file():
            raise ValueError("version-controlled model card or license is missing")
        files = collect(args.models)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    entries = [
        {"path": destination, "bytes": source.stat().st_size, "sha256": digest(source)}
        for source, destination in files
    ]
    manifest = {
        "format": "skintokens-gguf-manifest-v1",
        "repository": args.repo,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_code_revision": SOURCE_CODE_REVISION,
        "source_checkpoint_sha256": SOURCE_HASHES,
        "files": entries,
    }
    sums = "".join(f"{item['sha256']}  {item['path']}\n" for item in entries)

    print(f"repo:  https://huggingface.co/{args.repo}")
    print(f"base:  https://huggingface.co/{SOURCE_REPOSITORY}")
    print(f"files: {len(entries)} GGUFs, {sum(item['bytes'] for item in entries) / 1e9:.2f} GB")
    for item in entries:
        print(f"  {item['sha256']}  {item['bytes']:>12}  {item['path']}")
    if not args.upload:
        print("\n[dry-run] nothing created or uploaded.")
        return 0
    if not args.confirm_upstream_license:
        print("error: --upload requires --confirm-upstream-license", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    uploads = [(card, "README.md"), (license_file, "LICENSE"), *files]
    for source, destination in uploads:
        print(f"uploading {destination} ...", flush=True)
        api.upload_file(path_or_fileobj=str(source), path_in_repo=destination,
                        repo_id=args.repo, repo_type="model",
                        commit_message=f"Add {destination}")
    for payload, destination in (
        (json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", "MANIFEST.json"),
        (sums.encode(), "SHA256SUMS"),
    ):
        api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=destination,
                        repo_id=args.repo, repo_type="model",
                        commit_message=f"Add {destination}")
    print(f"done -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
