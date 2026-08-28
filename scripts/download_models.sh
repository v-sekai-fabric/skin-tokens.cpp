#!/usr/bin/env bash
set -euo pipefail

repo="VAST-AI/SkinTokens"
revision="79736cad0fd84de384d5eede659b4ebd24effe33"
destination="${1:-checkpoints}"

command -v hf >/dev/null 2>&1 || {
  echo "hf CLI is required (available in 'nix develop')" >&2
  exit 1
}

mkdir -p "$destination/.download"
hf download "$repo" \
  experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt \
  experiments/skin_vae_2_10_32768/last.ckpt \
  --revision "$revision" --local-dir "$destination/.download"

install -m 0644 \
  "$destination/.download/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt" \
  "$destination/grpo_1400.ckpt"
install -m 0644 \
  "$destination/.download/experiments/skin_vae_2_10_32768/last.ckpt" \
  "$destination/last.ckpt"

printf '%s  %s\n' \
  f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692 "$destination/grpo_1400.ckpt" \
  4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4 "$destination/last.ckpt" \
  | sha256sum --check --strict

echo "downloaded verified SkinTokens checkpoints at revision $revision"
