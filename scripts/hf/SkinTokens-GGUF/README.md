---
license: mit
base_model: VAST-AI/SkinTokens
base_model_relation: quantized
tags:
  - gguf
  - ggml
  - skintokens
  - rigging
  - animation
---

# SkinTokens — GGUF

GGUF F16 and F32 conversions of
[`VAST-AI/SkinTokens`](https://huggingface.co/VAST-AI/SkinTokens) for
[`skin-tokens.cpp`](https://github.com/localai-org/skin-tokens.cpp), a C++23
GGML CPU/Vulkan implementation of TokenRig and SkinVAE. These files are a
format/precision conversion of the released model; they were not retrained.

The source repository is the authoritative description of supported inputs,
parity status, runtime behavior, and known limitations. The original model card
remains authoritative for intended use, training data, research claims, and
citation.

## Files

Each directory is a complete bundle. Keep its three files together because the
runtime checks their embedded source identities at load time.

| Bundle | File | Component | Size |
|---|---|---|---:|
| `F16` | `mesh-encoder.gguf` | Michelangelo point encoder | 57.8 MB |
| `F16` | `tokenrig.gguf` | Qwen3-0.6B TokenRig policy | 948.6 MB |
| `F16` | `skin-vae.gguf` | SkinTokens condition encoder and decoder | 244.0 MB |
| `F32` | `mesh-encoder.gguf` | Michelangelo point encoder | 115.4 MB |
| `F32` | `tokenrig.gguf` | Qwen3-0.6B TokenRig policy | 1,778.5 MB |
| `F32` | `skin-vae.gguf` | SkinTokens condition encoder and decoder | 487.2 MB |

F16 is the normal distribution. F32 is retained as the numerical reference
bundle used for CPU/Vulkan parity work.

## Provenance

- Source weights: [`VAST-AI/SkinTokens`](https://huggingface.co/VAST-AI/SkinTokens),
  revision `79736cad0fd84de384d5eede659b4ebd24effe33`.
- TokenRig checkpoint SHA-256:
  `f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692`.
- SkinVAE checkpoint SHA-256:
  `4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4`.
- Conversion/reference code: `VAST-AI-Research/SkinTokens` revision
  `273b691d35989d71cd17ff2895fdc735097b92d1`.
- Unsafe Lightning checkpoints were opened only in the pinned disposable
  reference container. The normal converter accepts verified safetensors only.
- `MANIFEST.json` and `SHA256SUMS` record the exact published GGUF identities.

## License and citation

The source repository and model card publish the code and weights under MIT;
that license is included here. Please cite the original SkinTokens work as
specified on the [source model card](https://huggingface.co/VAST-AI/SkinTokens).
