# Trusted reference lane

The official SkinTokens checkpoints are Lightning pickle containers. They are
never opened by the host converter. Verify the pinned hashes, then extract them
once inside this disposable container:

```sh
docker build -t skintokens-reference:2.7 reference
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  skintokens-reference:2.7 \
  --tokenrig /work/checkpoints/grpo_1400.ckpt \
  --skin-vae /work/checkpoints/last.ckpt \
  --output /work/generated/safe
```

`scripts/convert_to_gguf.py` accepts only the resulting safetensors and JSON.
Layer fixtures use the same pinned image plus the upstream source revision
recorded in the model bundle.

## Parity captures

The small deterministic captures exercise one layer or generation step:

- `capture_mesh_encoder.py`
- `capture_qwen_layer.py`
- `capture_skin_vae.py`
- `capture_token_generation.py`

The integration captures run the released networks against data emitted by the
C++ runtime. Set `SKINTOKENS_DUMP_MESH_INPUT_PREFIX` or
`SKINTOKENS_DUMP_VAE_TRACE_PREFIX` while running `skintokens-cli`, then evaluate
the resulting files with `capture_mesh_encoder_full.py` or
`capture_skin_condition_full.py`. This checks the complete 54,000-point
conditioning paths without relying on independently sampled point clouds.

`capture_unconstrained_generation.py` captures the upstream Qwen policy. Pass a
complete token stream with `--prefix` to exercise the recommended constrained
skin-generation path for a supplied skeleton. The script records both tokens
and initial logits so policy parity can be separated from framework-specific
random sampling.

Final animated-vertex parity is deliberately separate from network tensor
parity. Capture five exact keyframes from an exported animation, then compare
the independent NumPy glTF/LBS result with the C++ evaluator:

```sh
python3 reference/capture_animated_vertices.py animation.glb generated/animation-parity
build/debug/bin/skintokens-animation-parity generated/animation-parity
```

The fixture includes all mesh positions, four skin influences, bind pose,
local animation tracks and reference vertices. The comparison reports maximum
absolute error, relative L2 error and the worst vertex for every selected
frame. It does not use renderer output, so camera and rasterization cannot hide
a skinning transform error.

For an actual learned-binding comparison, generate the GLB while setting
`SKINTOKENS_DUMP_VAE_TRACE_PREFIX`, `SKINTOKENS_DUMP_TOKEN_TRACE_PREFIX`, and
`SKINTOKENS_DUMP_BINDING_TRACE_PREFIX`,
make the fixture above, and replace its expected influences/vertices with a
direct PyTorch evaluation of the verified SkinVAE safetensors:

```sh
python3 reference/capture_binding_vertices.py \
  --weights generated/safe/tokenrig.safetensors \
  --fixture generated/animation-parity \
  --vae-trace-prefix generated/trace/vae \
  --token-trace-prefix generated/trace/tokens \
  --cpp-binding-trace-prefix generated/trace/binding
build/debug/bin/skintokens-animation-parity generated/animation-parity
```

This layered full-model test holds the 54,000 sampled points, condition query
indices and generated FSQ codes constant. Python independently evaluates the
full F32 condition encoder and decoder, upstream inverse-distance 8-neighbour
integration, learned-weight top-four export and animated vertices. It also
writes a self-contained fixture for running the exact production C++ path on
both backends:

```sh
build/debug/bin/skintokens-full-binding-parity \
  models/skintokens-f32 generated/animation-parity cpu
build/debug/bin/skintokens-full-binding-parity \
  models/skintokens-f32 generated/animation-parity vulkan
```

The final comparison asserts both continuous error and selected-bone support.
The released attention processor uses an unusual concatenate-then-head-split
Q/K/V layout; the reference captures and GGML graph preserve that layout
exactly rather than using a conventional independent head reshape.

`capture_upstream_binding.py` is a separate behavioral diagnostic that imports
the official Lightning classes and runs their normal BF16 path. Do not use its
`--precision float32` mode as the clean F32 target: `TokenRig.__init__` first
rounds the PMPE frequency/phase buffers to BF16, and later promotion cannot
recover them. The direct safetensor capture constructs those buffers in F32,
matching this project's explicitly F32 model/activation milestone.
