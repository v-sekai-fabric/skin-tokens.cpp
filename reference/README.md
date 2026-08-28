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
full F32 condition encoder and decoder, then calls the official upstream
`Asset.from_data` interpolation and optional `voxel_skin` implementation. The
fixture retains raw learned and postprocessed stages separately, plus final
top-four influences and animated vertices. It also writes a self-contained
fixture for running the exact production C++ path on both backends:

```sh
build/debug/bin/skintokens-full-binding-parity \
  models/skintokens-f32 generated/animation-parity cpu
build/debug/bin/skintokens-full-binding-parity \
  models/skintokens-f32 generated/animation-parity vulkan
```

To register those same full-model checks with CTest, configure with
`SKINTOKENS_PARITY_MODEL_DIR` and `SKINTOKENS_PARITY_FIXTURE_DIR`. They are not
registered in weight-free builds, so ordinary unit tests never download model
files.

The final comparison asserts continuous dense/sparse error, selected-bone
support, and deformed vertices from the newly computed C++ binding. The
released demo leaves `voxel_skin` disabled by default; it remains an explicit
diagnostic rather than being conflated with normal raw learned export.
The released attention processor reshapes each combined projection into heads
before splitting K/V or Q/K/V within each head. The reference captures and
GGML graph preserve that ordering exactly; splitting the flat projection into
full-width Q/K/V blocks is not equivalent.

`capture_upstream_binding.py` is a separate behavioral diagnostic that imports
the official Lightning classes and runs their normal BF16 path. Do not use its
`--precision float32` mode as the clean F32 target: `TokenRig.__init__` first
rounds the PMPE frequency/phase buffers to BF16, and later promotion cannot
recover them. The direct safetensor capture constructs those buffers in F32,
matching this project's explicitly F32 model/activation milestone.

## Official giraffe exemplar

`capture_official_example.py` runs the only mesh exemplar checked into the
upstream repository through its Blender loader, predict transform, full
TokenRig generation, SkinVAE decoder, and exporter. It records both the normal
released BF16 result and a clean F32 decode of the identical condition and FSQ
codes. The capture also retains the optional voxel graph as a diagnostic.

```sh
docker run --rm --device nvidia.com/gpu=all \
  --user "$(id -u):$(id -g)" --workdir /work/reference/upstream \
  -e PYTHONPATH=/work/reference:/work/reference/upstream \
  -v "$PWD:/work" --entrypoint python skintokens-reference:2.7 \
  /work/reference/capture_official_example.py \
  --upstream-root /work/reference/upstream \
  --checkpoint /work/checkpoints/grpo_1400.ckpt \
  --skin-checkpoint /work/checkpoints/last.ckpt \
  --input /work/reference/upstream/examples/giraffe.glb \
  --output /work/generated/giraffe-upstream-full --seed 0

build/debug/bin/skintokens-giraffe-parity \
  models/skintokens-f32 generated/giraffe-upstream-full cpu
build/debug/bin/skintokens-giraffe-parity \
  models/skintokens-f32 generated/giraffe-upstream-full vulkan
```

Acceptance covers the learned decoder, raw dense vertex weights, and raw
exported top-four weights—the upstream demo's default endpoint. The BF16
production result and opt-in `voxel_skin` endpoint are printed separately so
arithmetic-mode or heuristic drift remains visible. In particular, passing the
F32 milestone does not claim that released CUDA BF16 arithmetic is identical;
the tool reports that behavioral delta explicitly.
