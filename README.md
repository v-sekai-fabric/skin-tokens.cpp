# skin-tokens.cpp

A C++23/GGML port of
[SkinTokens / TokenRig](https://github.com/VAST-AI-Research/SkinTokens) for
automatic skeleton and skin-weight generation on CPU or Vulkan.

Skin weights say how strongly every mesh vertex follows each bone. Without
them, moving a skeleton does not deform the character surface correctly.
SkinTokens takes a static mesh, predicts a suitable skeleton and its vertex
weights, and writes a portable rigged GLB. 

## Build and install on Linux

Install Git, CMake 3.25 or newer, Ninja, a C++23 compiler,
`nlohmann-json`, and Vulkan headers, loader, and shader compiler. Vulkan can be
disabled with `-DSKINTOKENS_ENABLE_VULKAN=OFF`. From a source checkout:

```sh
git submodule update --init --recursive
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j
cmake --install build/release --prefix ./dist
```

The install contains the shared library, C and C++ headers, CLI, GGML runtime,
and enabled dynamic backends. To test from the build tree:

```sh
ctest --test-dir build/release --output-on-failure
./build/release/bin/skintokens-cli inspect models/SkinTokens-GGUF/F16
```

Nix is optional. It supplies the build tools and Vulkan development packages,
but does not replace the normal CMake build:

```sh
nix develop
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug
```

The sanitizer build is the normal development lane:

```sh
nix develop
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan -j
ctest --preset asan-ubsan
```

## Download the weights

Downloading the converted F16 GGUF bundle is the recommended setup. With the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download LocalAI-io/SkinTokens-GGUF \
  --include "F16/*" \
  --local-dir models/SkinTokens-GGUF
```

Use `models/SkinTokens-GGUF/F16` as `MODEL_DIR` in the commands below. The
repository also contains an F32 bundle for numerical parity work; normal
inference should use F16.

## Rig an arbitrary mesh

The standalone path needs only a static `.glb` mesh. It generates the skeleton
and learned skin weights, then stores a one-frame rest pose in the output so it
opens as a conventional skinned glTF asset:

```sh
./build/release/bin/skintokens-cli rig \
  models/SkinTokens-GGUF/F16 \
  character.glb character-rigged.glb \
  --device vulkan --postprocess
```

The model accepts arbitrary triangle meshes, although—as with any learned
rigging model—results are strongest on shapes resembling its training
distribution. `--postprocess` enables the upstream surface-locality heuristic;
omit it to preserve the raw learned weights exactly.

## Status

Safe checkpoint extraction, checked GGUF loading, CPU/Vulkan backend selection,
the complete TokenRig grammar and Qwen3 stack, Kimodo animation import, and
skinned GLB export are implemented. The F32 Qwen stack reaches `8.1e-6`
relative L2 on CPU; Vulkan reaches `7.1e-4` for the final hidden state and
`1.4e-3` for logits.

The learned binding path runs the released Michelangelo mesh encoder, Qwen
TokenRig policy, FSQ code expansion, condition encoder, and chunked SkinVAE
decoder through GGML. It samples the same 54,000 surface points as upstream
and caps dense decoder queries at 16K vertices per graph. `--supplied-skeleton` uses the
input motion hierarchy and is the recommended Kimodo integration. Omitting it
enables upstream's unconstrained skeleton generation, which remains
experimental because the released policy can produce unsuitable topologies.
`--geometric` is an explicit non-learned diagnostic.

Upstream's existing-skeleton interface is deliberately generic: `--use_skeleton`
retains an armature found in the input and generates skin only. It does not
promise that every arbitrary hierarchy is in-distribution or prescribe a
retargeting method. Imported GLBs are tokenized as generic articulations. The
released checkpoint stores `order_config`, while `TokenizerPart.parse` reads
`order`, so its executable imported-GLB path emits no spring/part token and
does not activate a hidden Mixamo-specific inference mode from joint names.

The full 54K-point Michelangelo encoder matches the corrected F32 PyTorch
reference at `6.5e-4` relative L2 on Vulkan. A constrained 120-step fixture
checks the production Qwen prefix and greedy policy decision at every step;
Vulkan reproduces every reference code.
The full 54K-point SkinVAE and final raw learned binding are checked
independently. CPU reaches `3.6e-6` decoder relative L2, identical top-four
bones on every vertex, and `1.4e-7` animated-vertex relative L2. The
representative Vulkan fixture reaches `9.1e-4` decoder relative L2, with
99.96% of final top-four slots shared and `5.2e-4` animated-vertex relative L2.
The optional `voxel_skin` heuristic is available with `--postprocess`, matching
the upstream demo's opt-in setting rather than being silently enabled.

Use `--device vulkan`, `--device cpu`, or `--device auto` at runtime.

## Convert the upstream checkpoints yourself

This is an advanced reproducibility path; most users should download the GGUF
bundle above. The two official checkpoints are MIT-labelled but use Lightning
pickle containers. Downloads are explicit, hash-checked, and excluded from
git:

```sh
./scripts/download_models.sh checkpoints
docker build -t skintokens-reference:2.7 reference
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  skintokens-reference:2.7 \
  --tokenrig /work/checkpoints/grpo_1400.ckpt \
  --skin-vae /work/checkpoints/last.ckpt \
  --output /work/generated/safe
python scripts/convert_to_gguf.py \
  --tokenrig generated/safe/tokenrig.safetensors \
  --skin-vae generated/safe/skin-vae.safetensors \
  --config generated/safe/config.json \
  --output models/skintokens-f32 --ftype 0
```

The normal converter accepts safetensors only. Every GGUF component records
the upstream revision and both intermediate SHA-256 identities, and model load
rejects mixed bundles.

## Optional end-to-end humanoid pipeline: Trellis2 + Kimodo

Trellis2 can generate and remesh a humanoid, Kimodo can generate its motion,
and skin-tokens.cpp can bind that motion hierarchy to the mesh. This is a
concrete full animated-asset pipeline, not a dependency or requirement of
SkinTokens:

```sh
trellis2cpp/build/examples/mesh2glb \
  trellis-mesh.t2mesh trellis-remeshed.glb 2048 \
  --print 0.5 0.0166667 --quad 20000
./build/release/bin/skintokens-cli bind \
  models/SkinTokens-GGUF/F16 \
  trellis-remeshed.glb kimodo-animation.glb character-animated.glb \
  --device vulkan --supplied-skeleton
```

For a controlled humanoid A/B test, the direct lane above conditions on fitted
SOMA30. The second lane builds the exact 52-joint order published in upstream's
`configs/skeleton/mixamo.yaml`, fits its rest positions from the same SOMA
anatomy, and uses an explicit semantic motion map:

```sh
./build/release/bin/skintokens-cli bind \
  models/SkinTokens-GGUF/F16 \
  character.glb kimodo-soma30.glb character-mixamo.glb \
  --device vulkan --supplied-skeleton --target-rig mixamo52

./build/release/bin/skintokens-cli retarget-check kimodo-soma30.glb
```

`retarget-check` rotates each mapped source joint by +/-30 degrees around all
three axes and compares corresponding FK trajectories in normalized body
space. It also reports full-clip joint-position and foot-velocity errors. This
tests retargeting independently of learned skin weights; the demo's SOMA30 /
Mixamo52 selector then tests both learned binding outcomes on identical inputs.

Both `.glb` and trellis2cpp's versioned `.t2mesh` are accepted as mesh input.
Skinning should use the manifold-wrapped, quad-remeshed Trellis export, not its dense raw
marching-cubes reconstruction: the latter is millions of triangles with
generation topology that is unsuitable for animation. The demo performs this
wrap and remesh automatically for `.t2mesh` sources. Direct quad remeshing is
not sufficient for raw reconstruction topology; Alpha Wrap must clean it first.
GLB scene-node transforms and vertex colours are preserved; atlas textures are
not yet round-tripped. The Kimodo rig is fitted to the mesh before
conditioning, while its animation and hierarchy are retained in the output.
SkinVAE predictions are decoded on the upstream 54K cloud and transferred to
source vertices with the same SciPy cKDTree inverse-distance 8-neighbour
interpolation, then selects and renormalizes the greatest four influences for
glTF. With `--postprocess`, the optional upstream heuristic first multiplies
those channels by its `voxel_skin` surface-geodesic matrix. The C++ diagnostic
retains Open3D voxel ordering and SciPy nearest-neighbour tie behaviour.
Generation currently recomputes the Qwen graph without a KV cache and may
take several minutes.

## C API

[`skintokens.h`](include/skintokens/skintokens.h) is a flat C11-compatible API.
It uses an opaque model handle, caller-owned fixed error buffers, explicit
defaults, and status returns; C++ exceptions never cross the ABI. Returned
backend and `last_error` strings are borrowed from the model and remain valid
until its next call or destruction.

```c
#include <skintokens/skintokens.h>

char error[512];
st_model *model = NULL;
st_runtime_options runtime = st_default_runtime_options();
runtime.device = ST_DEVICE_VULKAN;
if (st_model_load("models/SkinTokens-GGUF/F16", &runtime, &model,
                  error, sizeof(error)) != ST_OK) {
    /* report error */
}

st_generation_options generation = st_default_generation_options();
generation.target_rig = ST_TARGET_MIXAMO52;
generation.surface_postprocess = 1;
int learned = 0;
st_status result = st_bind_files(model, "character.glb", "motion.glb",
                                 "animated.glb", &generation, &learned,
                                 error, sizeof(error));
st_model_free(model);
```

`st_inspect_mesh_file` and `st_inspect_motion_glb_file` validate uploads without
loading GGUF weights. The public parsing and error-buffer surface is exercised
by the ASan/UBSan libFuzzer build; GGUF loading is intentionally kept outside
that fuzzer.

## Demo

The dependency-free Go/WebGL demo accepts uploaded GLBs and can scan local
Kimodo and trellis2cpp galleries. It keeps a persistent history, runs one
bounded worker, previews the GPU-skinned animation beside its synchronized
driving skeleton, and offers the output GLB for download or direct reuse in
another Three.js application.

```sh
cd demo
go run . \
  --listen 127.0.0.1:8095 \
  --cli ../build/release/bin/skintokens-cli \
  --model ../models/SkinTokens-GGUF/F16 \
  --kimodo /path/to/kimodo/gallery \
  --trellis /path/to/trellis/gallery \
  --trellis-mesh2glb /path/to/trellis2cpp/mesh2glb
```

Open `http://localhost:8095`. Paths and the listen address are flags; no local
machine paths are compiled into the C++ library or web server. The Kimodo,
Trellis, remesher, and optional giraffe-example paths default to disabled; pass
only the integrations available on the host.

## Reference parity

Capture operation boundaries from the safe checkpoint in the pinned reference
image, then compare every Qwen, mesh-encoder, SkinVAE, and generation boundary.
The unconstrained fixture additionally records the skeleton/skin switch and
upstream's unusual use of model EOS as the final FSQ code:

```sh
docker build -t skintokens-reference:2.7-parity reference
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  --entrypoint python skintokens-reference:2.7-parity \
  /work/reference/capture_qwen_layer.py \
  --weights /work/generated/safe/tokenrig.safetensors \
  --output /work/fixtures/reference-f32 --device cpu --dtype float32

./build/release/bin/skintokens-qwen-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-mesh-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-vae-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
./build/release/bin/skintokens-generation-parity \
  models/skintokens-f32 fixtures/reference-f32 cpu
```

Final binding acceptance captures normalized top-four weights and reference
deformed vertices. The full checker runs the production GGML decoder and
skinning evaluator in one process:

```sh
./build/release/bin/skintokens-full-binding-parity \
  models/skintokens-f32 fixtures/animation-parity cpu
./build/release/bin/skintokens-full-binding-parity \
  models/skintokens-f32 fixtures/animation-parity vulkan
```

For the strict Vulkan diagnostic, disable reduced-precision matrix paths:

```sh
GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 \
GGML_VK_DISABLE_COOPMAT2=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 \
  ./build/release/bin/skintokens-qwen-parity \
  models/skintokens-f32 fixtures/reference-f32 vulkan
```

## Upstream scope and licence

SkinTokens is a learned skin-weight representation. TokenRig is the complete
mesh-to-rig system: a Michelangelo point encoder, Qwen3-0.6B causal model, and
conditional FSQ-VAE decoder. This use is therefore within its intended scope,
not an attempt to treat SkinTokens as a motion generator.

The upstream repository and Hugging Face model card identify code and weights
as MIT. This project is Apache-2.0 and keeps upstream attribution in `NOTICE`.
