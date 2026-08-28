# skin-tokens.cpp

A C++23/GGML port of
[SkinTokens / TokenRig](https://github.com/VAST-AI-Research/SkinTokens) for
automatic skeleton and skin-weight generation on CPU or Vulkan. The primary
integration takes a mesh exported by trellis2cpp and an animated skeleton GLB
exported by Kimodo. The runtime fits Kimodo's humanoid hierarchy to the mesh,
TokenRig predicts its skin codes, and SkinVAE decodes dense weights before a
portable animated GLB is written.

This is active parity work. Safe checkpoint extraction, checked GGUF loading,
CPU/Vulkan backend selection, TokenRig's complete skeleton grammar and
detokenizer, the complete
28-layer Qwen3 block stack, Kimodo animation import, and skinned GLB export are
implemented. The F32 Qwen stack matches its pinned PyTorch reference to
8.1e-6 relative L2 on CPU; on Vulkan the final hidden state is `7.1e-4` and
logits are `1.4e-3` relative L2.

The learned binding path runs the released Michelangelo mesh encoder, Qwen
TokenRig policy, FSQ code expansion, condition encoder, and chunked SkinVAE
decoder through GGML. It samples the same 54,000 surface points as upstream
and caps dense decoder queries at 16K vertices per graph. `--supplied-skeleton` uses the
input motion hierarchy and is the recommended Kimodo integration. Omitting it
enables upstream's unconstrained skeleton generation, which remains
experimental because the released policy can produce unsuitable topologies.
`--geometric` is an explicit non-learned diagnostic.

The Michelangelo encoder finishes at `3.8e-7` relative L2 on CPU and `1.8e-4`
on Vulkan. A constrained 120-step fixture checks the production Qwen prefix and
greedy policy decision at every step; Vulkan reproduces every reference code.
The full 54K-point SkinVAE and final learned binding are checked independently:
CPU reaches `1.0e-5` decoder relative L2 with identical top-four bones on every
vertex. The representative Vulkan fixture reaches `9.1e-4`, with 99.82% of
top-four bone slots identical; its five-frame animated-vertex check is
`1.65e-3` relative L2.

## Build on Linux

Install CMake 3.25+, Ninja, a C++23 compiler, Vulkan development files,
nlohmann-json, and Git. Then initialize the pinned GGML submodule and build:

```sh
git submodule update --init --recursive
cmake --preset release
cmake --build --preset release -j
./build/release/bin/skintokens-cli inspect models/skintokens-f32
```

Use `--device vulkan`, `--device cpu`, or `--device auto`. Nix is optional; it
provides the same tools without installing them globally:

```sh
nix develop
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug
```

ASan/UBSan remains the default diagnostic lane while model sizes permit it:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan -j
ctest --preset asan-ubsan
```

## Download and convert

The two official checkpoints are MIT-labelled but use Lightning pickle
containers. Downloads are explicit, hash-checked, and excluded from git:

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

## Intended workflow

```sh
trellis2cpp/build/examples/mesh2glb \
  trellis-mesh.t2mesh trellis-remeshed.glb 2048 \
  --print 0.5 0.0166667 --quad 20000
./build/release/bin/skintokens-cli bind \
  models/skintokens-f32 \
  trellis-remeshed.glb kimodo-animation.glb character-animated.glb \
  --device vulkan --supplied-skeleton
```

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
source vertices with the same inverse-distance 8-neighbour interpolation.
Exactly as in the upstream Blender exporter, the greatest four interpolated
learned weights become the portable glTF influence slots and are then
normalized. No geometric or bone-locality heuristic alters learned output.
Generation currently recomputes the Qwen graph without a KV cache and may
take several minutes.

## Demo

The dependency-free Go/WebGL demo accepts uploaded GLBs and can scan local
Kimodo and trellis2cpp galleries. It keeps a persistent history, runs one
bounded worker, previews the GPU-skinned animation beside its synchronized
driving skeleton, and offers the output GLB for download or direct reuse in
another Three.js application.

```sh
cd demo
go run . \
  --listen 0.0.0.0:8095 \
  --cli ../build/release/bin/skintokens-cli \
  --model ../models/skintokens-f32 \
  --kimodo ../../kimodo.cpp/demo-output \
  --trellis ../../trellis2cpp/generations \
  --trellis-mesh2glb ../../trellis2cpp/build/examples/mesh2glb
```

Open `http://localhost:8095`. Paths and the listen address are flags; no local
machine paths are compiled into the C++ library or web server.

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
