# Model inputs

Downloaded checkpoints and converted GGUF files are intentionally excluded
from git. `scripts/download_models.sh` fetches the pinned official inputs and
verifies their recorded revision and SHA-256 digests. The converter embeds the
same identities in every output component.

The release publisher is dry-run by default:

```sh
python3 scripts/publish_hf.py
```

It validates the complete F16 and F32 bundles under `models/`, checks GGUF
magic, and prints every size and SHA-256. It targets
`LocalAI-io/SkinTokens-GGUF`, with `VAST-AI/SkinTokens` recorded as the base
model. Repository creation requires the explicit `--upload` and
`--confirm-upstream-license` flags.
