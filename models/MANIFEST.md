# Model inputs

Downloaded checkpoints and converted GGUF files are intentionally excluded
from git. `scripts/download_models.sh` fetches the pinned official inputs and
verifies their recorded revision and SHA-256 digests. The converter embeds the
same identities in every output component.
