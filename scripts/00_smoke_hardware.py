"""Phase 0 smoke test: prove every precision loads, generates and reports memory.

For each precision: verify the model file against its frozen SHA256, load it,
time a 256-token generation, report VRAM, and check whether greedy decoding is
byte-identical across repeated runs. Also reports the measured prompt length of
each systems-suite bucket. Output is pasted into the README hardware sheet.
Writes nothing to results/.
"""

import logging
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import load_systems_prompts, wrap_llama32_user
from faithful_compression.utils.hashing import verify_model
from faithful_compression.utils.logging import configure_logging

CONFIGS_DIR = Path("configs")
SYSTEMS_PROMPTS = Path("data/systems/prompts.jsonl")
SMOKE_PROMPT = wrap_llama32_user(
    "Write a detailed explanation of how a CPU cache hierarchy works, covering L1, L2 "
    "and L3 caches, cache lines, and what happens on a cache miss."
)


def load_yaml(name: str) -> dict:
    """Load one config file from configs/."""
    return yaml.safe_load((CONFIGS_DIR / name).read_text(encoding="utf-8"))


def main() -> None:
    """Run the smoke test across all three precisions."""
    configure_logging()
    logging.getLogger("faithful_compression.backends.llama_cpp_backend").setLevel(logging.DEBUG)

    hardware = load_yaml("hardware.yaml")
    model_cfg = load_yaml("model_primary.yaml")
    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    print(f"backend: {backend.backend_info()}")

    config = GenConfig(temperature=0.0, max_tokens=256, seed=None)
    model_dir = Path(model_cfg["model_dir"])
    for index, (precision, entry) in enumerate(model_cfg["precisions"].items()):
        path = model_dir / entry["filename"]
        print(f"\n=== {precision}: {entry['filename']} ===")
        verify_model(path, entry["sha256"])
        print("sha256: matches frozen value")

        backend.load(str(path), hardware["ctx_len"], hardware["n_gpu_layers"])
        try:
            backend.generate(SMOKE_PROMPT, config)
            result = backend.generate(SMOKE_PROMPT, config)
            repeats = [result.text] + [backend.generate(SMOKE_PROMPT, config).text for _ in "ab"]
            memory = backend.memory_report()
            file_mb = path.stat().st_size / (1024 * 1024)

            print(f"load_time_s: {backend.load_time_s:.2f}")
            print(
                f"generated {result.n_tokens} tokens | decode {result.decode_tok_s:.1f} tok/s | "
                f"ttft {result.ttft_ms:.1f} ms | wall {result.wall_time_ms:.1f} ms | "
                f"prompt {result.prompt_tokens} tokens"
            )
            print(
                f"VRAM MiB: weights {memory.weights_vram_mb} | kv {memory.kv_cache_vram_mb} | "
                f"compute {memory.compute_vram_mb} | file {file_mb:.1f} | "
                f"weights/file {(memory.weights_vram_mb or 0) / file_mb:.3f}"
            )
            print(f"peak RSS MiB: {memory.peak_rss_mb:.1f}")
            print(f"determinism: {'identical' if len(set(repeats)) == 1 else 'divergent'}")
            print(f"sample: {result.text[:160]!r}")

            if index == 0:
                one_token = GenConfig(temperature=0.0, max_tokens=1, seed=None)
                buckets: dict[str, list[int]] = {}
                for row in load_systems_prompts(SYSTEMS_PROMPTS):
                    tokens = backend.generate(row["prompt"], one_token).prompt_tokens
                    buckets.setdefault(row["length_bucket"], []).append(tokens)
                for bucket, counts in buckets.items():
                    print(
                        f"prompt tokens [{bucket}]: mean {sum(counts) / len(counts):.0f}, "
                        f"min {min(counts)}, max {max(counts)}"
                    )
        finally:
            backend.unload()


if __name__ == "__main__":
    main()
