"""Suite R: run GSM8K items 0-79 under three sampling seeds for every precision."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import load_gsm8k
from faithful_compression.eval.reliability import run_reliability
from faithful_compression.utils.hashing import verify_model
from faithful_compression.utils.logging import configure_logging
from faithful_compression.utils.manifest import (
    build_manifest_entry,
    new_run_id,
    write_manifest_entry,
)

CONFIGS_DIR = Path("configs")
RESULTS_DIR = Path("results/raw")


def load_yaml(configs_dir: Path, name: str) -> dict:
    """Load one YAML config file."""
    return yaml.safe_load((configs_dir / name).read_text(encoding="utf-8"))


def main(configs_dir: Path = CONFIGS_DIR, results_dir: Path = RESULTS_DIR) -> None:
    """Run Suite R across all precisions and seeds.

    Each model is loaded once and runs every seed before the next model loads.

    Args:
        configs_dir: Directory holding the YAML configs.
        results_dir: Directory receiving the raw JSONL and the run manifest.
    """
    configure_logging()
    hardware = load_yaml(configs_dir, "hardware.yaml")
    model_cfg = load_yaml(configs_dir, "model_primary.yaml")
    suite = load_yaml(configs_dir, "suite_reliability.yaml")
    sampled = load_yaml(configs_dir, "gen_sampled.yaml")
    if sampled.pop("seeds") != suite["seeds"]:
        raise ValueError("seeds differ between gen_sampled.yaml and suite_reliability.yaml")

    items = load_gsm8k(
        suite["dataset"], suite["config"], suite["split"], suite["n_items"], suite["start_index"]
    )
    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    model_dir = Path(model_cfg["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for precision, entry in model_cfg["precisions"].items():
        path = model_dir / entry["filename"]
        verify_model(path, entry["sha256"])
        backend.load(str(path), hardware["ctx_len"], hardware["n_gpu_layers"])
        try:
            for seed in suite["seeds"]:
                gen_config = GenConfig(**sampled, seed=seed)
                manifest = build_manifest_entry(
                    suite="reliability",
                    precision=precision,
                    model_file=entry["filename"],
                    model_sha256=entry["sha256"],
                    gen_config=gen_config,
                    hardware=hardware,
                    backend_info=backend.backend_info(),
                    prompt_texts=[item["prompt"] for item in items],
                    run_id=new_run_id(),
                )
                rows = run_reliability(backend, items, precision, manifest["run_id"], gen_config)
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                out = results_dir / f"reliability_{precision}_{seed}_{stamp}.jsonl"
                out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                write_manifest_entry(results_dir / "run_manifest.jsonl", manifest)
                print(f"{precision} seed {seed}: {len(rows)} rows -> {out}")
        finally:
            backend.unload()


if __name__ == "__main__":
    main()
