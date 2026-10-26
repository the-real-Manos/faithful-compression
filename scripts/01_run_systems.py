"""Suite S: run the systems suite for every precision and save raw results."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import load_systems_prompts
from faithful_compression.eval.systems import run_systems
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
    """Run Suite S across all precisions.

    Args:
        configs_dir: Directory holding the YAML configs.
        results_dir: Directory receiving the raw JSONL and the run manifest.
    """
    configure_logging()
    hardware = load_yaml(configs_dir, "hardware.yaml")
    model_cfg = load_yaml(configs_dir, "model_primary.yaml")
    suite = load_yaml(configs_dir, "suite_systems.yaml")
    gen_config = GenConfig(**load_yaml(configs_dir, "gen_greedy.yaml"))

    prompts = load_systems_prompts(Path(suite["prompts_file"]))
    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    model_dir = Path(model_cfg["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for precision, entry in model_cfg["precisions"].items():
        path = model_dir / entry["filename"]
        verify_model(path, entry["sha256"])
        manifest = build_manifest_entry(
            suite="systems",
            precision=precision,
            model_file=entry["filename"],
            model_sha256=entry["sha256"],
            gen_config=gen_config,
            hardware=hardware,
            backend_info=backend.backend_info(),
            prompt_texts=[p["prompt"] for p in prompts],
            run_id=new_run_id(),
        )

        backend.load(str(path), hardware["ctx_len"], hardware["n_gpu_layers"])
        try:
            rows = run_systems(
                backend,
                prompts,
                gen_config,
                warmup_runs=suite["warmup_runs"],
                measured_runs=suite["measured_runs"],
                precision=precision,
                run_id=manifest["run_id"],
                load_time_s=backend.load_time_s,
            )
        finally:
            backend.unload()

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = results_dir / f"systems_{precision}_{stamp}.jsonl"
        out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_manifest_entry(results_dir / "run_manifest.jsonl", manifest)
        print(f"{precision}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
