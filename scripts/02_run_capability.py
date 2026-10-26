"""Suite C: run GSM8K and ARC-Challenge for every precision and save raw results."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import load_arc_challenge, load_gsm8k
from faithful_compression.eval.capability import run_task
from faithful_compression.utils.hashing import verify_model
from faithful_compression.utils.logging import configure_logging
from faithful_compression.utils.manifest import (
    build_manifest_entry,
    new_run_id,
    write_manifest_entry,
)

CONFIGS_DIR = Path("configs")
RESULTS_DIR = Path("results/raw")
LOADERS = {"gsm8k": load_gsm8k, "arc_challenge": load_arc_challenge}


def load_yaml(configs_dir: Path, name: str) -> dict:
    """Load one YAML config file."""
    return yaml.safe_load((configs_dir / name).read_text(encoding="utf-8"))


def main(configs_dir: Path = CONFIGS_DIR, results_dir: Path = RESULTS_DIR) -> None:
    """Run Suite C across all tasks and precisions.

    Each model is loaded once and runs every task before the next model loads.

    Args:
        configs_dir: Directory holding the YAML configs.
        results_dir: Directory receiving the raw JSONL and the run manifest.
    """
    configure_logging()
    hardware = load_yaml(configs_dir, "hardware.yaml")
    model_cfg = load_yaml(configs_dir, "model_primary.yaml")
    suite = load_yaml(configs_dir, "suite_capability.yaml")
    gen_config = GenConfig(**load_yaml(configs_dir, "gen_greedy.yaml"))

    tasks = {}
    for task, cfg in suite["tasks"].items():
        items = LOADERS[task](
            cfg["dataset"], cfg["config"], cfg["split"], cfg["n_items"], cfg["start_index"]
        )
        skipped = cfg["n_items"] - len(items)
        logging.info("%s: %d items loaded, %d skipped", task, len(items), skipped)
        tasks[task] = items

    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    model_dir = Path(model_cfg["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for precision, entry in model_cfg["precisions"].items():
        path = model_dir / entry["filename"]
        verify_model(path, entry["sha256"])
        manifests = {
            task: build_manifest_entry(
                suite="capability",
                precision=precision,
                model_file=entry["filename"],
                model_sha256=entry["sha256"],
                gen_config=gen_config,
                hardware=hardware,
                backend_info=backend.backend_info(),
                prompt_texts=[item["prompt"] for item in items],
                run_id=new_run_id(),
            )
            for task, items in tasks.items()
        }

        backend.load(str(path), hardware["ctx_len"], hardware["n_gpu_layers"])
        try:
            results = {
                task: run_task(
                    backend, items, task, precision, manifests[task]["run_id"], gen_config
                )
                for task, items in tasks.items()
            }
        finally:
            backend.unload()

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for task, rows in results.items():
            out = results_dir / f"capability_{task}_{precision}_{stamp}.jsonl"
            out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            write_manifest_entry(results_dir / "run_manifest.jsonl", manifests[task])
            extracted = sum(row["extracted_answer"] is not None for row in rows)
            accuracy = sum(row["correct"] for row in rows) / len(rows)
            print(
                f"{task} {precision}: {len(rows)} rows, extracted {extracted}/{len(rows)}, "
                f"accuracy {accuracy:.3f} -> {out}"
            )


if __name__ == "__main__":
    main()
