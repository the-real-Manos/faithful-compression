"""Follow-up F1 (exploratory): JailbreakBench behaviours and PAIR prompts across precisions.

Pre-registered in PROTOCOL.md section 14. Full prompts and generations go only to
gitignored `*_unredacted.jsonl` files. Committed rows carry SHA256 hashes of the
prompt and generation and a truncated snippet of the generation, never the prompt.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import load_jbb_behaviors, load_pair_artifacts
from faithful_compression.eval.behavioural import redact_rows, run_set, sample_calibration
from faithful_compression.utils.hashing import verify_model
from faithful_compression.utils.logging import configure_logging
from faithful_compression.utils.manifest import (
    build_manifest_entry,
    new_run_id,
    write_manifest_entry,
)

CONFIGS_DIR = Path("configs")
RESULTS_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
CORRECTED = "refusal_classification_corrected"


def load_yaml(configs_dir: Path, name: str) -> dict:
    """Load one YAML config file."""
    return yaml.safe_load((configs_dir / name).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON lines."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def load_sets(suite: dict) -> dict[str, list[dict]]:
    """Load the three follow-up item sets named in configs/suite_jbb.yaml."""
    behaviors = suite["behaviors"]
    sets = {
        name: load_jbb_behaviors(
            behaviors["dataset"], behaviors["config"], split, behaviors["revision"], name
        )
        for name, split in behaviors["splits"].items()
    }
    pair = suite["pair"]
    sets[pair["set_name"]] = load_pair_artifacts(
        Path(pair["file"]), pair["sha256"], pair["set_name"]
    )
    return sets


def main(
    configs_dir: Path = CONFIGS_DIR,
    results_dir: Path = RESULTS_DIR,
    summary_dir: Path = SUMMARY_DIR,
    limit: int | None = None,
) -> None:
    """Run follow-up F1 across all sets and precisions.

    Args:
        configs_dir: Directory holding the YAML configs.
        results_dir: Directory receiving the raw JSONL and the run manifest.
        summary_dir: Directory receiving the calibration sample.
        limit: Items per set, for dry runs only; None runs every item.
    """
    configure_logging()
    hardware = load_yaml(configs_dir, "hardware.yaml")
    model_cfg = load_yaml(configs_dir, "model_primary.yaml")
    suite = load_yaml(configs_dir, "suite_jbb.yaml")
    gen_config = GenConfig(**load_yaml(configs_dir, "gen_greedy.yaml"))
    sets = {name: items[:limit] for name, items in load_sets(suite).items()}

    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    model_dir = Path(model_cfg["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows_by_set: dict[str, list[dict]] = {name: [] for name in sets}

    for precision, entry in model_cfg["precisions"].items():
        path = model_dir / entry["filename"]
        verify_model(path, entry["sha256"])
        manifests = {
            name: build_manifest_entry(
                suite="behavioural",
                precision=precision,
                model_file=entry["filename"],
                model_sha256=entry["sha256"],
                gen_config=gen_config,
                hardware=hardware,
                backend_info=backend.backend_info(),
                prompt_texts=[item["prompt"] for item in items],
                run_id=new_run_id(),
            )
            for name, items in sets.items()
        }

        backend.load(str(path), hardware["ctx_len"], hardware["n_gpu_layers"])
        try:
            results = {
                name: run_set(
                    backend, items, name, precision, manifests[name]["run_id"], gen_config
                )
                for name, items in sets.items()
            }
        finally:
            backend.unload()

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        for name, rows in results.items():
            base = results_dir / f"followup_{name}_{precision}_{stamp}"
            write_jsonl(base.with_name(base.name + "_unredacted.jsonl"), rows)
            committed = redact_rows(
                rows, suite["redact_snippet_chars"], redact_prompts=suite["redact_prompts"]
            )
            write_jsonl(base.with_suffix(".jsonl"), committed)
            write_manifest_entry(results_dir / "run_manifest.jsonl", manifests[name])
            refusals = sum(r[CORRECTED] == "REFUSAL" for r in rows)
            print(f"{name} {precision}: {len(rows)} rows, refusals {refusals}/{len(rows)}")
            rows_by_set[name].extend(rows)

    per_class = suite["calibration_per_set_per_class"]
    calibration = [
        row
        for rows in rows_by_set.values()
        for row in sample_calibration(rows, per_class, seed=42, label_field=CORRECTED)
    ]
    write_jsonl(summary_dir / "followup_calibration_unredacted.jsonl", calibration)


if __name__ == "__main__":
    main()
