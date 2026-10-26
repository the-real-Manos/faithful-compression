"""Suite B: run the harmful and jailbreak sets for every precision and save raw results.

Full generations go to gitignored `*_unredacted.jsonl` files; the committed files
carry a truncated snippet and a SHA256 of each generation instead.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from faithful_compression.backends.llama_cpp_backend import LlamaServerBackend
from faithful_compression.backends.types import GenConfig
from faithful_compression.data.loaders import (
    build_jailbreak_items,
    load_advbench_harmful,
    load_jailbreak_templates,
)
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


def load_yaml(configs_dir: Path, name: str) -> dict:
    """Load one YAML config file."""
    return yaml.safe_load((configs_dir / name).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON lines."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main(
    configs_dir: Path = CONFIGS_DIR,
    results_dir: Path = RESULTS_DIR,
    summary_dir: Path = SUMMARY_DIR,
) -> None:
    """Run Suite B across both sets and all precisions.

    Args:
        configs_dir: Directory holding the YAML configs.
        results_dir: Directory receiving the raw JSONL and the run manifest.
        summary_dir: Directory receiving the calibration sample.
    """
    configure_logging()
    hardware = load_yaml(configs_dir, "hardware.yaml")
    model_cfg = load_yaml(configs_dir, "model_primary.yaml")
    suite = load_yaml(configs_dir, "suite_behavioural.yaml")
    gen_config = GenConfig(**load_yaml(configs_dir, "gen_greedy.yaml"))

    harmful_cfg, jail_cfg = suite["harmful"], suite["jailbreak"]
    harmful = load_advbench_harmful(
        Path(harmful_cfg["file"]), harmful_cfg["n_items"], harmful_cfg["start_index"]
    )
    templates = load_jailbreak_templates(Path(jail_cfg["templates_file"]))[
        : jail_cfg["n_templates"]
    ]
    sets = {
        "harmful": harmful,
        "jailbreak": build_jailbreak_items(templates, harmful, jail_cfg["harmful_indices"]),
    }

    backend = LlamaServerBackend(Path(hardware["llama_server_path"]))
    model_dir = Path(model_cfg["model_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

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
            base = results_dir / f"behavioural_{name}_{precision}_{stamp}"
            write_jsonl(base.with_name(base.name + "_unredacted.jsonl"), rows)
            committed = (
                redact_rows(rows, suite["redact_snippet_chars"])
                if suite["redact_generations"]
                else rows
            )
            write_jsonl(base.with_suffix(".jsonl"), committed)
            write_manifest_entry(results_dir / "run_manifest.jsonl", manifests[name])
            refusals = sum(r["refusal_classification"] == "REFUSAL" for r in rows)
            print(f"{name} {precision}: {len(rows)} rows, refusals {refusals}/{len(rows)}")
            all_rows.extend(rows)

    write_jsonl(summary_dir / "refusal_calibration_unredacted.jsonl", sample_calibration(all_rows))


if __name__ == "__main__":
    main()
