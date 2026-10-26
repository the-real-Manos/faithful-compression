"""Run manifest construction and append-only logging.

Every suite run appends exactly one line to results/raw/run_manifest.jsonl
recording the full configuration that produced it. A result file with no
matching manifest entry is not evidence.
"""

import json
import platform
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from faithful_compression.backends.types import GenConfig
from faithful_compression.utils.hashing import content_hash

MANIFEST_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "suite",
    "precision",
    "model_file",
    "model_sha256",
    "gen_config",
    "ctx_len",
    "n_gpu_layers",
    "backend",
    "backend_version",
    "gpu",
    "driver_version",
    "os",
    "python_version",
    "content_hash",
)
"""Fields every manifest entry must carry. Validated by build_manifest_entry."""

VALID_SUITES: tuple[str, ...] = ("systems", "capability", "reliability", "behavioural")


def new_run_id() -> str:
    """Return a fresh uuid4 string identifying one suite run."""
    return str(uuid.uuid4())


def build_manifest_entry(
    suite: str,
    precision: str,
    model_file: str,
    model_sha256: str,
    gen_config: GenConfig,
    hardware: dict,
    backend_info: dict,
    prompt_texts: Iterable[str],
    run_id: str | None = None,
) -> dict:
    """Assemble one manifest entry.

    Args:
        suite: One of VALID_SUITES.
        precision: Precision tier label, e.g. "f16".
        model_file: GGUF filename actually loaded.
        model_sha256: Frozen hash from configs/model_primary.yaml.
        gen_config: The generation config used for every item in this run.
        hardware: Parsed configs/hardware.yaml.
        backend_info: Result of Backend.backend_info().
        prompt_texts: Every prompt this run will send, in order.
        run_id: Reuse an existing id, or None to mint one.

    Returns:
        Dict containing exactly MANIFEST_FIELDS.

    Raises:
        ValueError: If suite is not a recognised suite name.
    """
    if suite not in VALID_SUITES:
        raise ValueError(f"unknown suite {suite!r}, expected one of {VALID_SUITES}")

    return {
        "run_id": run_id or new_run_id(),
        "timestamp": datetime.now(UTC).isoformat(),
        "suite": suite,
        "precision": precision,
        "model_file": model_file,
        "model_sha256": model_sha256,
        "gen_config": gen_config.as_manifest_dict(),
        "ctx_len": hardware["ctx_len"],
        "n_gpu_layers": hardware["n_gpu_layers"],
        "backend": backend_info.get("backend", ""),
        "backend_version": backend_info.get("version", ""),
        "gpu": hardware["gpu"],
        "driver_version": hardware["driver_version"],
        "os": hardware["os"],
        "python_version": platform.python_version(),
        "content_hash": content_hash(prompt_texts),
    }


def write_manifest_entry(path: Path, entry: dict) -> None:
    """Append one manifest entry as a JSON line.

    Args:
        path: Target JSONL file, created with parents if absent.
        entry: Entry from build_manifest_entry.

    Raises:
        ValueError: If the entry is missing any required field.
    """
    missing = [field for field in MANIFEST_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"manifest entry missing fields: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    """Read every manifest entry from a JSONL file.

    Args:
        path: Manifest file. A missing file yields an empty list.

    Returns:
        Entries in file order.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
