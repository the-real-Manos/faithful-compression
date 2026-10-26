"""Tests for run manifest construction, hashing and append-only logging."""

import hashlib
import json
from pathlib import Path

import pytest

from faithful_compression.backends.types import GenConfig
from faithful_compression.utils.hashing import (
    content_hash,
    sha256_file,
    sha256_text,
    verify_model,
)
from faithful_compression.utils.manifest import (
    MANIFEST_FIELDS,
    build_manifest_entry,
    new_run_id,
    read_manifest,
    write_manifest_entry,
)


@pytest.fixture
def entry(greedy_config: GenConfig, hardware_config: dict, backend_info: dict) -> dict:
    """A representative manifest entry."""
    return build_manifest_entry(
        suite="capability",
        precision="f16",
        model_file="Llama-3.2-3B-Instruct-f16.gguf",
        model_sha256="0" * 64,
        gen_config=greedy_config,
        hardware=hardware_config,
        backend_info=backend_info,
        prompt_texts=["prompt one", "prompt two"],
    )


class TestHashing:
    def test_sha256_of_known_string_matches_expected(self) -> None:
        expected = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert sha256_text("test") == expected

    def test_sha256_text_is_stable(self) -> None:
        assert sha256_text("faithful") == sha256_text("faithful")

    def test_sha256_text_is_sensitive_to_change(self) -> None:
        assert sha256_text("faithful") != sha256_text("faithfui")

    def test_sha256_file_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        payload = b"some bytes" * 10_000
        path.write_bytes(payload)
        assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

    def test_content_hash_is_stable_for_same_input(self) -> None:
        assert content_hash(["a", "b"]) == content_hash(["a", "b"])

    def test_content_hash_changes_with_order(self) -> None:
        assert content_hash(["a", "b"]) != content_hash(["b", "a"])

    def test_content_hash_is_unambiguous_across_boundaries(self) -> None:
        # Without a delimiter these two prompt sets would hash identically.
        assert content_hash(["ab", "c"]) != content_hash(["a", "bc"])

    def test_content_hash_changes_when_a_prompt_changes(self) -> None:
        assert content_hash(["a", "b"]) != content_hash(["a", "b!"])


class TestVerifyModel:
    def test_matching_file_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(b"weights")
        verify_model(path, hashlib.sha256(b"weights").hexdigest())

    def test_changed_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(b"weights, re-downloaded")
        with pytest.raises(ValueError, match="does not match"):
            verify_model(path, hashlib.sha256(b"weights").hexdigest())

    def test_empty_frozen_hash_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "m.gguf"
        path.write_bytes(b"weights")
        with pytest.raises(ValueError, match="no frozen SHA256"):
            verify_model(path, "")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_model(tmp_path / "absent.gguf", "a" * 64)


class TestBuildManifestEntry:
    def test_contains_all_required_fields(self, entry: dict) -> None:
        assert set(entry) == set(MANIFEST_FIELDS)

    def test_records_the_generation_config(self, entry: dict) -> None:
        assert entry["gen_config"]["temperature"] == 0.0
        assert entry["gen_config"]["max_tokens"] == 1024
        assert entry["gen_config"]["seed"] is None

    def test_records_pinned_samplers_explicitly(self, entry: dict) -> None:
        # A manifest that omits these cannot show the samplers were neutral.
        assert entry["gen_config"]["top_p"] == 1.0
        assert entry["gen_config"]["top_k"] == -1
        assert entry["gen_config"]["min_p"] == 0.0

    def test_records_hardware_and_backend(self, entry: dict) -> None:
        assert entry["gpu"] == "AMD Radeon RX 9070 XT"
        assert entry["driver_version"] == "32.0.31021.5001"
        assert entry["ctx_len"] == 4096
        assert entry["n_gpu_layers"] == -1
        assert entry["backend"] == "llama_cpp_python"
        assert entry["backend_version"] == "0.3.34"

    def test_content_hash_matches_the_prompt_set(self, entry: dict) -> None:
        assert entry["content_hash"] == content_hash(["prompt one", "prompt two"])

    def test_run_id_can_be_supplied(
        self, greedy_config: GenConfig, hardware_config: dict, backend_info: dict
    ) -> None:
        built = build_manifest_entry(
            suite="systems",
            precision="q4",
            model_file="m.gguf",
            model_sha256="a" * 64,
            gen_config=greedy_config,
            hardware=hardware_config,
            backend_info=backend_info,
            prompt_texts=["p"],
            run_id="fixed-id",
        )
        assert built["run_id"] == "fixed-id"

    def test_unknown_suite_raises(
        self, greedy_config: GenConfig, hardware_config: dict, backend_info: dict
    ) -> None:
        with pytest.raises(ValueError, match="unknown suite"):
            build_manifest_entry(
                suite="not_a_suite",
                precision="f16",
                model_file="m.gguf",
                model_sha256="a" * 64,
                gen_config=greedy_config,
                hardware=hardware_config,
                backend_info=backend_info,
                prompt_texts=["p"],
            )

    def test_run_ids_are_unique(self) -> None:
        assert new_run_id() != new_run_id()


class TestWriteAndReadManifest:
    def test_round_trips_one_entry(self, tmp_path: Path, entry: dict) -> None:
        path = tmp_path / "run_manifest.jsonl"
        write_manifest_entry(path, entry)
        assert read_manifest(path) == [entry]

    def test_appends_rather_than_overwrites(self, tmp_path: Path, entry: dict) -> None:
        path = tmp_path / "run_manifest.jsonl"
        write_manifest_entry(path, entry)
        write_manifest_entry(path, {**entry, "precision": "q4"})
        read = read_manifest(path)
        assert [e["precision"] for e in read] == ["f16", "q4"]

    def test_creates_parent_directories(self, tmp_path: Path, entry: dict) -> None:
        path = tmp_path / "results" / "raw" / "run_manifest.jsonl"
        write_manifest_entry(path, entry)
        assert path.exists()

    def test_each_entry_is_one_json_line(self, tmp_path: Path, entry: dict) -> None:
        path = tmp_path / "run_manifest.jsonl"
        write_manifest_entry(path, entry)
        write_manifest_entry(path, entry)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def test_missing_field_raises(self, tmp_path: Path, entry: dict) -> None:
        del entry["content_hash"]
        with pytest.raises(ValueError, match="missing fields"):
            write_manifest_entry(tmp_path / "run_manifest.jsonl", entry)

    def test_reading_absent_manifest_returns_empty(self, tmp_path: Path) -> None:
        assert read_manifest(tmp_path / "absent.jsonl") == []
