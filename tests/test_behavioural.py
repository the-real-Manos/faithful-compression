"""Tests for behavioural-suite redaction and calibration sampling.

Redaction is safety-critical: committed results must never carry a full model
completion to a harmful prompt.
"""

import hashlib
import json
from pathlib import Path

import pytest

from faithful_compression.data.loaders import load_pair_artifacts
from faithful_compression.eval.behavioural import (
    classification_fields,
    redact_rows,
    sample_calibration,
)
from faithful_compression.eval.refusal_classifier import COMPLIANCE, REFUSAL
from faithful_compression.utils.hashing import sha256_text


def _row(item_id: str, label: str, text: str = "x" * 500) -> dict:
    """Build a minimal behavioural row."""
    return {
        "run_id": "r",
        "suite": "behavioural",
        "set": "harmful",
        "precision": "f16",
        "item_id": item_id,
        "prompt_text": "p",
        "generated_text": text,
        "refusal_classification": label,
        "matched_pattern": None,
        "n_tokens": 1,
        "wall_time_ms": 1.0,
    }


class TestClassificationFields:
    def test_records_frozen_and_corrected_labels(self) -> None:
        fields = classification_fields("I can’t help with that.")
        assert fields["refusal_classification"] == COMPLIANCE
        assert fields["matched_pattern"] is None
        assert fields["refusal_classification_corrected"] == REFUSAL
        assert fields["matched_pattern_corrected"] == "i can't"


class TestRedactRows:
    def test_truncates_every_generation(self) -> None:
        rows = [_row("a", REFUSAL, "y" * 1000), _row("b", COMPLIANCE, "short")]
        redacted = redact_rows(rows, snippet_chars=120)
        assert all(len(r["generated_text"]) <= 120 for r in redacted)

    def test_hash_identifies_the_full_text(self) -> None:
        full = "the complete completion " * 40
        redacted = redact_rows([_row("a", COMPLIANCE, full)], snippet_chars=10)
        assert redacted[0]["generated_text_sha256"] == sha256_text(full)

    def test_does_not_modify_the_input_rows(self) -> None:
        rows = [_row("a", COMPLIANCE, "z" * 300)]
        redact_rows(rows, snippet_chars=5)
        assert len(rows[0]["generated_text"]) == 300

    def test_keeps_classification_fields(self) -> None:
        redacted = redact_rows([_row("a", REFUSAL)], snippet_chars=5)
        assert redacted[0]["refusal_classification"] == REFUSAL
        assert redacted[0]["item_id"] == "a"


class TestSampleCalibration:
    def test_draws_ten_of_each_class(self) -> None:
        rows = [_row(f"r{i}", REFUSAL) for i in range(30)]
        rows += [_row(f"c{i}", COMPLIANCE) for i in range(30)]
        sample = sample_calibration(rows)
        labels = [r["refusal_classification"] for r in sample]
        assert labels.count(REFUSAL) == 10
        assert labels.count(COMPLIANCE) == 10

    def test_takes_all_of_a_short_class(self) -> None:
        rows = [_row(f"r{i}", REFUSAL) for i in range(30)] + [_row("c0", COMPLIANCE)]
        sample = sample_calibration(rows)
        assert [r["refusal_classification"] for r in sample].count(COMPLIANCE) == 1

    def test_is_reproducible(self) -> None:
        rows = [_row(f"r{i}", REFUSAL) for i in range(30)]
        rows += [_row(f"c{i}", COMPLIANCE) for i in range(30)]
        assert sample_calibration(rows) == sample_calibration(rows)

    def test_leaves_human_label_blank(self) -> None:
        sample = sample_calibration([_row("r0", REFUSAL), _row("c0", COMPLIANCE)])
        assert all(r["human_label"] == "" for r in sample)


class TestFollowupRedactionAndSampling:
    def test_prompt_redaction_removes_prompt_but_keeps_hash(self) -> None:
        row = {**_row("a", REFUSAL), "prompt_text": "a private prompt"}
        redacted = redact_rows([row], snippet_chars=10, redact_prompts=True)[0]
        assert redacted["prompt_text"] is None
        assert redacted["prompt_text_sha256"] == sha256_text("a private prompt")

    def test_prompts_kept_by_default(self) -> None:
        redacted = redact_rows([_row("a", REFUSAL)], snippet_chars=10)[0]
        assert redacted["prompt_text"] == "p"
        assert "prompt_text_sha256" not in redacted

    def test_sampling_can_use_the_corrected_label(self) -> None:
        rows = [
            {
                **_row(f"r{i}", COMPLIANCE),
                "refusal_classification_corrected": REFUSAL,
                "matched_pattern_corrected": "i can't",
            }
            for i in range(3)
        ]
        sample = sample_calibration(
            rows, n_per_class=2, label_field="refusal_classification_corrected"
        )
        assert [r["refusal_classification"] for r in sample] == [REFUSAL, REFUSAL]
        assert all(r["matched_pattern"] == "i can't" for r in sample)


class TestLoadPairArtifacts:
    def _write(self, tmp_path: Path) -> tuple[Path, str]:
        data = {
            "jailbreaks": [
                {"index": 1, "category": "c", "prompt": "first"},
                {"index": 2, "category": "c", "prompt": None},
                {"index": 3, "category": "c", "prompt": "third"},
            ]
        }
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_skips_entries_without_a_prompt(self, tmp_path: Path) -> None:
        path, sha = self._write(tmp_path)
        items = load_pair_artifacts(path, sha, "jbb_pair")
        assert [i["item_id"] for i in items] == ["jbb_pair_001", "jbb_pair_003"]
        assert items[0]["prompt"].startswith("<|begin_of_text|>")

    def test_rejects_a_changed_file(self, tmp_path: Path) -> None:
        path, _ = self._write(tmp_path)
        with pytest.raises(ValueError, match="does not match"):
            load_pair_artifacts(path, "0" * 64, "jbb_pair")
