"""Tests for item-level pairing of results across precision tiers."""

import json
from pathlib import Path

import numpy as np
import pytest

from faithful_compression.stats.paired import (
    dedupe_by_item,
    flipped_items,
    merge_by_item,
    read_jsonl,
)


def _rows(run_id: str, scores: dict) -> list[dict]:
    """Build result rows for one run from an item id to score mapping."""
    return [
        {"run_id": run_id, "item_id": item, "correct": score} for item, score in scores.items()
    ]


class TestReadJsonl:
    def test_reads_records_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        rows = _rows("run-a", {0: 1, 1: 0, 2: 1})
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert read_jsonl(path) == rows

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_jsonl(tmp_path / "absent.jsonl")


class TestDedupeByItem:
    def test_passthrough_when_no_duplicates(self) -> None:
        rows = _rows("run-a", {0: 1, 1: 0})
        assert set(dedupe_by_item(rows)) == {0, 1}

    def test_later_run_supersedes_earlier(self) -> None:
        rows = _rows("run-a", {0: 1}) + _rows("run-b", {0: 0})
        kept = dedupe_by_item(rows)
        assert kept[0]["run_id"] == "run-b"
        assert kept[0]["correct"] == 0

    def test_partial_rerun_only_replaces_overlapping_items(self) -> None:
        rows = _rows("run-a", {0: 1, 1: 1, 2: 1}) + _rows("run-b", {1: 0})
        kept = dedupe_by_item(rows)
        assert [kept[i]["run_id"] for i in (0, 1, 2)] == ["run-a", "run-b", "run-a"]

    def test_logs_when_a_run_is_superseded(self, caplog: pytest.LogCaptureFixture) -> None:
        rows = _rows("run-a", {0: 1}) + _rows("run-b", {0: 0})
        with caplog.at_level("INFO"):
            dedupe_by_item(rows)
        assert "superseded" in caplog.text


class TestMergeByItem:
    def test_aligns_three_precisions(self) -> None:
        by_precision = {
            "f16": _rows("r1", {0: 1, 1: 1, 2: 0}),
            "q8": _rows("r2", {0: 1, 1: 0, 2: 0}),
            "q4": _rows("r3", {0: 0, 1: 1, 2: 0}),
        }
        item_ids, aligned = merge_by_item(by_precision)
        assert item_ids == [0, 1, 2]
        assert np.array_equal(aligned["f16"], np.array([1, 1, 0]))
        assert np.array_equal(aligned["q4"], np.array([0, 1, 0]))

    def test_alignment_is_by_item_id_not_file_order(self) -> None:
        by_precision = {
            "f16": _rows("r1", {0: 1, 1: 0}),
            "q4": _rows("r2", {1: 0, 0: 1}),
        }
        _, aligned = merge_by_item(by_precision)
        assert np.array_equal(aligned["f16"], aligned["q4"])

    def test_string_item_ids_supported(self) -> None:
        by_precision = {
            "f16": _rows("r1", {"harm_001": 1, "harm_000": 0}),
            "q4": _rows("r2", {"harm_000": 0, "harm_001": 1}),
        }
        item_ids, aligned = merge_by_item(by_precision)
        assert item_ids == ["harm_000", "harm_001"]
        assert np.array_equal(aligned["f16"], np.array([0, 1]))

    def test_missing_item_in_one_precision_raises(self) -> None:
        by_precision = {
            "f16": _rows("r1", {0: 1, 1: 1}),
            "q4": _rows("r2", {0: 1}),
        }
        with pytest.raises(ValueError, match="missing"):
            merge_by_item(by_precision)

    def test_error_names_the_offending_precision(self) -> None:
        by_precision = {
            "f16": _rows("r1", {0: 1, 1: 1}),
            "q4": _rows("r2", {0: 1}),
        }
        with pytest.raises(ValueError, match="q4"):
            merge_by_item(by_precision)

    def test_duplicate_run_ids_collapse_before_alignment(self) -> None:
        by_precision = {
            "f16": _rows("r1", {0: 1, 1: 0}) + _rows("r2", {0: 0, 1: 0}),
            "q4": _rows("r3", {0: 0, 1: 0}),
        }
        _, aligned = merge_by_item(by_precision)
        assert np.array_equal(aligned["f16"], np.array([0, 0]))

    def test_alternative_value_field(self) -> None:
        by_precision = {
            "f16": [{"run_id": "r1", "item_id": 0, "n_tokens": 120}],
            "q4": [{"run_id": "r2", "item_id": 0, "n_tokens": 96}],
        }
        _, aligned = merge_by_item(by_precision, value_field="n_tokens")
        assert aligned["f16"][0] == 120

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            merge_by_item({})


class TestFlippedItems:
    def test_returns_ids_that_differ(self) -> None:
        ids = [10, 11, 12, 13]
        a = np.array([1, 1, 0, 1])
        b = np.array([1, 0, 0, 0])
        assert flipped_items(ids, a, b) == [11, 13]

    def test_empty_when_identical(self) -> None:
        ids = [0, 1, 2]
        a = np.array([1, 0, 1])
        assert flipped_items(ids, a, a) == []

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            flipped_items([0, 1], np.array([1, 0, 1]), np.array([1, 0, 1]))
