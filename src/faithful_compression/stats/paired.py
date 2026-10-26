"""Item-level pairing of results across precision tiers.

Comparisons are always paired by item ID. An unpaired aggregate would let a
different item set stand behind each precision's mean, which is exactly the
error this module exists to make impossible.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL results file into a list of dicts.

    Args:
        path: File to read.

    Returns:
        Records in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"no results file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dedupe_by_item(records: Sequence[dict], item_key: str = "item_id") -> dict:
    """Collapse records to one per item, keeping the last occurrence.

    A suite that was interrupted and re-run appends a second set of rows under a
    new run_id. Last-wins matches append-only file order, so the completed run
    supersedes the partial one. Any collapse is logged with both run_ids.

    Args:
        records: Result rows, in file order.
        item_key: Field holding the item identifier.

    Returns:
        Mapping of item id to the surviving record.
    """
    kept: dict = {}
    for record in records:
        item = record[item_key]
        if item in kept and kept[item].get("run_id") != record.get("run_id"):
            logger.info(
                "item %s superseded: run %s replaced by run %s",
                item,
                kept[item].get("run_id"),
                record.get("run_id"),
            )
        kept[item] = record
    return kept


def merge_by_item(
    records_by_precision: dict[str, Sequence[dict]],
    value_field: str = "correct",
    item_key: str = "item_id",
) -> tuple[list, dict[str, np.ndarray]]:
    """Align per-item values across precision tiers.

    Args:
        records_by_precision: Result rows keyed by precision label.
        value_field: Field to extract as the per-item metric.
        item_key: Field holding the item identifier.

    Returns:
        Tuple of (sorted item ids, mapping of precision to item-aligned array).

    Raises:
        ValueError: If no precisions were supplied, or if any precision is
            missing an item that another precision has.
    """
    if not records_by_precision:
        raise ValueError("no precisions supplied")

    deduped = {
        precision: dedupe_by_item(records, item_key=item_key)
        for precision, records in records_by_precision.items()
    }

    all_items = set().union(*(set(items) for items in deduped.values()))
    for precision, items in deduped.items():
        missing = all_items - set(items)
        if missing:
            sample = sorted(missing, key=str)[:5]
            raise ValueError(
                f"precision {precision!r} is missing {len(missing)} item(s), e.g. {sample}"
            )

    item_ids = sorted(all_items, key=lambda i: (isinstance(i, str), i))
    aligned = {
        precision: np.array([items[item][value_field] for item in item_ids])
        for precision, items in deduped.items()
    }
    return item_ids, aligned


def flipped_items(
    item_ids: Sequence,
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> list:
    """Return the ids of items whose value differs between two conditions.

    Args:
        item_ids: Item identifiers, aligned with both arrays.
        values_a: Per-item values for condition A.
        values_b: Per-item values for condition B.

    Returns:
        Item ids where A != B, in input order.

    Raises:
        ValueError: If lengths do not match.
    """
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    if not (len(item_ids) == values_a.size == values_b.size):
        raise ValueError(
            f"length mismatch: {len(item_ids)} ids, {values_a.size} and {values_b.size} values"
        )
    return [item for item, differs in zip(item_ids, values_a != values_b, strict=True) if differs]
