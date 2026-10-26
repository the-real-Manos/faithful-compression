"""Suite B: refusal behaviour on harmful and jailbreak prompts per precision tier."""

import random

from tqdm import tqdm

from faithful_compression.backends.types import Backend, GenConfig
from faithful_compression.eval.refusal_classifier import COMPLIANCE, REFUSAL, classify
from faithful_compression.utils.hashing import sha256_text


def run_set(
    backend: Backend,
    items: list[dict],
    set_name: str,
    precision: str,
    run_id: str,
    gen_config: GenConfig,
) -> list[dict]:
    """Generate and classify every item of one behavioural set.

    Args:
        backend: Backend with the model for this precision already loaded.
        items: Harmful or jailbreak items, each carrying `item_id` and `prompt`.
        set_name: "harmful" or "jailbreak".
        precision: Precision tier label, written into each row.
        run_id: Manifest run id, written into each row.
        gen_config: Greedy generation config, identical across precisions.

    Returns:
        One row per item with the full generation, following the spec 10.5 schema.
    """
    rows = []
    for item in tqdm(items, desc=f"{set_name} {precision}", unit="item"):
        result = backend.generate(item["prompt"], gen_config)
        rows.append(
            {
                "run_id": run_id,
                "suite": "behavioural",
                "set": set_name,
                "precision": precision,
                "item_id": item["item_id"],
                "prompt_text": item["prompt"],
                "generated_text": result.text,
                **classification_fields(result.text),
                "n_tokens": result.n_tokens,
                "wall_time_ms": result.wall_time_ms,
            }
        )
    return rows


def classification_fields(text: str) -> dict:
    """Return both the frozen and the corrected classification (PROTOCOL.md deviation 12).

    Args:
        text: Full generated text.

    Returns:
        The frozen label and pattern under the spec 10.5 names, and the
        apostrophe-normalised label and pattern under `*_corrected` names.
    """
    frozen, frozen_pattern = classify(text, normalise_apostrophes=False)
    corrected, corrected_pattern = classify(text)
    return {
        "refusal_classification": frozen,
        "matched_pattern": frozen_pattern,
        "refusal_classification_corrected": corrected,
        "matched_pattern_corrected": corrected_pattern,
    }


def redact_rows(rows: list[dict], snippet_chars: int, redact_prompts: bool = False) -> list[dict]:
    """Return copies of rows safe to commit: generation truncated, full text hashed.

    Args:
        rows: Rows from run_set, holding full generations.
        snippet_chars: Characters of each generation to keep.
        redact_prompts: Also replace `prompt_text` with None and a SHA256.

    Returns:
        New rows where `generated_text` is at most snippet_chars long and
        `generated_text_sha256` identifies the full text.
    """
    redacted = []
    for row in rows:
        new = {
            **row,
            "generated_text": row["generated_text"][:snippet_chars],
            "generated_text_sha256": sha256_text(row["generated_text"]),
        }
        if redact_prompts:
            new["prompt_text"] = None
            new["prompt_text_sha256"] = sha256_text(row["prompt_text"])
        redacted.append(new)
    return redacted


def sample_calibration(
    rows: list[dict],
    n_per_class: int = 10,
    seed: int = 42,
    label_field: str = "refusal_classification",
) -> list[dict]:
    """Sample rows for the manual refusal-classifier calibration.

    Draws n_per_class REFUSAL and n_per_class COMPLIANCE rows at random, pooled
    across sets and precisions. If a class has fewer rows than requested, all of
    them are taken and the shortfall is visible in the returned counts.

    Args:
        rows: Unredacted rows from run_set.
        n_per_class: Rows to draw per classification.
        seed: RNG seed, so the sample is reproducible.
        label_field: Classification column to sample on, frozen or corrected.

    Returns:
        Rows with the full generation and an empty `human_label` to fill in.
    """
    rng = random.Random(seed)
    sample = []
    for label in (REFUSAL, COMPLIANCE):
        pool = [row for row in rows if row[label_field] == label]
        sample.extend(rng.sample(pool, min(n_per_class, len(pool))))
    return [
        {
            "item_id": row["item_id"],
            "set": row["set"],
            "precision": row["precision"],
            "prompt_text": row["prompt_text"],
            "generated_text": row["generated_text"],
            "refusal_classification": row[label_field],
            "matched_pattern": row[
                "matched_pattern" + label_field.removeprefix("refusal_classification")
            ],
            "human_label": "",
        }
        for row in sample
    ]
