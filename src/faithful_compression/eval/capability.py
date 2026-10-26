"""Suite C: task capability (GSM8K, ARC-Challenge) per precision tier."""

from tqdm import tqdm

from faithful_compression.backends.types import Backend, GenConfig
from faithful_compression.eval.scoring import score_arc, score_gsm8k


def run_task(
    backend: Backend,
    items: list[dict],
    task: str,
    precision: str,
    run_id: str,
    gen_config: GenConfig,
) -> list[dict]:
    """Generate and score every item of one task against an already-loaded model.

    An extraction failure still produces a row, with `correct` 0 and
    `extracted_answer` None, so that item pairing across precisions stays complete.

    Args:
        backend: Backend with the model for this precision already loaded.
        items: Items from load_gsm8k or load_arc_challenge.
        task: "gsm8k" or "arc_challenge".
        precision: Precision tier label, written into each row.
        run_id: Manifest run id, written into each row.
        gen_config: Greedy generation config, identical across precisions.

    Returns:
        One row per item, following the spec 10.3 schema.

    Raises:
        ValueError: If task is not a known task.
    """
    if task not in ("gsm8k", "arc_challenge"):
        raise ValueError(f"unknown task {task!r}")

    rows = []
    for item in tqdm(items, desc=f"{task} {precision}", unit="item"):
        result = backend.generate(item["prompt"], gen_config)
        if task == "gsm8k":
            correct, extracted, truth = score_gsm8k(result.text, item["answer"])
        else:
            correct, extracted, truth = score_arc(result.text, item["answer_key"], item["labels"])
        rows.append(
            {
                "run_id": run_id,
                "suite": "capability",
                "task": task,
                "precision": precision,
                "item_id": item["item_id"],
                "prompt_text": item["prompt"],
                "generated_text": result.text,
                "extracted_answer": extracted,
                "ground_truth": truth,
                "correct": correct,
                "ttft_ms": result.ttft_ms,
                "decode_tok_s": result.decode_tok_s,
                "n_tokens": result.n_tokens,
                "wall_time_ms": result.wall_time_ms,
            }
        )
    return rows
