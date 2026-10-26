"""Suite R: sampled-generation consistency per precision tier."""

from tqdm import tqdm

from faithful_compression.backends.types import Backend, GenConfig
from faithful_compression.eval.scoring import score_gsm8k


def run_reliability(
    backend: Backend,
    items: list[dict],
    precision: str,
    run_id: str,
    gen_config: GenConfig,
) -> list[dict]:
    """Generate and score every GSM8K item once with one sampling seed.

    Args:
        backend: Backend with the model for this precision already loaded.
        items: Items from load_gsm8k.
        precision: Precision tier label, written into each row.
        run_id: Manifest run id, written into each row.
        gen_config: Sampled config carrying this run's seed.

    Returns:
        One row per item, following the spec 10.4 schema.
    """
    rows = []
    for item in tqdm(items, desc=f"reliability {precision} seed {gen_config.seed}", unit="item"):
        result = backend.generate(item["prompt"], gen_config)
        correct, extracted, truth = score_gsm8k(result.text, item["answer"])
        rows.append(
            {
                "run_id": run_id,
                "suite": "reliability",
                "task": "gsm8k",
                "precision": precision,
                "item_id": item["item_id"],
                "seed": gen_config.seed,
                "generated_text": result.text,
                "extracted_answer": extracted,
                "ground_truth": truth,
                "correct": correct,
                "n_tokens": result.n_tokens,
                "wall_time_ms": result.wall_time_ms,
            }
        )
    return rows
