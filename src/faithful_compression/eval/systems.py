"""Suite S: systems cost (latency, throughput, memory) per precision tier."""

import logging

from tqdm import tqdm

from faithful_compression.backends.types import Backend, GenConfig

logger = logging.getLogger(__name__)


def run_systems(
    backend: Backend,
    prompts: list[dict],
    gen_config: GenConfig,
    warmup_runs: int,
    measured_runs: int,
    precision: str,
    run_id: str,
    load_time_s: float,
) -> list[dict]:
    """Run the systems suite against an already-loaded model.

    Warmup generations run on the first prompt and are never recorded. Each
    measured row follows the spec 10.2 schema.

    Args:
        backend: Backend with the model for this precision already loaded.
        prompts: Rows from load_systems_prompts, each carrying a rendered `prompt`.
        gen_config: Greedy generation config, identical across precisions.
        warmup_runs: Discarded generations before measurement starts.
        measured_runs: Recorded generations per prompt.
        precision: Precision tier label, written into each row.
        run_id: Manifest run id, written into each row.
        load_time_s: Seconds the backend took to load this model.

    Returns:
        One row per measured generation.
    """
    for _ in range(warmup_runs):
        backend.generate(prompts[0]["prompt"], gen_config)

    rows = []
    total = len(prompts) * measured_runs
    with tqdm(total=total, desc=f"systems {precision}", unit="gen") as progress:
        for prompt in prompts:
            for repetition in range(1, measured_runs + 1):
                result = backend.generate(prompt["prompt"], gen_config)
                memory = backend.memory_report()
                gpu_parts = (
                    memory.weights_vram_mb,
                    memory.kv_cache_vram_mb,
                    memory.compute_vram_mb,
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "precision": precision,
                        "prompt_id": prompt["prompt_id"],
                        "length_bucket": prompt["length_bucket"],
                        "repetition": repetition,
                        "load_time_s": load_time_s,
                        "ttft_ms": result.ttft_ms,
                        "decode_tok_s": result.decode_tok_s,
                        "n_tokens": result.n_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "wall_time_ms": result.wall_time_ms,
                        "peak_rss_mb": memory.peak_rss_mb,
                        "peak_vram_mb": None
                        if memory.weights_vram_mb is None
                        else sum(part or 0.0 for part in gpu_parts),
                        "is_warmup": False,
                    }
                )
                progress.update(1)
    return rows
