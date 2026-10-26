"""Backend interface types shared by every inference backend.

Eval loops never branch on precision tier. They call `Backend.generate()`
identically; precision differs only in which GGUF file `Backend.load()` receives.
"""

from dataclasses import dataclass, field
from typing import Protocol

PRECISIONS: tuple[str, ...] = ("f16", "q8", "q4")
"""Precision tier labels, ordered from highest to lowest bit width."""


@dataclass(frozen=True)
class GenConfig:
    """Generation configuration. Identical across precisions for a given suite.

    Every sampler field is explicit because llama.cpp applies non-neutral
    defaults (top_k=40, top_p=0.95, min_p=0.05) to any parameter left unset. A
    backend must pass all of these through rather than relying on defaults.

    Attributes:
        temperature: Sampling temperature. 0.0 means greedy decode.
        max_tokens: Hard cap on generated tokens.
        seed: RNG seed, or None for greedy decode where sampling is unused.
        stop: Extra stop strings. Empty means the model's EOS token only.
        top_p: Nucleus sampling threshold. 1.0 disables it.
        top_k: Top-k cutoff. -1 disables it.
        min_p: Minimum probability threshold. 0.0 disables it.
    """

    temperature: float
    max_tokens: int
    seed: int | None
    stop: list[str] = field(default_factory=list)
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0

    def as_manifest_dict(self) -> dict:
        """Return the config as a plain dict for the run manifest."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }


@dataclass(frozen=True)
class GenResult:
    """Single generation result from any backend.

    Attributes:
        text: Generated text, excluding the prompt.
        ttft_ms: Time to first token in milliseconds.
        decode_tok_s: Tokens per second during decode, excluding TTFT.
        n_tokens: Number of tokens generated.
        wall_time_ms: Total wall-clock time in milliseconds.
        prompt_tokens: Number of prompt tokens.
        logprobs: Per-token logprobs if the backend supplied them.
    """

    text: str
    ttft_ms: float
    decode_tok_s: float
    n_tokens: int
    wall_time_ms: float
    prompt_tokens: int
    logprobs: list[float] | None = None


@dataclass(frozen=True)
class MemoryReport:
    """Memory footprint of a loaded model.

    RSS alone is misleading when n_gpu_layers is -1, because the weights live in
    VRAM rather than host RAM. `*_vram_mb` fields come from llama.cpp's own
    allocation report and count GPU-device buffers only, never host buffers.

    Attributes:
        weights_vram_mb: Model weight buffers allocated on the GPU device.
        kv_cache_vram_mb: KV cache buffers allocated on the GPU device.
        compute_vram_mb: Compute scratch buffers allocated on the GPU device.
        peak_vram_observed_mb: Observed peak dedicated GPU memory. Always None:
            sampling was cut in favour of llama.cpp's exact allocation figures.
        peak_rss_mb: Peak host resident set size of the inference process.
    """

    weights_vram_mb: float | None
    kv_cache_vram_mb: float | None
    compute_vram_mb: float | None
    peak_vram_observed_mb: float | None
    peak_rss_mb: float


class Backend(Protocol):
    """Interface that every backend must implement."""

    def load(self, model_path: str, ctx_len: int, n_gpu_layers: int) -> None:
        """Load model into memory. Call once per precision tier."""
        ...

    def generate(self, prompt: str, config: GenConfig) -> GenResult:
        """Generate text. Must populate all GenResult fields."""
        ...

    def unload(self) -> None:
        """Free model from memory."""
        ...

    def backend_info(self) -> dict:
        """Return metadata dict: backend name, version, build flags."""
        ...

    def memory_report(self) -> MemoryReport:
        """Return the memory footprint observed since the last load()."""
        ...
