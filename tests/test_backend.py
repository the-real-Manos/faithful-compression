"""Tests for the llama-server backend's pure functions. No server, no model.

Fixture strings are copied verbatim from a real llama.cpp v0.4.0 (b10809) run on
the study hardware, so a parser change that breaks on the real format fails here.
"""

import pytest

from faithful_compression.backends.llama_cpp_backend import (
    _build_completion_payload,
    _parse_completion_response,
    _parse_vram_from_log,
)
from faithful_compression.backends.types import GenConfig

# Captured from `llama-server -lv 4` loading Llama-3.2-3B-Instruct-f16.gguf.
F16_LOAD_LOG = """\
0.04.227.524 I load_tensors: offloaded 29/29 layers to GPU
0.04.227.530 I load_tensors:   CPU_Mapped model buffer size =   751.50 MiB
0.04.227.531 I load_tensors:      Vulkan0 model buffer size =  6128.17 MiB
0.06.190.199 I llama_context: Vulkan_Host  output buffer size =     0.49 MiB
0.06.274.428 I llama_kv_cache:    Vulkan0 KV buffer size =   448.00 MiB
0.06.309.552 I sched_reserve:    Vulkan0 compute buffer size =    64.01 MiB
0.06.309.555 I sched_reserve: Vulkan_Host compute buffer size =    16.01 MiB
"""

# Captured `timings` block from a /completion response on the same run.
REAL_TIMINGS = {
    "cache_n": 0,
    "prompt_n": 45,
    "prompt_ms": 160.204,
    "prompt_per_token_ms": 3.560088888888889,
    "prompt_per_second": 280.8918628748346,
    "predicted_n": 256,
    "predicted_ms": 3053.203,
    "predicted_per_token_ms": 11.973345098039216,
    "predicted_per_second": 83.51884889409581,
}


class TestParseVramFromLog:
    def test_counts_only_gpu_buffers_on_the_real_log(self) -> None:
        # The 751.50 MiB CPU_Mapped buffer must not be counted as VRAM (R1).
        assert _parse_vram_from_log(F16_LOAD_LOG) == (6128.17, 448.0, 64.01)

    def test_excludes_every_host_buffer_kind(self) -> None:
        log = (
            "CPU model buffer size = 10.00 MiB\n"
            "CPU_REPACK model buffer size = 20.00 MiB\n"
            "Vulkan_Host compute buffer size = 30.00 MiB\n"
        )
        assert _parse_vram_from_log(log) == (None, None, None)

    def test_sums_buffers_split_across_devices(self) -> None:
        log = "Vulkan0 model buffer size = 100.00 MiB\nVulkan1 model buffer size = 50.50 MiB\n"
        assert _parse_vram_from_log(log)[0] == 150.5

    def test_empty_log_yields_none_not_zero(self) -> None:
        # None means "not reported", which is different from "zero VRAM".
        assert _parse_vram_from_log("") == (None, None, None)


class TestParseCompletionResponse:
    def test_maps_real_timings(self) -> None:
        result = _parse_completion_response(
            {"content": "hello", "timings": REAL_TIMINGS}, wall_time_ms=3300.0
        )
        assert result.text == "hello"
        assert result.n_tokens == 256
        assert result.prompt_tokens == 45
        assert result.decode_tok_s == pytest.approx(83.5188, rel=1e-4)
        assert result.ttft_ms == pytest.approx(160.204 + 11.9733, rel=1e-4)

    def test_wall_time_is_the_client_measurement(self) -> None:
        # R11: wall time comes from the caller's clock, not the server's timings.
        result = _parse_completion_response({"timings": REAL_TIMINGS}, wall_time_ms=9999.0)
        assert result.wall_time_ms == 9999.0

    def test_zero_generated_tokens_adds_no_decode_step_to_ttft(self) -> None:
        timings = {**REAL_TIMINGS, "predicted_n": 0}
        result = _parse_completion_response({"timings": timings}, wall_time_ms=1.0)
        assert result.ttft_ms == pytest.approx(160.204)

    def test_missing_timings_raises(self) -> None:
        with pytest.raises(KeyError):
            _parse_completion_response({"content": "x"}, wall_time_ms=1.0)


class TestBuildCompletionPayload:
    def test_pins_every_sampler_to_neutral(self, greedy_config: GenConfig) -> None:
        payload = _build_completion_payload("p", greedy_config)
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["top_k"] == -1
        assert payload["min_p"] == 0.0
        assert payload["repeat_penalty"] == 1.0
        assert payload["presence_penalty"] == 0.0
        assert payload["frequency_penalty"] == 0.0
        assert payload["dry_multiplier"] == 0.0
        assert payload["xtc_probability"] == 0.0
        assert payload["typical_p"] == 1.0
        assert payload["mirostat"] == 0

    def test_disables_prompt_cache(self, greedy_config: GenConfig) -> None:
        assert _build_completion_payload("p", greedy_config)["cache_prompt"] is False

    def test_passes_seed_and_limits_through(self, sampled_config: GenConfig) -> None:
        payload = _build_completion_payload("p", sampled_config)
        assert payload["seed"] == 23
        assert payload["temperature"] == 0.7
        assert payload["n_predict"] == 1024

    def test_greedy_seed_none_maps_to_server_default(self, greedy_config: GenConfig) -> None:
        assert _build_completion_payload("p", greedy_config)["seed"] == -1
