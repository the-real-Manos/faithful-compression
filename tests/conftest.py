"""Shared fixtures. No fixture requires a GPU, a model file or network access."""

import numpy as np
import pytest

from faithful_compression.backends.types import GenConfig, GenResult


@pytest.fixture
def greedy_config() -> GenConfig:
    """Greedy generation config as used by the capability suite."""
    return GenConfig(temperature=0.0, max_tokens=1024, seed=None)


@pytest.fixture
def sampled_config() -> GenConfig:
    """Sampled generation config as used by the reliability suite."""
    return GenConfig(temperature=0.7, max_tokens=1024, seed=23)


@pytest.fixture
def mock_gen_result() -> GenResult:
    """A representative backend result."""
    return GenResult(
        text="The answer is 42.",
        ttft_ms=100.0,
        decode_tok_s=40.0,
        n_tokens=6,
        wall_time_ms=250.0,
        prompt_tokens=50,
    )


@pytest.fixture
def binary_scores_a() -> np.ndarray:
    """Per-item binary scores standing in for a high-precision tier."""
    return np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])


@pytest.fixture
def binary_scores_b() -> np.ndarray:
    """Per-item binary scores standing in for a low-precision tier."""
    return np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1])


@pytest.fixture
def gsm8k_items() -> list[dict]:
    """5 mock GSM8K-style items for testing scoring logic."""
    return [
        {"item_id": 0, "question": "What is 2 + 2?", "answer": "2 + 2 = 4\n#### 4"},
        {"item_id": 1, "question": "What is 3 * 5?", "answer": "3 * 5 = 15\n#### 15"},
        {"item_id": 2, "question": "What is 100 / 4?", "answer": "100 / 4 = 25\n#### 25"},
        {"item_id": 3, "question": "What is 7 + 8?", "answer": "7 + 8 = 15\n#### 15"},
        {"item_id": 4, "question": "What is 12 - 3?", "answer": "12 - 3 = 9\n#### 9"},
    ]


@pytest.fixture
def hardware_config() -> dict:
    """Parsed configs/hardware.yaml equivalent."""
    return {
        "gpu": "AMD Radeon RX 9070 XT",
        "vram_gb": 16,
        "architecture": "RDNA 4 (gfx1201)",
        "os": "Windows 10 Home 10.0.19045",
        "driver_version": "32.0.31021.5001",
        "backend": "llama_cpp_vulkan",
        "vulkan_confirmed": True,
        "ctx_len": 4096,
        "n_gpu_layers": -1,
    }


@pytest.fixture
def backend_info() -> dict:
    """Representative Backend.backend_info() output."""
    return {"backend": "llama_cpp_python", "version": "0.3.34", "build_flags": "GGML_VULKAN=ON"}
