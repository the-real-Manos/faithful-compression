"""Backend implementing the Backend protocol via a llama-server subprocess.

Wraps a pinned, prebuilt llama.cpp `llama-server` binary (path and build in
configs/hardware.yaml) over its local HTTP API, using only the standard library
for transport. `llama-cpp-python` is not used: it is source-only on PyPI and its
Vulkan build needs a Vulkan SDK and CMake; see PROTOCOL.md section 10.

Design choices that matter for correctness of the study:

- Requests go to `/completion`, not `/v1/chat/completions`. The chat template
  is already applied in `faithful_compression.data.loaders`, so the server must
  treat the prompt as raw text rather than re-templating it.
- `cache_prompt` is always False. The prefix cache would make repeated identical
  prompts (Suite S sends each prompt 10 times) artificially fast.
- Every sampler is sent explicitly, including ones GenConfig does not expose, so
  no server default can silently alter the declared sampling distribution.
"""

import json
import logging
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

import psutil

from faithful_compression.backends.types import GenConfig, GenResult, MemoryReport

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"version:\s*(.+)")
_DEVICE_RE = re.compile(r"^\s*(\w+):\s*(.+?)\s*\(", re.MULTILINE)
_BUFFER_RE = re.compile(r"(\S+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB")

# Samplers GenConfig does not expose. These never vary in this study, so they are
# constants rather than config, pinned to the value that disables each one.
_NEUTRAL_SAMPLERS: dict = {
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "xtc_probability": 0.0,
    "typical_p": 1.0,
    "mirostat": 0,
}


def _build_completion_payload(prompt: str, config: GenConfig) -> dict:
    """Build the JSON body for a llama-server /completion request.

    Args:
        prompt: Fully rendered prompt, already chat-templated.
        config: Generation config to translate into server parameters.

    Returns:
        JSON-serialisable request body.
    """
    return {
        "prompt": prompt,
        "n_predict": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "seed": config.seed if config.seed is not None else -1,
        "stop": list(config.stop),
        "cache_prompt": False,
        **_NEUTRAL_SAMPLERS,
    }


def _parse_completion_response(data: dict, wall_time_ms: float) -> GenResult:
    """Translate a llama-server /completion response into a GenResult.

    TTFT is the prompt processing time plus one mean decode step, derived from
    the server's `timings` block rather than a streamed first-token timestamp.

    Args:
        data: Parsed JSON body of a /completion response.
        wall_time_ms: Client-side wall-clock time for the whole request.

    Returns:
        GenResult with all fields populated.

    Raises:
        KeyError: If the response has no `timings` block.
    """
    timings = data["timings"]
    prompt_ms = float(timings.get("prompt_ms", 0.0))
    per_token_ms = float(timings.get("predicted_per_token_ms", 0.0))
    n_tokens = int(timings.get("predicted_n", 0))

    return GenResult(
        text=data.get("content", ""),
        ttft_ms=prompt_ms + (per_token_ms if n_tokens > 0 else 0.0),
        decode_tok_s=float(timings.get("predicted_per_second", 0.0)),
        n_tokens=n_tokens,
        wall_time_ms=wall_time_ms,
        prompt_tokens=int(timings.get("prompt_n", 0)),
        logprobs=None,
    )


def _parse_vram_from_log(log_text: str) -> tuple[float | None, float | None, float | None]:
    """Extract GPU buffer sizes from a llama.cpp startup log.

    llama.cpp prints one line per allocated buffer, naming the device before the
    buffer kind, e.g. "Vulkan0 model buffer size = 6128.17 MiB". Host-memory
    buffers ("CPU", "CPU_Mapped", "Vulkan_Host", ...) are excluded: counting
    them would inflate the VRAM figure by the embedding table llama.cpp keeps in
    host memory.

    Args:
        log_text: Combined stdout/stderr captured during Backend.load().

    Returns:
        Tuple of (weights_mb, kv_cache_mb, compute_mb), each None if no GPU
        buffer of that kind was found.
    """
    totals: dict[str, float] = {}
    for device, kind, size in _BUFFER_RE.findall(log_text):
        if device.startswith("CPU") or device.endswith("_Host"):
            continue
        totals[kind] = totals.get(kind, 0.0) + float(size)
    return totals.get("model"), totals.get("KV"), totals.get("compute")


class LlamaServerBackend:
    """Backend implementation backed by a llama-server subprocess.

    Satisfies the Backend protocol in backends/types.py by duck typing.
    """

    def __init__(
        self,
        server_exe: Path,
        host: str = "127.0.0.1",
        port: int = 8090,
        startup_timeout_s: float = 600.0,
        request_timeout_s: float = 300.0,
    ) -> None:
        """Initialise the backend. Does not start a subprocess yet.

        Args:
            server_exe: Path to the pinned llama-server binary.
            host: Loopback address the server binds to.
            port: Port the server binds to.
            startup_timeout_s: How long to wait for /health after launch. A normal
                load takes about 10 s, but the first run after a driver or binary
                change compiles GPU shaders and can take several minutes.
            request_timeout_s: Per-request HTTP timeout; long prompts with 1024
                max_tokens need headroom beyond typical client defaults.
        """
        if not server_exe.exists():
            raise FileNotFoundError(f"llama-server binary not found at {server_exe}")
        self._server_exe = server_exe.resolve()
        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._log_handle: TextIO | None = None
        self._psutil_process: psutil.Process | None = None
        self._load_time_s: float = 0.0
        self._peak_rss_mb: float = 0.0
        self._weights_vram_mb: float | None = None
        self._kv_cache_vram_mb: float | None = None
        self._compute_vram_mb: float | None = None
        self._info: dict | None = None

    @property
    def load_time_s(self) -> float:
        """Seconds from process launch to a healthy server, for the last load()."""
        return self._load_time_s

    def load(self, model_path: str, ctx_len: int, n_gpu_layers: int) -> None:
        """Launch llama-server with the given model and wait for it to be ready.

        Raises:
            RuntimeError: If a server is already running, or if anything is
                already listening on the port. An orphaned server from a crashed
                run would otherwise answer the health check with the wrong model.
        """
        if self._process is not None:
            raise RuntimeError("load() called while a server is already running")
        try:
            socket.create_connection((self._host, self._port), timeout=1.0).close()
        except OSError:
            pass
        else:
            raise RuntimeError(
                f"port {self._port} is already in use; stop any leftover llama-server "
                "process before starting a run"
            )

        self._log_path = Path(model_path).resolve().parent / f".llama_server_{self._port}.log"
        self._log_handle = self._log_path.open("w", encoding="utf-8")

        argv = [
            str(self._server_exe),
            "-m",
            model_path,
            "-c",
            str(ctx_len),
            "-ngl",
            str(n_gpu_layers),
            "--host",
            self._host,
            "--port",
            str(self._port),
            "-np",
            "1",
            "--no-webui",
            # Buffer-size lines (parsed for VRAM) are only logged at verbosity >= 4.
            "-lv",
            "4",
            # Never let the server silently adjust settings per model to fit memory.
            "--fit",
            "off",
            # Disable the server-wide RAM prompt cache, a second cache separate
            # from the per-request cache_prompt flag.
            "--cache-ram",
            "0",
        ]
        logger.info("launching llama-server: %s", " ".join(argv))
        start = time.monotonic()
        self._process = subprocess.Popen(
            argv,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self._psutil_process = psutil.Process(self._process.pid)
        self._peak_rss_mb = 0.0

        try:
            self._wait_for_health()
        except BaseException:
            self.unload()
            raise
        self._load_time_s = time.monotonic() - start

        log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
        weights, kv, compute = _parse_vram_from_log(log_text)
        self._weights_vram_mb, self._kv_cache_vram_mb, self._compute_vram_mb = weights, kv, compute
        logger.info(
            "llama-server ready in %.2fs (VRAM MiB: weights=%s, kv=%s, compute=%s)",
            self._load_time_s,
            weights,
            kv,
            compute,
        )

    def _wait_for_health(self) -> None:
        """Poll /health until the server responds 200 or the startup timeout elapses."""
        deadline = time.monotonic() + self._startup_timeout_s
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (code {self._process.returncode}); "
                    f"see {self._log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.5)
        raise TimeoutError(
            f"llama-server did not become healthy within {self._startup_timeout_s}s"
        )

    def generate(self, prompt: str, config: GenConfig) -> GenResult:
        """Generate text via a synchronous POST to /completion."""
        if self._process is None:
            raise RuntimeError("generate() called before load()")

        body = json.dumps(_build_completion_payload(prompt, config)).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=self._request_timeout_s) as response:
            data = json.loads(response.read())
        wall_time_ms = (time.monotonic() - start) * 1000.0
        logger.debug("timings: %s", data.get("timings"))
        logger.debug("generation_settings: %s", data.get("generation_settings"))

        if self._psutil_process is not None:
            try:
                rss_mb = self._psutil_process.memory_info().rss / (1024 * 1024)
                self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
            except psutil.NoSuchProcess:
                pass

        return _parse_completion_response(data, wall_time_ms)

    def unload(self) -> None:
        """Terminate the server subprocess and release its resources."""
        if self._process is None:
            return
        logger.info("stopping llama-server (pid %d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not exit in time, killing")
            self._process.kill()
            self._process.wait(timeout=10.0)
        self._process = None
        self._psutil_process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def backend_info(self) -> dict:
        """Return backend metadata read from the binary itself, cached after first call.

        `build_flags` lists the compute devices the binary reports, so the manifest
        records what the engine actually uses rather than what a config claims.
        """
        if self._info is None:
            version = subprocess.run(
                [str(self._server_exe), "--version"], capture_output=True, text=True, timeout=30.0
            )
            devices = subprocess.run(
                [str(self._server_exe), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            match = _VERSION_RE.search(version.stdout + version.stderr)
            found = _DEVICE_RE.findall(devices.stdout + devices.stderr)
            self._info = {
                "backend": "llama_cpp_cli",
                "version": match.group(1).strip() if match else "unknown",
                "build_flags": "; ".join(f"{name}: {desc}" for name, desc in found) or "unknown",
            }
        return self._info

    def memory_report(self) -> MemoryReport:
        """Return the memory footprint recorded since the last load()."""
        return MemoryReport(
            weights_vram_mb=self._weights_vram_mb,
            kv_cache_vram_mb=self._kv_cache_vram_mb,
            compute_vram_mb=self._compute_vram_mb,
            peak_vram_observed_mb=None,
            peak_rss_mb=self._peak_rss_mb,
        )
