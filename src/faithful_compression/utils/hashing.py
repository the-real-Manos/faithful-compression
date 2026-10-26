"""SHA256 hashing for model files and prompt content."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024


def sha256_text(text: str) -> str:
    """Return the SHA256 hex digest of a string, encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file, read in 1 MiB chunks.

    Args:
        path: File to hash. GGUF weights are several GB, so this streams.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected_sha256: str) -> None:
    """Check a model file against its frozen SHA256 before it is loaded.

    Args:
        path: GGUF file about to be loaded.
        expected_sha256: Frozen hash from configs/model_primary.yaml.

    Raises:
        ValueError: If no hash is frozen, or the file does not match it.
        FileNotFoundError: If the file is absent.
    """
    if not expected_sha256:
        raise ValueError(f"no frozen SHA256 for {path.name} in configs/model_primary.yaml")
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path} (see README, manual data steps)")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{path.name} does not match its frozen SHA256: expected {expected_sha256}, "
            f"got {actual}. The model file changed; do not run the study on it."
        )


def content_hash(texts: Iterable[str]) -> str:
    """Return a SHA256 over a sequence of prompt texts.

    Each text is newline-terminated before hashing so that concatenation is
    unambiguous: ["ab", "c"] and ["a", "bc"] produce different digests.

    Args:
        texts: Prompt strings in the order they will be evaluated.

    Returns:
        Lowercase hex digest identifying the exact prompt set.
    """
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
