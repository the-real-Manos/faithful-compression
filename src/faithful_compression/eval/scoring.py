"""Answer extraction and exact-match scoring for GSM8K and ARC-Challenge.

Scoring is deterministic and identical across precision tiers. Extraction
failure is recorded as an incorrect answer with extracted_answer None, never as
a dropped item, so that item pairing stays complete.
"""

import re
from collections.abc import Sequence

_GSM8K_MARKER_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
_GSM8K_FALLBACK_RE = re.compile(r"-?[\d,]*\.?\d+")

_ARC_LETTERS: str = "ABCDE"


def normalise_number(raw: str) -> str | None:
    """Normalise a numeric string to a canonical comparable form.

    Strips commas, currency symbols, trailing full stops and whitespace, then
    collapses integral floats so that "4.0", "4" and " 4, " all compare equal.

    Args:
        raw: Raw numeric substring.

    Returns:
        Canonical numeric string, or None if it does not parse as a number.
    """
    cleaned = raw.strip().replace(",", "").replace("$", "").rstrip(".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def extract_gsm8k_answer(text: str, allow_fallback: bool = False) -> str | None:
    """Extract the final numeric answer from a GSM8K generation.

    Takes the number following the last '####' marker, which is the format the
    prompt requests and the format the dataset's own answer field uses.

    Args:
        text: Generated text, or a dataset answer field.
        allow_fallback: If True and no '####' marker is present, fall back to
            the last bare number in the text. Off by default so that the
            headline score measures instruction following as well as arithmetic;
            the fallback rate is reported separately as a format-failure metric.

    Returns:
        Canonical numeric string, or None if no answer could be extracted.
    """
    marked = _GSM8K_MARKER_RE.findall(text)
    if marked:
        return normalise_number(marked[-1])
    if not allow_fallback:
        return None
    bare = [m for m in _GSM8K_FALLBACK_RE.findall(text) if normalise_number(m) is not None]
    if not bare:
        return None
    return normalise_number(bare[-1])


def score_gsm8k(
    generated_text: str,
    ground_truth_answer: str,
) -> tuple[int, str | None, str | None]:
    """Score one GSM8K generation by exact numeric match.

    Args:
        generated_text: Model output.
        ground_truth_answer: The dataset's `answer` field, containing '#### N'.

    Returns:
        Tuple of (correct as 0 or 1, extracted answer, ground truth answer).
    """
    predicted = extract_gsm8k_answer(generated_text)
    truth = extract_gsm8k_answer(ground_truth_answer)
    correct = int(predicted is not None and truth is not None and predicted == truth)
    return correct, predicted, truth


def normalise_arc_labels(labels: Sequence[str]) -> list[str]:
    """Map an ARC label set onto contiguous capital letters.

    ARC-Challenge is not uniformly four-way lettered: items carry three, four or
    five choices, and some use numeric labels ("1".."4") rather than letters.
    Positional mapping gives every item a single letter alphabet regardless.

    Args:
        labels: Raw `choices.label` values in dataset order.

    Returns:
        Letters A, B, C, ... one per choice, in the same order.

    Raises:
        ValueError: If there are more choices than available letters.
    """
    if len(labels) > len(_ARC_LETTERS):
        raise ValueError(f"{len(labels)} choices exceeds supported labels {_ARC_LETTERS}")
    return list(_ARC_LETTERS[: len(labels)])


def normalise_arc_answer_key(answer_key: str, labels: Sequence[str]) -> str | None:
    """Translate a dataset answerKey into the normalised letter alphabet.

    Args:
        answer_key: The item's `answerKey`, letter or digit.
        labels: Raw `choices.label` values in dataset order.

    Returns:
        Normalised letter, or None if the key does not match any label.
    """
    key = answer_key.strip().upper()
    raw = [str(label).strip().upper() for label in labels]
    if key not in raw:
        return None
    return normalise_arc_labels(labels)[raw.index(key)]


def extract_arc_letter(text: str, n_choices: int = 4) -> str | None:
    """Extract the chosen option letter from an ARC generation.

    Looks for a standalone capital letter within the item's valid range, which
    handles "C", "C) photosynthesis", "(C)" and "The answer is C." alike. If no
    capital letter matches, retries case-insensitively so that a lowercase reply
    still scores; a bare lowercase "a" inside prose is the main false-positive
    risk, which is why the capital pass runs first.

    Args:
        text: Model output.
        n_choices: Number of options this item offers.

    Returns:
        Normalised letter, or None if no option letter was found.
    """
    if n_choices < 1 or n_choices > len(_ARC_LETTERS):
        raise ValueError(f"n_choices must be 1..{len(_ARC_LETTERS)}, got {n_choices}")
    valid = _ARC_LETTERS[:n_choices]
    strict = re.search(rf"\b([{valid}])\b", text)
    if strict:
        return strict.group(1)
    loose = re.search(rf"\b([{valid}{valid.lower()}])\b", text)
    if loose:
        return loose.group(1).upper()
    return None


def score_arc(
    generated_text: str,
    answer_key: str,
    labels: Sequence[str],
) -> tuple[int, str | None, str | None]:
    """Score one ARC-Challenge generation by exact letter match.

    Args:
        generated_text: Model output.
        answer_key: The item's `answerKey` field.
        labels: Raw `choices.label` values in dataset order.

    Returns:
        Tuple of (correct as 0 or 1, extracted letter, normalised answer key).
    """
    truth = normalise_arc_answer_key(answer_key, labels)
    predicted = extract_arc_letter(generated_text, n_choices=len(labels))
    correct = int(predicted is not None and truth is not None and predicted == truth)
    return correct, predicted, truth
