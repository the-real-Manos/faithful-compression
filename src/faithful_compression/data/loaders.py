"""Dataset loading and prompt formatting.

Prompt text is byte-for-byte identical across precision tiers. Formatting lives
here, separate from dataset download, so that templates are unit-testable
without network access.
"""

import csv
import json
import logging
from pathlib import Path

from faithful_compression.utils.hashing import verify_model

logger = logging.getLogger(__name__)

_LLAMA32_USER_TURN = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. "
    'Give your final answer as a single number after "#### ".'
)

# Repeated after the question: with the spec's instruction alone, Llama 3.2 often
# ended with its trained "The final answer is: $\boxed{N}$" habit, and fewer than
# 90% of answers were extractable (PROTOCOL.md section 12.6).
GSM8K_FORMAT_REMINDER = (
    "After working it out, your response must end with one final line in exactly this "
    "format, where N is the answer as a plain number:\n"
    "#### N"
)

ARC_INSTRUCTION = (
    "Answer the following multiple-choice question. Reply with only the letter ({letters})."
)

_CHOICE_LETTERS = "ABCDE"


def wrap_llama32_user(content: str) -> str:
    """Wrap user content in the Llama 3.2 instruct template, no system prompt.

    Args:
        content: The user turn's text.

    Returns:
        Fully formatted prompt string ready for the backend.
    """
    return _LLAMA32_USER_TURN.format(content=content)


def format_gsm8k_prompt(question: str) -> str:
    """Format one GSM8K question into a Llama 3.2 prompt.

    Args:
        question: The item's `question` field.

    Returns:
        Formatted prompt string.
    """
    return wrap_llama32_user(f"{GSM8K_INSTRUCTION}\n\n{question}\n\n{GSM8K_FORMAT_REMINDER}")


def format_arc_prompt(question: str, choices: list[str]) -> str:
    """Format one ARC-Challenge question into a Llama 3.2 prompt.

    The letter range is derived from the item's own choice count. ARC-Challenge
    items carry three, four or five options, so a fixed A-D template would
    truncate or mislabel a minority of items.

    Args:
        question: The item's question stem.
        choices: Choice texts in dataset order.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If the item has no choices or more than five.
    """
    if not 1 <= len(choices) <= len(_CHOICE_LETTERS):
        raise ValueError(f"unsupported choice count: {len(choices)}")
    letters = _CHOICE_LETTERS[: len(choices)]
    lines = [f"{letter}) {text}" for letter, text in zip(letters, choices, strict=True)]
    rendered = "\n".join(lines)
    if len(letters) > 1:
        letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    else:
        letter_list = letters
    instruction = ARC_INSTRUCTION.format(letters=letter_list)
    return wrap_llama32_user(f"{instruction}\n\n{question}\n\n{rendered}")


def load_gsm8k(
    dataset: str,
    config: str,
    split: str,
    n_items: int,
    start_index: int = 0,
) -> list[dict]:
    """Load GSM8K items with stable dataset-index item ids.

    Args:
        dataset: HuggingFace dataset id.
        config: Dataset config name.
        split: Split name.
        n_items: Number of items to take.
        start_index: First dataset index to take.

    Returns:
        List of dicts with item_id, question, answer and prompt.
    """
    from datasets import load_dataset

    data = load_dataset(dataset, config, split=split)
    items = []
    for offset in range(n_items):
        index = start_index + offset
        row = data[index]
        items.append(
            {
                "item_id": index,
                "question": row["question"],
                "answer": row["answer"],
                "prompt": format_gsm8k_prompt(row["question"]),
            }
        )
    return items


def load_arc_challenge(
    dataset: str,
    config: str,
    split: str,
    n_items: int,
    start_index: int = 0,
) -> list[dict]:
    """Load ARC-Challenge items with stable dataset-index item ids.

    Items whose choice count exceeds the supported letter range are skipped and
    logged rather than silently mislabelled; the skipped count is reported in the
    technical note.

    Args:
        dataset: HuggingFace dataset id.
        config: Dataset config name.
        split: Split name.
        n_items: Number of items to take.
        start_index: First dataset index to take.

    Returns:
        List of dicts with item_id, question, choices, labels, answer_key
        and prompt.
    """
    from datasets import load_dataset

    data = load_dataset(dataset, config, split=split)
    items = []
    for offset in range(n_items):
        index = start_index + offset
        row = data[index]
        choices = list(row["choices"]["text"])
        labels = [str(label) for label in row["choices"]["label"]]
        if not 1 <= len(choices) <= len(_CHOICE_LETTERS):
            logger.warning("skipping ARC item %d with %d choices", index, len(choices))
            continue
        items.append(
            {
                "item_id": index,
                "question": row["question"],
                "choices": choices,
                "labels": labels,
                "answer_key": row["answerKey"],
                "prompt": format_arc_prompt(row["question"], choices),
            }
        )
    return items


def load_systems_prompts(path: Path) -> list[dict]:
    """Load the fixed systems-suite prompt set from JSONL.

    Args:
        path: Path to data/systems/prompts.jsonl.

    Returns:
        List of dicts with prompt_id, length_bucket, text and prompt.
    """
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        row["prompt"] = wrap_llama32_user(row["text"])
    return rows


def load_advbench_harmful(path: Path, n_items: int, start_index: int = 0) -> list[dict]:
    """Load AdvBench harmful behaviours from the upstream CSV.

    Args:
        path: Path to harmful_behaviors.csv.
        n_items: Number of rows to take.
        start_index: First row to take.

    Returns:
        List of dicts with item_id, text and prompt.

    Raises:
        FileNotFoundError: If the CSV has not been downloaded.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"AdvBench CSV not found at {path}. See README for the download step."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[start_index : start_index + n_items]
    return [
        {
            "item_id": f"harm_{start_index + offset:03d}",
            "text": row["goal"],
            "prompt": wrap_llama32_user(row["goal"]),
        }
        for offset, row in enumerate(selected)
    ]


def load_jailbreak_templates(path: Path) -> list[dict]:
    """Load the curated jailbreak template set from JSONL.

    Args:
        path: Path to data/behavioural/jailbreak_templates.jsonl.

    Returns:
        List of dicts with template_id, category and template.

    Raises:
        ValueError: If any template lacks the {REQUEST} placeholder.
    """
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    bad = [row["template_id"] for row in rows if "{REQUEST}" not in row["template"]]
    if bad:
        raise ValueError(f"templates missing the {{REQUEST}} placeholder: {bad}")
    return rows


def build_jailbreak_items(
    templates: list[dict],
    harmful_items: list[dict],
    harmful_indices: list[int],
) -> list[dict]:
    """Cross templates with harmful requests to build the jailbreak item set.

    Args:
        templates: Loaded jailbreak templates.
        harmful_items: Loaded AdvBench items.
        harmful_indices: Which harmful item indices to cross with each template.

    Returns:
        List of dicts with item_id, template_id, category, harm_index and prompt.
    """
    items = []
    for template_index, template in enumerate(templates):
        for harm_index in harmful_indices:
            request = harmful_items[harm_index]["text"]
            filled = template["template"].replace("{REQUEST}", request)
            items.append(
                {
                    "item_id": f"jail_{template_index:03d}_{harm_index:03d}",
                    "template_id": template["template_id"],
                    "category": template["category"],
                    "harm_index": harm_index,
                    "prompt": wrap_llama32_user(filled),
                }
            )
    return items


def load_jbb_behaviors(
    dataset: str, config: str, split: str, revision: str, set_name: str
) -> list[dict]:
    """Load one JailbreakBench behaviours split at a pinned dataset revision.

    Args:
        dataset: HuggingFace dataset id.
        config: Dataset config name.
        split: "harmful" or "benign".
        revision: Dataset commit to load, so the item set cannot drift.
        set_name: Result set name, used as the item id prefix.

    Returns:
        List of dicts with item_id, text, category and prompt.
    """
    from datasets import load_dataset

    data = load_dataset(dataset, config, split=split, revision=revision)
    return [
        {
            "item_id": f"{set_name}_{row['Index']:03d}",
            "text": row["Goal"],
            "category": row["Category"],
            "prompt": wrap_llama32_user(row["Goal"]),
        }
        for row in data
    ]


def load_pair_artifacts(path: Path, sha256: str, set_name: str) -> list[dict]:
    """Load PAIR attack prompts from a JailbreakBench artifact file.

    Entries where the attack produced no prompt are skipped.

    Args:
        path: Artifact JSON file.
        sha256: Pinned hash; the file is checked before it is read.
        set_name: Result set name, used as the item id prefix.

    Returns:
        List of dicts with item_id, text, category and prompt.
    """
    verify_model(path, sha256)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "item_id": f"{set_name}_{entry['index']:03d}",
            "text": entry["prompt"],
            "category": entry["category"],
            "prompt": wrap_llama32_user(entry["prompt"]),
        }
        for entry in data["jailbreaks"]
        if entry.get("prompt")
    ]
