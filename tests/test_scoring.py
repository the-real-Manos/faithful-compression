"""Tests for GSM8K and ARC-Challenge prompt formatting, extraction and scoring.

Prompt formatting lives in data/loaders.py but is tested here rather than in a
separate file, because format and extraction are one contract: a template change
that renumbers the options silently invalidates the extractor.
"""

import pytest

from faithful_compression.data.loaders import format_arc_prompt, format_gsm8k_prompt
from faithful_compression.eval.scoring import (
    extract_arc_letter,
    extract_gsm8k_answer,
    normalise_arc_answer_key,
    normalise_arc_labels,
    normalise_number,
    score_arc,
    score_gsm8k,
)


class TestPromptFormatting:
    def test_gsm8k_prompt_uses_llama32_template(self) -> None:
        prompt = format_gsm8k_prompt("What is 2 + 2?")
        assert prompt.startswith("<|begin_of_text|><|start_header_id|>user<|end_header_id|>")
        assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_gsm8k_prompt_carries_the_answer_format_instruction(self) -> None:
        assert "#### " in format_gsm8k_prompt("What is 2 + 2?")

    def test_gsm8k_prompt_contains_the_question(self) -> None:
        assert "What is 2 + 2?" in format_gsm8k_prompt("What is 2 + 2?")

    def test_arc_prompt_labels_four_choices(self) -> None:
        prompt = format_arc_prompt("Why?", ["one", "two", "three", "four"])
        for letter, text in zip("ABCD", ["one", "two", "three", "four"], strict=True):
            assert f"{letter}) {text}" in prompt

    def test_arc_prompt_adapts_to_three_choices(self) -> None:
        prompt = format_arc_prompt("Why?", ["one", "two", "three"])
        assert "C) three" in prompt
        assert "D)" not in prompt
        assert "(A, B, or C)" in prompt

    def test_arc_prompt_adapts_to_five_choices(self) -> None:
        prompt = format_arc_prompt("Why?", ["a", "b", "c", "d", "e"])
        assert "E) e" in prompt
        assert "(A, B, C, D, or E)" in prompt

    def test_arc_prompt_rejects_zero_choices(self) -> None:
        with pytest.raises(ValueError):
            format_arc_prompt("Why?", [])

    def test_arc_prompt_rejects_too_many_choices(self) -> None:
        with pytest.raises(ValueError):
            format_arc_prompt("Why?", list("abcdef"))

    def test_arc_format_and_extractor_agree_on_letter_range(self) -> None:
        # The template's letters and the extractor's valid range must match, or
        # every item with an unusual choice count silently scores zero.
        for n in (3, 4, 5):
            choices = [f"choice {i}" for i in range(n)]
            prompt = format_arc_prompt("Why?", choices)
            last_letter = "ABCDE"[n - 1]
            assert f"{last_letter}) choice {n - 1}" in prompt
            assert extract_arc_letter(last_letter, n_choices=n) == last_letter


class TestNormaliseNumber:
    def test_strips_commas_and_whitespace(self) -> None:
        assert normalise_number(" 1,234 ") == "1234"

    def test_strips_currency_symbol(self) -> None:
        assert normalise_number("$72") == "72"

    def test_collapses_integral_float(self) -> None:
        assert normalise_number("4.0") == "4"

    def test_preserves_genuine_decimal(self) -> None:
        assert normalise_number("2.5") == "2.5"

    def test_handles_negative(self) -> None:
        assert normalise_number("-18") == "-18"

    def test_strips_trailing_full_stop(self) -> None:
        assert normalise_number("42.") == "42"

    def test_returns_none_for_non_numeric(self) -> None:
        assert normalise_number("forty two") is None

    def test_returns_none_for_empty(self) -> None:
        assert normalise_number("   ") is None


class TestExtractGsm8kAnswer:
    def test_extracts_after_marker(self) -> None:
        assert extract_gsm8k_answer("Working...\n#### 18") == "18"

    def test_takes_last_marker_when_several(self) -> None:
        assert extract_gsm8k_answer("#### 7\nOn reflection:\n#### 12") == "12"

    def test_strips_commas_in_answer(self) -> None:
        assert extract_gsm8k_answer("#### 1,200") == "1200"

    def test_handles_currency_after_marker(self) -> None:
        assert extract_gsm8k_answer("#### $35") == "35"

    def test_handles_no_space_after_marker(self) -> None:
        assert extract_gsm8k_answer("####42") == "42"

    def test_handles_negative_answer(self) -> None:
        assert extract_gsm8k_answer("#### -5") == "-5"

    def test_returns_none_when_marker_missing(self) -> None:
        assert extract_gsm8k_answer("The answer is 18.") is None

    def test_fallback_takes_last_number_when_enabled(self) -> None:
        assert extract_gsm8k_answer("First 3, then 18.", allow_fallback=True) == "18"

    def test_fallback_returns_none_when_no_numbers(self) -> None:
        assert extract_gsm8k_answer("no digits at all", allow_fallback=True) is None


class TestScoreGsm8k:
    def test_scores_exact_match_correct(self, gsm8k_items: list[dict]) -> None:
        correct, predicted, truth = score_gsm8k("so it is 4.\n#### 4", gsm8k_items[0]["answer"])
        assert (correct, predicted, truth) == (1, "4", "4")

    def test_scores_mismatch_incorrect(self, gsm8k_items: list[dict]) -> None:
        correct, predicted, truth = score_gsm8k("#### 5", gsm8k_items[0]["answer"])
        assert (correct, predicted, truth) == (0, "5", "4")

    def test_format_failure_scores_zero_not_error(self, gsm8k_items: list[dict]) -> None:
        correct, predicted, truth = score_gsm8k("The answer is 4.", gsm8k_items[0]["answer"])
        assert correct == 0
        assert predicted is None
        assert truth == "4"

    def test_comma_formatted_output_matches_plain_truth(self) -> None:
        correct, _, _ = score_gsm8k("#### 1,200", "long working\n#### 1200")
        assert correct == 1

    def test_all_fixture_items_self_score(self, gsm8k_items: list[dict]) -> None:
        for item in gsm8k_items:
            correct, _, _ = score_gsm8k(item["answer"], item["answer"])
            assert correct == 1


class TestArcLabelNormalisation:
    def test_letter_labels_map_to_themselves(self) -> None:
        assert normalise_arc_labels(["A", "B", "C", "D"]) == ["A", "B", "C", "D"]

    def test_numeric_labels_map_positionally(self) -> None:
        assert normalise_arc_labels(["1", "2", "3", "4"]) == ["A", "B", "C", "D"]

    def test_three_choice_item_supported(self) -> None:
        assert normalise_arc_labels(["A", "B", "C"]) == ["A", "B", "C"]

    def test_five_choice_item_supported(self) -> None:
        assert normalise_arc_labels(["1", "2", "3", "4", "5"]) == ["A", "B", "C", "D", "E"]

    def test_too_many_choices_raises(self) -> None:
        with pytest.raises(ValueError):
            normalise_arc_labels(list("123456"))

    def test_numeric_answer_key_translated(self) -> None:
        assert normalise_arc_answer_key("3", ["1", "2", "3", "4"]) == "C"

    def test_letter_answer_key_passthrough(self) -> None:
        assert normalise_arc_answer_key("D", ["A", "B", "C", "D"]) == "D"

    def test_unknown_answer_key_returns_none(self) -> None:
        assert normalise_arc_answer_key("Z", ["A", "B", "C", "D"]) is None


class TestExtractArcLetter:
    def test_bare_letter(self) -> None:
        assert extract_arc_letter("C") == "C"

    def test_letter_with_paren_and_text(self) -> None:
        assert extract_arc_letter("C) photosynthesis") == "C"

    def test_letter_in_parentheses(self) -> None:
        assert extract_arc_letter("(B)") == "B"

    def test_letter_inside_sentence(self) -> None:
        assert extract_arc_letter("The answer is D.") == "D"

    def test_lowercase_letter_recovered(self) -> None:
        assert extract_arc_letter("c") == "C"

    def test_capital_wins_over_later_lowercase(self) -> None:
        assert extract_arc_letter("I think B, not c") == "B"

    def test_letter_outside_choice_range_ignored(self) -> None:
        assert extract_arc_letter("E is my answer", n_choices=4) is None

    def test_fifth_letter_valid_for_five_choice_item(self) -> None:
        assert extract_arc_letter("E is my answer", n_choices=5) == "E"

    def test_returns_none_when_absent(self) -> None:
        assert extract_arc_letter("I am not sure about this one.") is None

    def test_rejects_invalid_choice_count(self) -> None:
        with pytest.raises(ValueError):
            extract_arc_letter("A", n_choices=0)


class TestScoreArc:
    def test_correct_letter_scores_one(self) -> None:
        correct, predicted, truth = score_arc("C", "C", ["A", "B", "C", "D"])
        assert (correct, predicted, truth) == (1, "C", "C")

    def test_numeric_labelled_item_scores_correctly(self) -> None:
        correct, predicted, truth = score_arc("C) sunlight", "3", ["1", "2", "3", "4"])
        assert (correct, predicted, truth) == (1, "C", "C")

    def test_wrong_letter_scores_zero(self) -> None:
        correct, _, _ = score_arc("A", "C", ["A", "B", "C", "D"])
        assert correct == 0

    def test_three_choice_item_scores(self) -> None:
        correct, predicted, truth = score_arc("The answer is B", "B", ["A", "B", "C"])
        assert (correct, predicted, truth) == (1, "B", "B")

    def test_extraction_failure_scores_zero(self) -> None:
        correct, predicted, _ = score_arc("I cannot tell.", "C", ["A", "B", "C", "D"])
        assert correct == 0
        assert predicted is None
