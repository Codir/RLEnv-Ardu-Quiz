"""Unit + anti-reward-hacking tests for vf-ardu-quiz.

Layer 3.1: reward-function unit tests (no LLM involved).
Layer 3.2: anti-hacking audit — every known exploit gets a locking test.
Layer 3.4: determinism — same inputs, same scores/dataset.

Tests are sync and drive async reward funcs via asyncio.run() so the suite
needs nothing beyond pytest itself.
"""

import asyncio
from collections import Counter

from vf_ardu_quiz import CASES, PARSER, TAXONOMY, exact_match, load_environment

ANSWER = "power_supply_sag"


def _completion(text: str):
    return [{"role": "assistant", "content": text}]


def _em(text: str, answer: str = ANSWER) -> float:
    return asyncio.run(exact_match(completion=_completion(text), answer=answer))


def _fmt(text: str) -> float:
    fmt_func = PARSER.get_format_reward_func()
    result = fmt_func(completion=_completion(text))
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return float(result)


# --- 3.1 unit tests: exact_match ---------------------------------------------


def test_correct_answer_scores_one():
    assert _em("<answer>power_supply_sag</answer>") == 1.0


def test_wrong_answer_scores_zero():
    assert _em("<answer>gps_glitch</answer>") == 0.0


def test_reasoning_before_tags_still_scores():
    text = "Vcc sags on current peaks, so...\n<answer>power_supply_sag</answer>"
    assert _em(text) == 1.0


def test_whitespace_and_case_normalized():
    assert _em("<answer>  Power_Supply_Sag \n</answer>") == 1.0


def test_reward_is_float():
    for text in ("<answer>power_supply_sag</answer>", "<answer>x</answer>", ""):
        assert isinstance(_em(text), float)


# --- 3.2 anti-hacking audit ---------------------------------------------------


def test_empty_completion_scores_zero():
    assert _em("") == 0.0


def test_prompt_echo_scores_zero():
    echo = "Symptoms from the dataflash log: POWR.Vcc drops from 5.0V to 4.3V..."
    assert _em(echo) == 0.0


def test_answer_without_tags_scores_zero():
    # The verifier must not reward the right label outside the required format.
    assert _em("power_supply_sag") == 0.0


def test_label_spam_collapses_to_first_tag():
    # Exploit attempt: emit every taxonomy label in its own <answer> tag.
    # The parser takes the FIRST tag only, so this degenerates into a
    # constant guess and cannot beat honest answering.
    spam = "".join(f"<answer>{label}</answer>" for label in TAXONOMY)
    parsed = PARSER.parse_answer(_completion(spam))
    assert parsed == TAXONOMY[0]
    assert _em(spam, answer=TAXONOMY[0]) == 1.0
    assert _em(spam, answer=TAXONOMY[1]) == 0.0


def test_formatted_garbage_gets_only_format_bonus():
    # Right format, meaningless answer -> only the 0.1-weighted format bonus.
    garbage = "<answer>flux_capacitor_misalignment</answer>"
    weighted = 1.0 * _em(garbage) + 0.1 * _fmt(garbage)
    assert weighted <= 0.1


def test_constant_guessing_is_bounded():
    # Blind constant guessing with the most frequent label must stay under
    # the 20% red line from the review checklist.
    counts = Counter(label for _, label in CASES)
    best_constant_rate = max(counts.values()) / len(CASES)
    assert best_constant_rate <= 0.20
    # Dataset is exactly balanced: every label appears the same number of times.
    assert set(counts.values()) == {2}


# --- dataset sanity -----------------------------------------------------------


def test_all_answers_in_taxonomy():
    assert all(label in TAXONOMY for _, label in CASES)


def test_all_taxonomy_labels_used():
    assert set(label for _, label in CASES) == set(TAXONOMY)


def test_symptoms_are_unique():
    symptoms = [symptom for symptom, _ in CASES]
    assert len(symptoms) == len(set(symptoms))


def test_dataset_schema_and_truncation():
    env = load_environment()
    ds = env.dataset() if callable(env.dataset) else env.dataset
    assert len(ds) == len(CASES) == 30
    row = ds[0]
    # SingleTurnEnv prepends the system prompt to each rollout's messages.
    assert isinstance(row["prompt"], list)
    assert row["prompt"][0]["role"] == "system"
    assert any(m["role"] == "user" for m in row["prompt"])
    assert row["answer"] in TAXONOMY

    small = load_environment(num_examples=5)
    ds_small = small.dataset() if callable(small.dataset) else small.dataset
    assert len(ds_small) == 5


# --- 3.4 determinism ----------------------------------------------------------


def test_dataset_is_deterministic_across_loads():
    def snapshot():
        env = load_environment()
        ds = env.dataset() if callable(env.dataset) else env.dataset
        return [(r["prompt"][0]["content"], r["answer"]) for r in ds]

    assert snapshot() == snapshot()


def test_scoring_is_deterministic():
    text = "<answer>power_supply_sag</answer>"
    assert [_em(text) for _ in range(5)] == [1.0] * 5
