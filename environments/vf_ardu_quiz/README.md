# vf-ardu-quiz

### Overview
- **Environment ID**: `vf-ardu-quiz`
- **Short description**: Single-turn ArduPilot dataflash-log diagnosis quiz — read log symptoms, pick the root cause from a closed 15-label taxonomy, scored by a deterministic exact-match verifier.
- **Tags**: ardupilot, diagnosis, single-turn, xml, eval, train

### Datasets
- **Primary dataset(s)**: 30 hand-written diagnostic cases bundled in the module (`CASES` in `vf_ardu_quiz.py`), each pairing realistic dataflash-log evidence (POWR/MAG/VIBE/RCOU/EV/CTUN traces, FFT observations) with one root-cause label.
- **Split sizes**: 30 eval cases, exactly 2 per label — perfectly balanced so blind constant-guessing is bounded at 1/15 (~6.7%).

### Task
- **Type**: single-turn
- **Output format**: one snake_case label inside `<answer>...</answer>` tags; free-form reasoning before the tags is allowed.
- **Rubric overview**:
  - `exact_match` (weight 1.0) — parsed `<answer>` label equals ground truth (case/whitespace-normalized).
  - `format_reward_func` (weight 0.1) — XMLParser format adherence bonus.

### Quickstart

```bash
# from Prime Intellect Hub
prime env install <owner>/vf-ardu-quiz
# or from source (run inside this environment's directory)
uv pip install -e .

uv run vf-eval vf-ardu-quiz --provider prime -m meta-llama/llama-3.3-70b-instruct -n 30 -r 2
```

Fully offline (no API key) against the bundled mock server:

```bash
python tools/mock_openai_server.py 8077 &
MOCK_API_KEY=dummy vf-eval vf-ardu-quiz -m oracle \
  --api-base-url http://127.0.0.1:8077/v1 --api-key-var MOCK_API_KEY \
  -n 30 -r 1 --disable-tui
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `-1` | Truncate the dataset to the first N cases (-1 = all 30). |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted sum: `1.0 * exact_match + 0.1 * format_reward_func`. |
| `exact_match` | 1.0 if the parsed label equals ground truth, else 0.0. |
| `format_reward_func` | XMLParser format adherence in [0, 1]. |

### Eval results (real models via Prime Inference, n=30, r=2)

| Model | exact_match | reward | format |
| ----- | ----------- | ------ | ------ |
| `meta-llama/llama-3.3-70b-instruct` | 0.933 | 1.033 | 1.0 |
| `meta-llama/Llama-3.2-3B-Instruct` | 0.500 | 0.600 | 1.0 |

Both models sit inside the useful difficulty band with a clear capability gap
(smooth gradient). The 3B run is strictly bimodal — 15 cases solved in both
rollouts, 15 failed in both — so the dataset splits cleanly into easy/hard
halves for small models. The 70B model flickers on only 2 cases and stably
fails 1: those are the natural candidates for hardening the dataset further.
Raw runs live under `outputs/evals/`.

### Verifier sanity ladder (scripted mock policies, full 30-case sweep, r=1)

Run offline via `tools/mock_openai_server.py`; raw runs are stored under `outputs/evals/`.

| Policy (mock model) | exact_match | reward | Interpretation |
| ------------------- | ----------- | ------ | -------------- |
| `oracle` | 1.000 | 1.080 | Ceiling check — verifier accepts every correct answer. |
| `strong-80` | 0.833 | 0.913 | Strong-model proxy. |
| `weak-40` | 0.433 | 0.513 | Weak-model proxy — smooth difficulty gradient exists. |
| `guesser` (constant label) | 0.067 | 0.147 | Matches the theoretical 1/15 constant-guess bound. |
| `no-format` (right answer, no tags) | 0.000 | 0.020 | Format channel isolated from correctness. |

Determinism: repeat runs reproduce these numbers exactly (hash-seeded policies, fixed dataset).

### Tests

17 pytest cases in `tests/test_vf_ardu_quiz.py` cover the reward functions without any LLM:
unit tests (correct/wrong/reasoning-prefix/normalization), anti-reward-hacking audit
(empty completion, prompt echo, untagged answer, `<answer>`-spam collapses to first tag,
formatted garbage capped at the 0.1 format bonus, constant-guess rate ≤ 20%),
dataset sanity (balance, taxonomy coverage, unique symptoms) and determinism.

```bash
cd environments/vf_ardu_quiz && python -m pytest tests/ -q
```
