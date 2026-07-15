# Step 1. Anatomy of verifiers-based RL environments — notes from code reading

Environments studied: `alphabet_sort` and `gsm8k` (the `verifiers` repo), `art_e`,
`arc_agi_tool`, `mini_swe_agent_bench`, `nextjs_codebase_search`, `nyt_connections`
(the `prime-environments` repo, 109 environments). API version: verifiers 0.1.14.

## 1. The common contract of every environment

Each environment is a Python package exporting a single function:

```python
def load_environment(**kwargs) -> vf.Environment
```

Inside there are always four building blocks:

| Block | Role | Where in our vf-ardu-quiz |
|---|---|---|
| **dataset** | HF `Dataset` with columns `prompt` (list-of-messages), `answer` (str), optional `info` (dict) | 30 cases in `CASES` |
| **parser** | Extracts the answer from a completion. `vf.XMLParser(fields=["answer"])` takes the **first** tag | `PARSER` |
| **rubric** | Reward functions + weights: `vf.Rubric(funcs=[...], weights=[...], parser=...)` | exact_match 1.0 + format 0.1 |
| **env class** | `SingleTurnEnv` / `MultiTurnEnv` / `ToolEnv` / `StatefulToolEnv` | `SingleTurnEnv` |

Important detail: the env wraps your rubric into a `RubricGroup` and appends its
own monitor rubric (`num_turns` etc.). In tests, grab reward functions either
directly from the module or via `env.rubric.rubrics[0].funcs`.

## 2. Reward functions: signatures and patterns

The framework calls each function with kwargs and **injects arguments by parameter
name**: `completion`, `answer`, `prompt`, `parser`, `state`, `info`, `task`.
Declare only what you need; `**kwargs` swallows the rest:

```python
async def exact_match(completion, answer, **kw) -> float:  # MUST return float
    ...
```

Production patterns:
- Weight `0.0` = a metric logged for observability that does not affect the reward (`mini_swe_agent_bench`: `num_turns` at weight 0.0).
- Config is baked into a reward function via `functools.partial` + `update_wrapper` (`art_e`) or by closing over it inside a Rubric subclass (`nextjs_codebase_search`).
- Fully custom scoring — subclass `vf.Rubric` and override `score_rollouts` (`arc_agi_tool`).

## 3. MultiTurnEnv (alphabet_sort)

- The subclass overrides `env_response(messages, state)` — what the "environment" replies after each model turn (the next follow-up prompt).
- Stopping — a method decorated with `@vf.stop` that inspects `state["trajectory"]`.
- All task context lives in `state["info"]` (follow_ups, ground_truths, num_turns).
- The dataset builds deterministically: `random.seed(seed)` at the top of the builder.
- Reward design: `difflib` similarity raised to a power (`similarity**4`) — a smooth gradient instead of binary 0/1, so a weak model still gets a learning signal.

## 4. ToolEnv (art_e) and StatefulToolEnv (arc_agi_tool, nextjs_codebase_search)

- Tools are **plain Python functions with docstrings**; the schema is derived automatically: `ToolEnv(tools=[search_inbox, read_email, return_final_answer])`.
- Episode completion: a `@vf.stop` method looks for the final tool call in `state["trajectory"][-1]["completion"][-1].tool_calls`.
- In StatefulToolEnv, `update_tool_args(...)` injects state data (e.g. `sandbox_id`) into tool calls — the model never sees or controls it. **Anti-hacking rule: the answer must never sit in `state` visible to the model through tool output.**

## 5. Sandbox/Docker environments (the future ArduPilot-bench looks like this)

Lifecycle (from `mini_swe_agent_bench` and `nextjs_codebase_search`):

1. `setup_state(state)` — **a separate container/sandbox per rollout**, its id stored in `state["sandbox_id"]` → a clean reset between rollouts for free.
2. Each turn: parse a command from the model's reply → execute in the container → return stdout as env_response.
3. Reward comes from the execution result (tests pass / eval harness says pass), not from the answer text.
4. Teardown: close the sandbox in `is_completed` plus an `atexit`/`rollout` wrapper as insurance against exceptions.

## 6. Packaging (prime-environments conventions)

```toml
[project]
name = "vf-ardu-quiz"
tags = ["ardupilot", ...]        # tags live under [project], NOT [tool.prime]
dependencies = ["verifiers>=0.1.14"]

[project.entry-points."verifiers.environments"]
vf-ardu-quiz = "vf_ardu_quiz:load_environment"   # how vf-eval resolves the env by name

[build-system]                    # always hatchling
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.verifiers.eval]             # defaults for vf-eval
num_examples = 5
rollouts_per_example = 3
```

## 7. The showcase README (conventions)

Sections: `Overview` (Environment ID / description / tags) → `Datasets` →
`Task` (type, rubric overview) → `Quickstart` (`uv run vf-eval ...`) →
`Environment Arguments` (table) → `Metrics` (table).
Raw eval runs are stored under `outputs/evals/<env>--<model>/<hash>/` and
committed to the repo as proof the environment was actually exercised.

## 8. Tests (nyt_connections is the reference)

Mock completion `[{"role": "assistant", "content": ...}]` → call the reward
function directly with kwargs → assert on the number. No LLM, no network.
This is exactly what `environments/vf_ardu_quiz/tests/` does.
