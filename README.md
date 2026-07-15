# RLEnv-Ardu-Quiz — full-cycle walkthrough of RL environments (Prime Intellect / verifiers)

A hands-on pass through the lifecycle of an RL environment:
toolchain → anatomy of existing environments → building an original environment →
four-layer testing → publication on the Environments Hub.

## Structure

```
environments/vf_ardu_quiz/      # the environment itself
├── vf_ardu_quiz.py             #   30 ArduPilot log-diagnosis cases, 15-label taxonomy
├── pyproject.toml              #   packaging per prime-environments conventions
├── README.md                   #   showcase README with eval tables
├── tests/test_vf_ardu_quiz.py  #   17 tests: unit + anti-reward-hacking + determinism
└── outputs/evals/              #   raw vf-eval runs (mock policies + real models)
tools/mock_openai_server.py     # OpenAI-compatible mock: 5 scripted policies for offline evals
docs/01-anatomy.md              # notes on environment anatomy (verifiers 0.1.14)
```

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install verifiers pytest
uv pip install -e environments/vf_ardu_quiz

# tests (no LLM, no network)
cd environments/vf_ardu_quiz && python -m pytest tests/ -q && cd ../..

# full eval loop offline (no API key required)
python tools/mock_openai_server.py 8077 &
MOCK_API_KEY=dummy vf-eval vf-ardu-quiz -m oracle \
  --api-base-url http://127.0.0.1:8077/v1 --api-key-var MOCK_API_KEY \
  -n 30 -r 1 --disable-tui
```

## Model discrimination results

Real models via Prime Inference (n=30, r=2):

| Model | exact_match | What it shows |
|---|---|---|
| meta-llama/llama-3.3-70b-instruct | 0.933 | strong model near ceiling |
| meta-llama/Llama-3.2-3B-Instruct | 0.500 | weak model scores well below strong — smooth difficulty gradient |

Scripted mock policies (offline sanity ladder, n=30, r=1):

| Policy | exact_match | What it proves |
|---|---|---|
| oracle | 1.000 | verifier accepts every correct answer |
| strong-80 | 0.833 | difficulty gradient exists |
| weak-40 | 0.433 | weak "model" scores below strong |
| guesser | 0.067 | constant guessing = theoretical 1/15, no exploit |
| no-format | 0.000 | format channel isolated from correctness |

## Definition of Done (status)

- [x] Toolchain alive: `prime` CLI, `verifiers` 0.1.14, `vf-eval`/`vf-init` working
- [x] Read the code of 7 environments (SingleTurn, MultiTurn, ToolEnv, StatefulToolEnv, 2×sandbox, tests) — `docs/01-anatomy.md`
- [x] Original environment: balanced dataset, 17 unit tests green
- [x] Anti-reward-hacking checklist passed, every exploit locked by a test
- [x] Determinism: repeat runs reproduce identical scores
- [x] Real evals via Prime Inference: Llama-3.2-3B = 0.500, Llama-3.3-70B = 0.933 — difficulty gradient confirmed
- [x] `prime env push` — environment published on the Environments Hub
