# agent-attacker

Model Defense Evaluation Using Adaptive Attack Agents (MSCS 613, NLP).

## Summary

How much does a jailbreak defense erode when the attacker knows about it and gets the same attack budget? And what does the remaining protection cost in false refusals and lost usefulness?

The plan is to compare a static attack with an adaptive one against a defended open model, and to report attack success rate, false-refusal rate, utility, query count and latency.

## Progress

- **Deliverable 0** (framing and dependency map): done.
- **Deliverable 1** (feasibility notebook, due Oct 25): the pipeline runs end to end on one small model (Qwen2.5-1.5B-Instruct) with HarmBench and XSTest prompts, an undefended baseline, a safety system prompt defense, a random-search suffix attack, and logged results in `results/pd1/`. Verdict so far is inconclusive: with a 50-query budget the attack was too weak to show a static versus adaptive gap, and the judge is a simple refusal-string check.

### Left to do

**Deliverable 2** (full experiment plan)
- Fix the experiment matrix, variables, seeds and statistics.
- Read the three key papers (Andriushchenko et al.; SafeInt; adaptive evaluation of prompt-injection defenses) and fix the threat model, defenses and attack budget rule.
- Add a stronger judge (maybe LLM) and a small hand-labelled set to check it.
- Make the undefended and defended runs differ only in the defense, and add a harder utility set.
- Raise the attack budget (hundreds of queries) until the attack actually works, then scale up behaviors.

**Deliverable 3 and later**
- Second attack family (an LLM attacker that reads refusals), second defense, and ideally a second model.
- Run the full study and compute the erosion: adaptive attack success minus static attack success, per defense.
- Analyze the security versus utility tradeoff and write up.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer.

```
uv venv
uv sync
```

`uv venv` creates `.venv`. `uv sync` installs the packages from `pyproject.toml` and `uv.lock`, including the dev tools (basedpyright and pandas-stubs).

## Run

```
uv run python src/main.py --out results/pd1 --budget 50
```

Runs the baseline, the defense and the static versus adaptive attack, then writes `summary.csv`. It resumes from the logs in `--out`.

## Type check

```
uv run basedpyright
```

## Layout

- `src/` Source code
  - `main.py` Runs the whole experiment from the command line
  - `data.py` The prompts the model is tested on: harmful, harmless, borderline and simple factual questions
  - `model.py` The model being attacked: generates answers and reports how likely it is to start with a given word
  - `defenses.py` The protections placed in front of the model, including "no protection" as the baseline
  - `attacks.py` The attackers that try to get past a defense
  - `judge.py` Decides whether a response was a refusal, a successful attack, or a correct answer
  - `run.py` Sends a prompt through a defense and the model, labels the result, and records it
  - `jsonl_log.py` Saves every result to disk so a run can be stopped and resumed
  - `metrics.py` Turns the saved results into the summary numbers
- `notebooks/` The Deliverable 1 write-up with the results
- `results/` Saved outputs from runs: every prompt, response, label and timing, plus the summary table
- `docs/` The Deliverable 0 submission and the project plan
