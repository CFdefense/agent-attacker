# agent-attacker

Model Defense Evaluation Using Adaptive Attack Agents (MSCS 613, NLP).

## Summary

How much does a jailbreak defense erode when the attacker knows about it and gets the same attack budget? And what does the remaining protection cost in false refusals and lost usefulness?

The plan is to compare a static attack with an adaptive one against a defended open model, and to report attack success rate, false-refusal rate, utility, query count and latency.

## Progress

- **Deliverable 0** (framing and dependency map): done. `docs/farrell_christian_project_deliverable_0.pdf`.
  - Research question: under a fixed attack budget, how much does a language model's security erode when the attacker can observe the defense and adapt, and how does that affect false refusals on benign requests?
  - Smallest useful study: harmful and benign prompt sets, an undefended baseline, and a defended model compared against a static attack and an adaptive one. The main measure is attack success rate (ASR), with false-refusal rate (FRR), query count and latency alongside it.
  - Dependency map: data (a fixed HarmBench or StrongREJECT subset), models and defenses, static and adaptive attack methods, compute, and evaluation with human review. Its main risk was not having enough compute to host the models. The backup scope was one open model, one attack family and two defenses.
  - Starting source: Andriushchenko, Croce and Flammarion, *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks* (ICLR 2025).
- **Deliverable 1** (feasibility notebook): done, with a YELLOW verdict. `notebooks/farrell_christian_PD1.ipynb`, outputs in `notebooks/pd1_outputs/`.
  - Resources fixed: `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa79` (chosen over SmolLM2-1.7B-Instruct and Llama-3.2-1B-Instruct), HarmBench standard behaviors and XSTest at pinned commits, and 10 own utility questions. Seed 613.
  - Pipeline: an undefended baseline, a safety system prompt defense, a random-search suffix attack (static and adaptive arms), JSONL logs and summary metrics. It ran on Apple MPS in fp16 with about 2.9 GB of memory, and the 50-prompt set across both configurations took about 5 minutes.
  - Result: rule-based ASR was 2/15 undefended and 1/15 defended. With a 50-query suffix search it was 2/10 (static, undefended), 1/10 (static, defended) and 1/10 (adaptive, defended). The search raised the log-probability of "Sure" but never made it the top token.
  - Why YELLOW: 50 queries was too few to show a static versus adaptive gap, and the refusal-string judge had visible false positives. The recommended scope was one model and two defense setups. The feedback was that the measurements, not a larger budget, needed fixing first. The five XSTest items it called "boundary" are unsafe contrast items (renamed in Deliverable 2).
- **Deliverable 2** (readiness package): done. The notebook is `notebooks/farrell_christian_PD2.ipynb` (HTML copy in `notebooks/html/`), with outputs in `notebooks/pd2_outputs/`. What changed:
  - **Review layer.** All 60 harmful responses, the 12 safe-prompt responses the rule flagged as refusals, and 8 sampled unflagged safe-prompt responses were reviewed (80 rows). The rule label and the review label sit in separate columns. An AI assistant drafted the labels and the author approved them, and the notebook says so. Reported attack success is now the reviewed one.
  - **Threat model stated as implemented.** The attacker is white-box and reads first-token log-probabilities. The search score is the log-probability of "Sure", which is a proxy for compliance and not a measure of harm. The budget is 50 queries per prompt, and the search stops early only if "Sure" becomes the top token.
  - **Prompt categories separated.** The five XSTest unsafe items are `unsafe_contrast`, not a safe boundary set. False refusal is estimated on the 20 XSTest safe prompts only. The utility check is answer-string matching (whole word).
  - **Fingerprinted cache keys.** Keys include the model, revision, dtype, decoding (including the repetition penalty), seed, judge rules, defense text and attack settings, and the full prompt text. Every scored search candidate is saved with its score. Interrupted writes are cleaned up on resume.
  - **Pilot result.** Reviewed attack success was 1 success in every condition (1/15 with no attack, 1/10 with a suffix), so there is no adaptive gain. Rule-based false refusals were 8/20 undefended and 4/20 defended. Reviewed, they were 4/20 and 3/20. Utility was 10/10 for both defenses. With 10 to 15 prompts per condition this cannot separate the conditions, and it does not show that the defense is robust or that 50 queries is too small.

### Left to do

**Main experiment checkpoint**
- Replace the string rule with a validated judge (the HarmBench classifier or StrongREJECT) and report its agreement with the reviewed labels. Whether either fits on a Colab T4 has not been checked.
- Freeze a larger final evaluation set (target 50 harmful behaviors and the full XSTest safe set) that excludes every development prompt.
- Run the main experiment with search seeds 613 to 615 and report reviewed rates with confidence intervals.
- Run a 200-query budget check on about 10 behaviors to see whether the "Sure" score ever turns into reviewed compliance.

**Later**
- Stretch: a black-box LLM attacker that reads refusals and rewrites the request, as a separate attack arm. Dropped first if time runs short.
- Second defense and a second model, only if time allows.
- Compute the erosion (adaptive minus static attack success, per defense), analyze the security versus utility tradeoff, and write up.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```
uv venv
uv sync
```

`uv venv` creates `.venv`. `uv sync` installs the packages from `pyproject.toml` and `uv.lock`, including the dev tools (basedpyright and pandas-stubs).

## Run

```
uv run python src/main.py --out notebooks/pd2_outputs --budget 50
```

Runs the baseline, the defense and the static versus adaptive attack, then writes `summary.csv`. It resumes from the logs in `--out`.

## Review sheet

```
uv run python src/review.py --out notebooks/pd2_outputs
```

Writes `review_sheet.csv` with the rule-based `auto_label` and an empty `human_label` column. It never overwrites an existing sheet, and it refuses to build one from a log that holds rows from more than one setup.

## Tests and checks

```
uv run pytest
uv run basedpyright
ruff check src tests
```

## Layout

- `src/` Source code
  - `main.py` Runs the whole experiment from the command line
  - `data.py` The prompts the model is tested on: harmful, safe, unsafe-contrast and simple factual questions
  - `model.py` The model being attacked: generates answers and reports how likely it is to start with a given word
  - `defenses.py` The protections placed in front of the model, including "no protection" as the baseline
  - `attacks.py` The attackers that try to get past a defense
  - `judge.py` Decides whether a response was a refusal, a successful attack, or a correct answer
  - `run.py` Sends a prompt through a defense and the model, labels the result, and records it
  - `jsonl_log.py` Saves every result to disk so a run can be stopped and resumed
  - `metrics.py` Turns the saved results into the summary numbers
  - `review.py` Builds the sheet of responses to check by hand and summarizes the checked labels next to the automatic ones
- `tests/` Unit tests that run without downloading the model
- `notebooks/` The Deliverable 1 and 2 write-ups (`.ipynb`, with HTML copies in `html/`)
- `notebooks/pd1_outputs/` and `notebooks/pd2_outputs/` Saved outputs from the two runs: every prompt, response, search candidate, label and timing, plus the summary table and run settings
- `docs/` The Deliverable 0 submission and the project plan
