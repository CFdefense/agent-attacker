# Implementation Plan: Adaptive Defense Evaluation

Goal for now: finish Project Deliverable 1 (PD1) by Sun Oct 25, 11:59 pm ET. This is a feasibility notebook, not the study. Background and research question are in the Deliverable 0 PDF in `docs/`.

The PD1 helper notebook (project 10) asks for: one open target model, a small benign/harmful/boundary prompt set, an undefended baseline with saved outputs, one candidate defense with a verified code path, and a defined success measure. Defense-aware attackers, fixed budgets and Pareto comparisons are listed as "later".

## Approach

Build a small Python package that runs a prompt through a target model, optionally through a defense, labels the result, and logs it to JSONL. The PD1 notebook imports it and runs a small matrix. The attack step is a swappable function, so the later attacker (random-search or an LLM agent) can be added without changing the rest. No agent framework: the later agent is a plain loop of attacker, defended target and judge.

Development happens on the M3 with a tiny model. The recorded run happens on a Colab T4 (about 16 GB); the notebook records the exact GPU.

## Layout

```
src/
  data.py       load and sample prompt sets (harmful, benign, boundary)
  model.py      load target model + tokenizer, generate, return logprobs
  defenses.py   defense wrappers: none, system_prompt, perplexity_filter
  judge.py      success label and refusal label
  jsonl_log.py  append-only JSONL writer, resume support
  run.py        run one (prompt, defense, attack) cell and log it
  attacks.py    static template (PD1); random-search suffix (stretch)
  metrics.py    ASR / FRR / utility / latency summaries
notebooks/
  farrell_christian_PD1.ipynb   copy of the helper, filled in
results/        JSONL + CSV outputs (small, committed)
```

## Build order

Each step is checked before starting the next.

1. **Environment.** `requirements.txt`, a seed helper, and a hardware report (CPU/GPU, versions). Tested on the M3 and on Colab.
2. **Prompt sets.** Load about 10 to 20 harmful behaviors (HarmBench or StrongREJECT subset), about 20 benign prompts (XSTest safe), and a few boundary prompts. Show one item and check for duplicates and missing fields. Cache a fixed sample file with the seed so the same prompts are used every run.
3. **Model and tokenizer.** Pick one 1.5B to 3B instruct model after comparing 2 or 3 candidates. Inspect tokenization of a normal and an odd input, and print the chat template. Generate on a few prompts and record load time and GPU memory.
4. **Undefended baseline.** Run all prompts with fixed decoding, log every response, and save to JSONL. This is the required core of PD1.
5. **Judge.** Start with a refusal-string check as the quick refusal label, plus one automatic judge for harmful success (HarmBench classifier or StrongREJECT evaluator, if it fits). Hand-label a small subset and report how often the judge and I agree.
6. **One defense.** Implement the cheapest candidate, a safety system prompt or a perplexity filter, behind the same interface as `none`. Rerun the same prompts and log them.
7. **Metrics.** Compute ASR on harmful prompts, false-refusal rate on benign prompts, one simple utility check (a few exact-match questions), query count and latency. Compare undefended and defended.
8. **Stretch: static vs adaptive.** Only if steps 1 to 7 are done early. Static: a fixed template plus a fixed suffix. Adaptive: random search over the suffix with budget N (one target-pipeline run per candidate), optimized against the defended pipeline. Small N, a few behaviors, one defense. The purpose is to show the loop runs and to time it.
9. **Notebook.** Fill in the helper sections from the logged results, with a short note on what changed since Deliverable 0. Add the GREEN, YELLOW or RED decision and the next step toward PD2. Export HTML.

## Per-query log row

`run_id, prompt_id, split, defense, attack, arm, query_idx, prompt, response, blocked, judge_label, refusal_label, human_label, latency_s, tokens_in, tokens_out`

Per-run config file: model ID and revision, chat template, decoding settings, defense settings, budget, seed, library versions, GPU.

## Decisions to make

| Decision | Leaning | How to decide |
|---|---|---|
| Harmful prompt set | HarmBench subset | Check that it loads and has a usable license. |
| Target model | 1.5B to 3B instruct, e.g. Qwen2.5-1.5B-Instruct | Compare 2 or 3 on model card, license, gating, and T4 memory. |
| First defense | System prompt or perplexity filter | Whichever is quicker to verify and run. |
| Judge | One fits-on-T4 classifier plus hand labels | Check size and license. |
| Utility check | About 10 exact-match questions | Keep it tiny; detects degraded answers. |
| Counts and budget N | See above | Set after timing a few runs. |

## Later (PD2 and beyond)

LLM-agent attacker (black-box, T refinement turns), second attack family, second model, 2 to 3 defenses, fixed budget rule across defenses, larger human-review set, erosion = ASR(adaptive) minus ASR(static) per defense. If compute is short, cut in this order: defenses, models, prompts and budget, then the second attack family. The adaptive comparison stays.

## Schedule

| Day | Work |
|---|---|
| Oct 22 | Verify candidates (model, prompt set, defense, judge). Steps 1 and 2. |
| Oct 23 | Steps 3 to 6 locally on a tiny model. |
| Oct 24 | Recorded run on Colab (steps 4 to 7, step 8 if time). Fill in the notebook. |
| Oct 25 | Export HTML, rerun top to bottom, submit. |

## Boundaries

Approved benchmark data and local open models only. Harmful completions stay in the results files; the notebook shows short sanitized examples. No tokens in the notebook (use Colab secrets). No paid services. Complete the AI Use Statement in the notebook.
