# A/B Evaluation — Baseline vs Agentic RAG

- **Generated:** 2026-07-04T21:05:53Z
- **Provider:** `optillm` · **Model:** `gpt-4o-mini`
- **Golden set:** 38 questions

## Headline (agent − baseline)

| Metric | Baseline | Agent | Δ |
|---|---:|---:|---:|
| **Pass rate** | 92.1% | 92.1% | → +0.0% |
| Severe hallucinations | 2 | 1 | -1 |
| Avg faithfulness (answerable) | 61.1% | 67.8% | +6.7% |
| Refusal accuracy | 91.7% | 91.7% | +0.0% |
| Latency mean (ms) | 12961 | 17118 | +4157 |

## Paired statistical signal

- Agent passed where baseline failed: **2**
- Baseline passed where agent failed: **2**
- Both pass: 33 · Both fail: 1
- Discordant pairs: 4 · two-sided sign-test p ≈ **1.0**
- ⚠️ **Low-power caveat:** only 4 discordant pairs; treat the direction as suggestive, not statistically conclusive.

## Per-question verdicts (discordant only)

| ID | Group | Question | Verdict |
|---|---|---|---|
| `bio_003` | biodiversity | Are Red Sea reef communities uniform along the entire Egypti | 🟢 agent |
| `cons_002` | conservation | How does tourism affect Red Sea coral reef ecosystems? | 🔴 baseline |
| `off_004` | off_topic | Write me a Python function that sorts a list of numbers. | 🟢 agent |
| `trap_002` | hallucination_trap | List five fish species from the corpus that are not actually | 🔴 baseline |
