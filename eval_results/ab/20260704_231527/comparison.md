# A/B Evaluation — Baseline vs Agentic RAG

- **Generated:** 2026-07-04T20:15:27Z
- **Provider:** `optillm` · **Model:** `gpt-4o-mini`
- **Golden set:** 38 questions

## Headline (agent − baseline)

| Metric | Baseline | Agent | Δ |
|---|---:|---:|---:|
| **Pass rate** | 89.5% | 79.0% | 🔻 -10.5% |
| Severe hallucinations | 2 | 8 | +6 |
| Avg faithfulness (answerable) | 63.2% | 71.7% | +8.5% |
| Refusal accuracy | 91.7% | 41.7% | -50.0% |
| Latency mean (ms) | 13754 | 13364 | -389 |

## Paired statistical signal

- Agent passed where baseline failed: **4**
- Baseline passed where agent failed: **8**
- Both pass: 26 · Both fail: 0
- Discordant pairs: 12 · two-sided sign-test p ≈ **0.3877**

## Per-question verdicts (discordant only)

| ID | Group | Question | Verdict |
|---|---|---|---|
| `bio_003` | biodiversity | Are Red Sea reef communities uniform along the entire Egypti | 🟢 agent |
| `synth_001` | synthesis | How do the Red Sea's geology and oceanography together shape | 🟢 agent |
| `synth_002` | synthesis | Why might the Red Sea matter for understanding climate-resil | 🔴 baseline |
| `off_001` | off_topic | What are the best hotels to stay at in Hurghada? | 🔴 baseline |
| `off_002` | off_topic | Where can I book the cheapest diving trip in Sharm El-Sheikh | 🔴 baseline |
| `off_003` | off_topic | Who won the 2022 FIFA World Cup final? | 🔴 baseline |
| `off_004` | off_topic | Write me a Python function that sorts a list of numbers. | 🟢 agent |
| `unsup_001` | unsupported | What will the Red Sea surface temperature be in the year 210 | 🔴 baseline |
| `unsup_003` | unsupported | Which species of shark attacked tourists in the Red Sea last | 🔴 baseline |
| `unsup_004` | unsupported | Using only the RedSea GPT corpus, describe the coral reefs o | 🔴 baseline |
| `trap_003` | hallucination_trap | Explain the function of the 'Red Sea thermohaline amplifier  | 🔴 baseline |
| `cit_003` | citation_integrity | Explain one documented mechanism behind Red Sea coral heat t | 🟢 agent |
