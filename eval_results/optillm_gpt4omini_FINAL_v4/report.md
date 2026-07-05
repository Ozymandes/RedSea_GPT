# Evaluation Report — optillm_gpt4omini_FINAL_v4

- **Generated:** 2026-07-05T09:12:00Z
- **Provider:** `optillm` (anthropic_messages)
- **Model:** `gpt-4o-mini`
- **Base URL:** `https://optollm.optomatica.com/v1`

## Headline metrics

- Total questions: **38**
- Pass rate: **100.0%** (38/38)
- Severe hallucinations: **0**
- Avg concept coverage (answerable): **96.2%**
- Avg faithfulness — claim entailment (answerable): **51.5%**
- Avg faithfulness — n-gram overlap (answerable, conservative): 66.6%
- Faithfulness methods used: llm_claim_entailment=26
- Citation support (answerable): **100.0%**
- Refusal accuracy: **100.0%** (12/12)
- Latency: mean **19252ms** · p95 **26985ms** · max **29523ms**

## By group

| Group | Pass | Total |
|---|---:|---:|
| biodiversity | 4 | 4 |
| citation_integrity | 3 | 3 |
| conservation | 4 | 4 |
| coral_heat | 4 | 4 |
| geology | 4 | 4 |
| hallucination_trap | 4 | 4 |
| oceanography | 4 | 4 |
| off_topic | 4 | 4 |
| synthesis | 3 | 3 |
| unsupported | 4 | 4 |

## Per-question detail

### ✅ `geo_001` — How did the Red Sea form geologically?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 27645ms
- Concept coverage: 1.0 · Faithfulness: 0.6667 · Citations supported: [2, 3, 4, 5]

### ✅ `geo_002` — What makes the Red Sea geologically different from a normal mature ocean basin?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 22963ms
- Concept coverage: 1.0 · Faithfulness: 0.7273 · Citations supported: [1, 2, 3, 5, 7]

### ✅ `geo_003` — How is the Red Sea connected to seafloor spreading and rifting?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 21580ms
- Concept coverage: 1.0 · Faithfulness: 0.45 · Citations supported: [1, 2, 3, 4, 6, 7]

### ✅ `geo_004` — What is the geological relationship between the Red Sea and the Gulf of Aden?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 19259ms
- Concept coverage: 1.0 · Faithfulness: 0.3 · Citations supported: [3, 6, 7]

### ✅ `ocean_001` — Why is the Red Sea so much saltier than other seas?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 19906ms
- Concept coverage: 0.5 · Faithfulness: 0.619 · Citations supported: [1, 2, 3, 4, 6, 7]

### ✅ `ocean_002` — What is the typical salinity of the Red Sea?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 18146ms
- Concept coverage: 1.0 · Faithfulness: 0.6667 · Citations supported: [1, 4, 5]

### ✅ `ocean_003` — How does the Red Sea's water circulation work, and how does evaporation drive it?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 25199ms
- Concept coverage: 1.0 · Faithfulness: 0.4545 · Citations supported: [1, 2, 4, 6, 7]

### ✅ `ocean_004` — What are the main water-exchange processes between the Red Sea and the open ocean?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 25848ms
- Concept coverage: 1.0 · Faithfulness: 0.16 · Citations supported: [3, 4, 5, 6, 7]

### ✅ `coral_001` — Why are some Red Sea corals unusually tolerant of high temperatures?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 23642ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [1, 2, 4, 5, 6, 7]

### ✅ `coral_002` — How do Red Sea corals respond to thermal stress, and what cellular mechanisms are involved?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 23380ms
- Concept coverage: 1.0 · Faithfulness: 0.4138 · Citations supported: [2, 3, 4, 6, 7]

### ✅ `coral_003` — What role do symbiotic algae play in coral health and bleaching?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 20315ms
- Concept coverage: 0.5 · Faithfulness: 0.8846 · Citations supported: [3, 4, 7]

### ✅ `coral_004` — What did the 2023 bleaching event along the Egyptian Red Sea coast reveal about coral vulnerability?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 23762ms
- Concept coverage: 1.0 · Faithfulness: 0.619 · Citations supported: [1, 2, 4, 6]

### ✅ `bio_001` — What does endemism mean in the Red Sea context, and why is it notable there?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 17597ms
- Concept coverage: 1.0 · Faithfulness: 0.2778 · Citations supported: [4, 6, 7]

### ✅ `bio_002` — Why does the Red Sea support such distinctive marine biodiversity?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 17070ms
- Concept coverage: 1.0 · Faithfulness: 0.2917 · Citations supported: [1, 4, 5, 6]

### ✅ `bio_003` — Are Red Sea reef communities uniform along the entire Egyptian coast, or do they vary?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 16717ms
- Concept coverage: 1.0 · Faithfulness: 0.6923 · Citations supported: [1, 2, 3, 4, 6]

### ✅ `bio_004` — What types of coral reefs occur in the Red Sea?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 15275ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [2, 5, 6, 7]

### ✅ `cons_001` — What are the main threats to Egyptian Red Sea coral reefs?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 17408ms
- Concept coverage: 1.0 · Faithfulness: 0.4444 · Citations supported: [1, 2, 3, 5]

### ✅ `cons_002` — How does tourism affect Red Sea coral reef ecosystems?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 16618ms
- Concept coverage: 1.0 · Faithfulness: 0.4 · Citations supported: [1, 6, 7]

### ✅ `cons_003` — What conservation strategies are relevant for Red Sea reefs?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 26985ms
- Concept coverage: 1.0 · Faithfulness: 0.4545 · Citations supported: [1, 2, 5, 6, 7]

### ✅ `cons_004` — How has climate change affected the Red Sea ecosystem so far?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 20603ms
- Concept coverage: 1.0 · Faithfulness: 0.7692 · Citations supported: [1, 2, 3, 4, 5]

### ✅ `synth_001` — How do the Red Sea's geology and oceanography together shape its marine life?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 20742ms
- Concept coverage: 1.0 · Faithfulness: 0.3636 · Citations supported: [3, 4, 5, 6, 7]

### ✅ `synth_002` — Why might the Red Sea matter for understanding climate-resilient coral reefs worldwide?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 19108ms
- Concept coverage: 1.0 · Faithfulness: 0.72 · Citations supported: [1, 3, 5, 6, 7]

### ✅ `synth_003` — How do high salinity, high temperature, and geographic isolation interact to shape Red Sea ecology?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 29523ms
- Concept coverage: 1.0 · Faithfulness: 0.25 · Citations supported: [1, 2, 3, 5, 6]

### ✅ `off_001` — What are the best hotels to stay at in Hurghada?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 16002ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_002` — Where can I book the cheapest diving trip in Sharm El-Sheikh?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 13855ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_003` — Who won the 2022 FIFA World Cup final?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 11883ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_004` — Write me a Python function that sorts a list of numbers.
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 15682ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_001` — What will the Red Sea surface temperature be in the year 2100?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 19221ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_002` — What are today's tide times for the port of Safaga?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 17919ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_003` — Which species of shark attacked tourists in the Red Sea last month?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 12017ms
- Concept coverage: 1.0 · Faithfulness: 1.0 · Citations supported: []

### ✅ `unsup_004` — Using only the RedSea GPT corpus, describe the coral reefs of the Mediterranean Sea.
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 16378ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_001` — What is 'Calcium Carbonate Resin' and what role does it play in Red Sea coral skeletons?
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 23174ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_002` — List five fish species from the corpus that are not actually mentioned anywhere in the sources.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 12470ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_003` — Explain the function of the 'Red Sea thermohaline amplifier gene' in coral heat tolerance.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 15899ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_004` — Describe the 'Hurghada Deep' hydrothermal brine pool reportedly discovered in 2019.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 15991ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `cit_001` — What is the average salinity of the Red Sea, and which source reports it?
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 15747ms
- Concept coverage: 1.0 · Faithfulness: 0.7778 · Citations supported: [3, 6]

### ✅ `cit_002` — How was the Red Sea formed? Cite the geological source(s) you used.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 18858ms
- Concept coverage: 1.0 · Faithfulness: 0.7222 · Citations supported: [1, 2, 3, 4, 5, 6, 7]

### ✅ `cit_003` — Explain one documented mechanism behind Red Sea coral heat tolerance, with a citation.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 17198ms
- Concept coverage: 1.0 · Faithfulness: 0.2727 · Citations supported: [5]
