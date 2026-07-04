# Evaluation Report — optillm_gpt4omini_FINAL

- **Generated:** 2026-07-04T18:47:45Z
- **Provider:** `optillm` (anthropic_messages)
- **Model:** `gpt-4o-mini`
- **Base URL:** `https://optollm.optomatica.com/v1`

## Headline metrics

- Total questions: **38**
- Pass rate: **89.5%** (34/38)
- Severe hallucinations: **3**
- Avg concept coverage (answerable): **82.7%**
- Avg faithfulness (answerable): **59.7%**
- Citation support (answerable): **96.2%**
- Refusal accuracy: **91.7%** (11/12)
- Latency: mean **11011ms** · p95 **14284ms** · max **14622ms**

## By group

| Group | Pass | Total |
|---|---:|---:|
| biodiversity | 2 | 4 |
| citation_integrity | 3 | 3 |
| conservation | 3 | 4 |
| coral_heat | 4 | 4 |
| geology | 4 | 4 |
| hallucination_trap | 4 | 4 |
| oceanography | 4 | 4 |
| off_topic | 3 | 4 |
| synthesis | 3 | 3 |
| unsupported | 4 | 4 |

## Per-question detail

### ✅ `geo_001` — How did the Red Sea form geologically?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 13082ms
- Concept coverage: 1.0 · Faithfulness: 0.556 · Citations supported: [2, 3, 4]

### ✅ `geo_002` — What makes the Red Sea geologically different from a normal mature ocean basin?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12416ms
- Concept coverage: 0.5 · Faithfulness: 0.714 · Citations supported: [1, 2, 3, 5]

### ✅ `geo_003` — How is the Red Sea connected to seafloor spreading and rifting?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 14622ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [2, 4, 5]

### ✅ `geo_004` — What is the geological relationship between the Red Sea and the Gulf of Aden?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 11840ms
- Concept coverage: 1.0 · Faithfulness: 0.714 · Citations supported: [3, 4]

### ✅ `ocean_001` — Why is the Red Sea so much saltier than other seas?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 11196ms
- Concept coverage: 0.5 · Faithfulness: 0.545 · Citations supported: [1, 5]

### ✅ `ocean_002` — What is the typical salinity of the Red Sea?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 12440ms
- Concept coverage: 1.0 · Faithfulness: 0.8 · Citations supported: [1, 4, 5]

### ✅ `ocean_003` — How does the Red Sea's water circulation work, and how does evaporation drive it?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 12786ms
- Concept coverage: 1.0 · Faithfulness: 0.727 · Citations supported: [1, 2, 4, 5]

### ✅ `ocean_004` — What are the main water-exchange processes between the Red Sea and the open ocean?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 13799ms
- Concept coverage: 1.0 · Faithfulness: 1.0 · Citations supported: [2, 4, 5]

### ✅ `coral_001` — Why are some Red Sea corals unusually tolerant of high temperatures?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 13225ms
- Concept coverage: 1.0 · Faithfulness: 0.4 · Citations supported: [1, 2, 3, 5]

### ✅ `coral_002` — How do Red Sea corals respond to thermal stress, and what cellular mechanisms are involved?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 10868ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [1, 3]

### ✅ `coral_003` — What role do symbiotic algae play in coral health and bleaching?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 11288ms
- Concept coverage: 0.5 · Faithfulness: 1.0 · Citations supported: [3, 4, 5]

### ✅ `coral_004` — What did the 2023 bleaching event along the Egyptian Red Sea coast reveal about coral vulnerability?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 12207ms
- Concept coverage: 1.0 · Faithfulness: 0.333 · Citations supported: [1, 3, 4]

### ❌ `bio_001` — What does endemism mean in the Red Sea context, and why is it notable there?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 11319ms
- Concept coverage: 0.0 · Faithfulness: 0.429 · Citations supported: [4, 5]
- ⚠️ **Severe hallucination:** Very low faithfulness (0%) AND missing required concepts.

### ✅ `bio_002` — Why does the Red Sea support such distinctive marine biodiversity?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 11940ms
- Concept coverage: 1.0 · Faithfulness: 0.667 · Citations supported: [1, 2, 3, 5]

### ❌ `bio_003` — Are Red Sea reef communities uniform along the entire Egyptian coast, or do they vary?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 10188ms
- Concept coverage: 0.0 · Faithfulness: 0.833 · Citations supported: [1, 5]
- ⚠️ **Severe hallucination:** Very low faithfulness (0%) AND missing required concepts.

### ✅ `bio_004` — What types of coral reefs occur in the Red Sea?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 10147ms
- Concept coverage: 1.0 · Faithfulness: 0.667 · Citations supported: [2, 3]

### ✅ `cons_001` — What are the main threats to Egyptian Red Sea coral reefs?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 12254ms
- Concept coverage: 1.0 · Faithfulness: 0.7 · Citations supported: [1, 2]

### ✅ `cons_002` — How does tourism affect Red Sea coral reef ecosystems?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 11778ms
- Concept coverage: 1.0 · Faithfulness: 0.333 · Citations supported: [5]

### ❌ `cons_003` — What conservation strategies are relevant for Red Sea reefs?
- Group: `conservation` · Expected: `answer` · Refused: `True` · Latency: 9910ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: []

### ✅ `cons_004` — How has climate change affected the Red Sea ecosystem so far?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 14445ms
- Concept coverage: 1.0 · Faithfulness: 0.8 · Citations supported: [1, 3, 5]

### ✅ `synth_001` — How do the Red Sea's geology and oceanography together shape its marine life?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 13503ms
- Concept coverage: 0.5 · Faithfulness: 0.091 · Citations supported: [2, 3, 4, 5]

### ✅ `synth_002` — Why might the Red Sea matter for understanding climate-resilient coral reefs worldwide?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 14284ms
- Concept coverage: 1.0 · Faithfulness: 0.667 · Citations supported: [1, 3]

### ✅ `synth_003` — How do high salinity, high temperature, and geographic isolation interact to shape Red Sea ecology?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 12823ms
- Concept coverage: 1.0 · Faithfulness: 0.2 · Citations supported: [1, 2, 3]

### ✅ `off_001` — What are the best hotels to stay at in Hurghada?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 10270ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_002` — Where can I book the cheapest diving trip in Sharm El-Sheikh?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 7962ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_003` — Who won the 2022 FIFA World Cup final?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 7021ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ❌ `off_004` — Write me a Python function that sorts a list of numbers.
- Group: `off_topic` · Expected: `refuse` · Refused: `False` · Latency: 10189ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []
- ⚠️ **Severe hallucination:** Answered a off_topic question that should have been refused.

### ✅ `unsup_001` — What will the Red Sea surface temperature be in the year 2100?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 6139ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_002` — What are today's tide times for the port of Safaga?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 8892ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_003` — Which species of shark attacked tourists in the Red Sea last month?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 7908ms
- Concept coverage: 1.0 · Faithfulness: 1.0 · Citations supported: []

### ✅ `unsup_004` — Using only the RedSea GPT corpus, describe the coral reefs of the Mediterranean Sea.
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 10458ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_001` — What is 'Calcium Carbonate Resin' and what role does it play in Red Sea coral skeletons?
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 8639ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_002` — List five fish species from the corpus that are not actually mentioned anywhere in the sources.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 8726ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_003` — Explain the function of the 'Red Sea thermohaline amplifier gene' in coral heat tolerance.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 9183ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_004` — Describe the 'Hurghada Deep' hydrothermal brine pool reportedly discovered in 2019.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 9471ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `cit_001` — What is the average salinity of the Red Sea, and which source reports it?
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 8859ms
- Concept coverage: 0.5 · Faithfulness: 0.5 · Citations supported: [1, 3]

### ✅ `cit_002` — How was the Red Sea formed? Cite the geological source(s) you used.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 11531ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [2, 3, 4]

### ✅ `cit_003` — Explain one documented mechanism behind Red Sea coral heat tolerance, with a citation.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 10813ms
- Concept coverage: 1.0 · Faithfulness: 0.333 · Citations supported: [2, 5]
