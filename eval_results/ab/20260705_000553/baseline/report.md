# Evaluation Report — baseline

- **Generated:** 2026-07-04T21:05:53Z
- **Provider:** `optillm` (anthropic_messages)
- **Model:** `gpt-4o-mini`
- **Base URL:** `https://optollm.optomatica.com/v1`

## Headline metrics

- Total questions: **38**
- Pass rate: **92.1%** (35/38)
- Severe hallucinations: **2**
- Avg concept coverage (answerable): **88.5%**
- Avg faithfulness (answerable): **61.1%**
- Citation support (answerable): **96.2%**
- Refusal accuracy: **91.7%** (11/12)
- Latency: mean **12961ms** · p95 **15662ms** · max **71876ms**

## By group

| Group | Pass | Total |
|---|---:|---:|
| biodiversity | 3 | 4 |
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
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12392ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [2, 4]

### ✅ `geo_002` — What makes the Red Sea geologically different from a normal mature ocean basin?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 15124ms
- Concept coverage: 1.0 · Faithfulness: 0.6 · Citations supported: [1, 3]

### ✅ `geo_003` — How is the Red Sea connected to seafloor spreading and rifting?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12192ms
- Concept coverage: 1.0 · Faithfulness: 0.636 · Citations supported: [3, 4, 5]

### ✅ `geo_004` — What is the geological relationship between the Red Sea and the Gulf of Aden?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12382ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [3, 4]

### ✅ `ocean_001` — Why is the Red Sea so much saltier than other seas?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 10958ms
- Concept coverage: 0.5 · Faithfulness: 0.889 · Citations supported: [1, 3, 5]

### ✅ `ocean_002` — What is the typical salinity of the Red Sea?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 11829ms
- Concept coverage: 1.0 · Faithfulness: 1.0 · Citations supported: [1, 3, 4]

### ✅ `ocean_003` — How does the Red Sea's water circulation work, and how does evaporation drive it?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 10744ms
- Concept coverage: 1.0 · Faithfulness: 0.833 · Citations supported: [1, 2, 4, 5]

### ✅ `ocean_004` — What are the main water-exchange processes between the Red Sea and the open ocean?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 13353ms
- Concept coverage: 1.0 · Faithfulness: 0.6 · Citations supported: [4, 5]

### ✅ `coral_001` — Why are some Red Sea corals unusually tolerant of high temperatures?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 13475ms
- Concept coverage: 1.0 · Faithfulness: 0.7 · Citations supported: [1, 2, 4]

### ✅ `coral_002` — How do Red Sea corals respond to thermal stress, and what cellular mechanisms are involved?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 11771ms
- Concept coverage: 1.0 · Faithfulness: 0.1 · Citations supported: [1, 2, 3, 4, 5]

### ✅ `coral_003` — What role do symbiotic algae play in coral health and bleaching?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 12721ms
- Concept coverage: 0.5 · Faithfulness: 0.875 · Citations supported: [3, 4, 5]

### ✅ `coral_004` — What did the 2023 bleaching event along the Egyptian Red Sea coast reveal about coral vulnerability?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 71876ms
- Concept coverage: 1.0 · Faithfulness: 0.25 · Citations supported: [1, 2, 3, 4]

### ✅ `bio_001` — What does endemism mean in the Red Sea context, and why is it notable there?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 15662ms
- Concept coverage: 1.0 · Faithfulness: 0.625 · Citations supported: [4, 5]

### ✅ `bio_002` — Why does the Red Sea support such distinctive marine biodiversity?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 18747ms
- Concept coverage: 1.0 · Faithfulness: 0.636 · Citations supported: [1, 3, 4]

### ❌ `bio_003` — Are Red Sea reef communities uniform along the entire Egyptian coast, or do they vary?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 10789ms
- Concept coverage: 0.0 · Faithfulness: 0.667 · Citations supported: [1, 5]
- ⚠️ **Severe hallucination:** Very low faithfulness (17%) AND missing required concepts.

### ✅ `bio_004` — What types of coral reefs occur in the Red Sea?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 10049ms
- Concept coverage: 1.0 · Faithfulness: 0.714 · Citations supported: [2, 3]

### ✅ `cons_001` — What are the main threats to Egyptian Red Sea coral reefs?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 11048ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [1, 2]

### ✅ `cons_002` — How does tourism affect Red Sea coral reef ecosystems?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 11648ms
- Concept coverage: 1.0 · Faithfulness: 0.125 · Citations supported: [5]

### ❌ `cons_003` — What conservation strategies are relevant for Red Sea reefs?
- Group: `conservation` · Expected: `answer` · Refused: `True` · Latency: 9995ms
- Concept coverage: 1.0 · Faithfulness: 0.333 · Citations supported: []

### ✅ `cons_004` — How has climate change affected the Red Sea ecosystem so far?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 14230ms
- Concept coverage: 1.0 · Faithfulness: 0.778 · Citations supported: [1, 3, 5]

### ✅ `synth_001` — How do the Red Sea's geology and oceanography together shape its marine life?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 12443ms
- Concept coverage: 0.5 · Faithfulness: 0.25 · Citations supported: [1, 2, 3, 5]

### ✅ `synth_002` — Why might the Red Sea matter for understanding climate-resilient coral reefs worldwide?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 12752ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [1, 3, 4]

### ✅ `synth_003` — How do high salinity, high temperature, and geographic isolation interact to shape Red Sea ecology?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 15107ms
- Concept coverage: 1.0 · Faithfulness: 0.4 · Citations supported: [1, 2, 5]

### ✅ `off_001` — What are the best hotels to stay at in Hurghada?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 9194ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_002` — Where can I book the cheapest diving trip in Sharm El-Sheikh?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 8908ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_003` — Who won the 2022 FIFA World Cup final?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 8276ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ❌ `off_004` — Write me a Python function that sorts a list of numbers.
- Group: `off_topic` · Expected: `refuse` · Refused: `False` · Latency: 9636ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []
- ⚠️ **Severe hallucination:** Answered a off_topic question that should have been refused.

### ✅ `unsup_001` — What will the Red Sea surface temperature be in the year 2100?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 6216ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_002` — What are today's tide times for the port of Safaga?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 11089ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_003` — Which species of shark attacked tourists in the Red Sea last month?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 6650ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_004` — Using only the RedSea GPT corpus, describe the coral reefs of the Mediterranean Sea.
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 9172ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_001` — What is 'Calcium Carbonate Resin' and what role does it play in Red Sea coral skeletons?
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 8138ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_002` — List five fish species from the corpus that are not actually mentioned anywhere in the sources.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 9335ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_003` — Explain the function of the 'Red Sea thermohaline amplifier gene' in coral heat tolerance.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 10191ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_004` — Describe the 'Hurghada Deep' hydrothermal brine pool reportedly discovered in 2019.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 12596ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `cit_001` — What is the average salinity of the Red Sea, and which source reports it?
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 9149ms
- Concept coverage: 0.5 · Faithfulness: 1.0 · Citations supported: [3]

### ✅ `cit_002` — How was the Red Sea formed? Cite the geological source(s) you used.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 12252ms
- Concept coverage: 1.0 · Faithfulness: 0.7 · Citations supported: [2, 3, 4, 5]

### ✅ `cit_003` — Explain one documented mechanism behind Red Sea coral heat tolerance, with a citation.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 10420ms
- Concept coverage: 1.0 · Faithfulness: 0.429 · Citations supported: [2, 5]
