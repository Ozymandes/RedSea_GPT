# Evaluation Report — baseline

- **Generated:** 2026-07-04T20:15:27Z
- **Provider:** `optillm` (anthropic_messages)
- **Model:** `gpt-4o-mini`
- **Base URL:** `https://optollm.optomatica.com/v1`

## Headline metrics

- Total questions: **38**
- Pass rate: **89.5%** (34/38)
- Severe hallucinations: **2**
- Avg concept coverage (answerable): **84.6%**
- Avg faithfulness (answerable): **63.2%**
- Citation support (answerable): **92.3%**
- Refusal accuracy: **91.7%** (11/12)
- Latency: mean **13754ms** · p95 **17244ms** · max **40383ms**

## By group

| Group | Pass | Total |
|---|---:|---:|
| biodiversity | 3 | 4 |
| citation_integrity | 2 | 3 |
| conservation | 4 | 4 |
| coral_heat | 4 | 4 |
| geology | 4 | 4 |
| hallucination_trap | 4 | 4 |
| oceanography | 4 | 4 |
| off_topic | 3 | 4 |
| synthesis | 2 | 3 |
| unsupported | 4 | 4 |

## Per-question detail

### ✅ `geo_001` — How did the Red Sea form geologically?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12026ms
- Concept coverage: 1.0 · Faithfulness: 0.667 · Citations supported: [2, 4, 5]

### ✅ `geo_002` — What makes the Red Sea geologically different from a normal mature ocean basin?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 12942ms
- Concept coverage: 0.5 · Faithfulness: 0.857 · Citations supported: [1, 2, 3, 5]

### ✅ `geo_003` — How is the Red Sea connected to seafloor spreading and rifting?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 11472ms
- Concept coverage: 1.0 · Faithfulness: 0.556 · Citations supported: [1, 2, 3, 4, 5]

### ✅ `geo_004` — What is the geological relationship between the Red Sea and the Gulf of Aden?
- Group: `geology` · Expected: `answer` · Refused: `False` · Latency: 13246ms
- Concept coverage: 1.0 · Faithfulness: 0.625 · Citations supported: [3, 4]

### ✅ `ocean_001` — Why is the Red Sea so much saltier than other seas?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 12663ms
- Concept coverage: 0.5 · Faithfulness: 0.778 · Citations supported: [1, 4, 5]

### ✅ `ocean_002` — What is the typical salinity of the Red Sea?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 10458ms
- Concept coverage: 1.0 · Faithfulness: 0.8 · Citations supported: [1, 3, 4]

### ✅ `ocean_003` — How does the Red Sea's water circulation work, and how does evaporation drive it?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 13762ms
- Concept coverage: 1.0 · Faithfulness: 0.9 · Citations supported: [1, 2, 4, 5]

### ✅ `ocean_004` — What are the main water-exchange processes between the Red Sea and the open ocean?
- Group: `oceanography` · Expected: `answer` · Refused: `False` · Latency: 13558ms
- Concept coverage: 1.0 · Faithfulness: 0.7 · Citations supported: [4, 5]

### ✅ `coral_001` — Why are some Red Sea corals unusually tolerant of high temperatures?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 13567ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [1, 2, 5]

### ✅ `coral_002` — How do Red Sea corals respond to thermal stress, and what cellular mechanisms are involved?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 12054ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [1, 3, 4]

### ✅ `coral_003` — What role do symbiotic algae play in coral health and bleaching?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 13116ms
- Concept coverage: 0.5 · Faithfulness: 0.667 · Citations supported: [3, 4, 5]

### ✅ `coral_004` — What did the 2023 bleaching event along the Egyptian Red Sea coast reveal about coral vulnerability?
- Group: `coral_heat` · Expected: `answer` · Refused: `False` · Latency: 12417ms
- Concept coverage: 1.0 · Faithfulness: 0.667 · Citations supported: [1]

### ✅ `bio_001` — What does endemism mean in the Red Sea context, and why is it notable there?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 14605ms
- Concept coverage: 1.0 · Faithfulness: 0.5 · Citations supported: [4, 5]

### ✅ `bio_002` — Why does the Red Sea support such distinctive marine biodiversity?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 11829ms
- Concept coverage: 1.0 · Faithfulness: 0.889 · Citations supported: [1, 4]

### ❌ `bio_003` — Are Red Sea reef communities uniform along the entire Egyptian coast, or do they vary?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 14293ms
- Concept coverage: 0.0 · Faithfulness: 1.0 · Citations supported: [1, 3, 5]
- ⚠️ **Severe hallucination:** Very low faithfulness (0%) AND missing required concepts.

### ✅ `bio_004` — What types of coral reefs occur in the Red Sea?
- Group: `biodiversity` · Expected: `answer` · Refused: `False` · Latency: 10247ms
- Concept coverage: 1.0 · Faithfulness: 0.833 · Citations supported: [2, 3]

### ✅ `cons_001` — What are the main threats to Egyptian Red Sea coral reefs?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 12627ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [2, 3]

### ✅ `cons_002` — How does tourism affect Red Sea coral reef ecosystems?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 12869ms
- Concept coverage: 1.0 · Faithfulness: 0.375 · Citations supported: [1, 5]

### ✅ `cons_003` — What conservation strategies are relevant for Red Sea reefs?
- Group: `conservation` · Expected: `answer` · Refused: `True` · Latency: 9885ms
- Concept coverage: 1.0 · Faithfulness: 0.25 · Citations supported: [1, 2]

### ✅ `cons_004` — How has climate change affected the Red Sea ecosystem so far?
- Group: `conservation` · Expected: `answer` · Refused: `False` · Latency: 13395ms
- Concept coverage: 1.0 · Faithfulness: 0.636 · Citations supported: [1, 3, 5]

### ❌ `synth_001` — How do the Red Sea's geology and oceanography together shape its marine life?
- Group: `synthesis` · Expected: `answer` · Refused: `True` · Latency: 14743ms
- Concept coverage: 0.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `synth_002` — Why might the Red Sea matter for understanding climate-resilient coral reefs worldwide?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 40383ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [1, 3, 4]

### ✅ `synth_003` — How do high salinity, high temperature, and geographic isolation interact to shape Red Sea ecology?
- Group: `synthesis` · Expected: `answer` · Refused: `False` · Latency: 20152ms
- Concept coverage: 1.0 · Faithfulness: 0.375 · Citations supported: [2, 3, 5]

### ✅ `off_001` — What are the best hotels to stay at in Hurghada?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 11892ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_002` — Where can I book the cheapest diving trip in Sharm El-Sheikh?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 17105ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `off_003` — Who won the 2022 FIFA World Cup final?
- Group: `off_topic` · Expected: `refuse` · Refused: `True` · Latency: 11799ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ❌ `off_004` — Write me a Python function that sorts a list of numbers.
- Group: `off_topic` · Expected: `refuse` · Refused: `False` · Latency: 14412ms
- Concept coverage: 1.0 · Faithfulness: 1.0 · Citations supported: []
- ⚠️ **Severe hallucination:** Answered a off_topic question that should have been refused.

### ✅ `unsup_001` — What will the Red Sea surface temperature be in the year 2100?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 10346ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_002` — What are today's tide times for the port of Safaga?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 12577ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_003` — Which species of shark attacked tourists in the Red Sea last month?
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 9855ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `unsup_004` — Using only the RedSea GPT corpus, describe the coral reefs of the Mediterranean Sea.
- Group: `unsupported` · Expected: `refuse` · Refused: `True` · Latency: 12961ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_001` — What is 'Calcium Carbonate Resin' and what role does it play in Red Sea coral skeletons?
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 11943ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_002` — List five fish species from the corpus that are not actually mentioned anywhere in the sources.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 15678ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_003` — Explain the function of the 'Red Sea thermohaline amplifier gene' in coral heat tolerance.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 17244ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `trap_004` — Describe the 'Hurghada Deep' hydrothermal brine pool reportedly discovered in 2019.
- Group: `hallucination_trap` · Expected: `refuse` · Refused: `True` · Latency: 11167ms
- Concept coverage: 1.0 · Faithfulness: 0.0 · Citations supported: []

### ✅ `cit_001` — What is the average salinity of the Red Sea, and which source reports it?
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 11734ms
- Concept coverage: 0.5 · Faithfulness: 0.667 · Citations supported: [3, 5]

### ✅ `cit_002` — How was the Red Sea formed? Cite the geological source(s) you used.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 14009ms
- Concept coverage: 1.0 · Faithfulness: 0.75 · Citations supported: [1, 2, 4, 5]

### ❌ `cit_003` — Explain one documented mechanism behind Red Sea coral heat tolerance, with a citation.
- Group: `citation_integrity` · Expected: `answer` · Refused: `False` · Latency: 13602ms
- Concept coverage: 1.0 · Faithfulness: 0.2 · Citations supported: []
