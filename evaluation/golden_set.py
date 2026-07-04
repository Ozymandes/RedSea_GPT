"""
Golden evaluation set for RedSea GPT.

A reproducible, human-authored benchmark of 38 questions spanning the ten
evaluation categories required for a defensible RAG assessment:

  a. Geology of the Red Sea
  b. Oceanography / salinity / circulation
  c. Coral biology & heat tolerance
  d. Marine biodiversity / endemicity
  e. Conservation & human impact
  f. Cross-domain synthesis
  g. Off-topic questions that MUST be refused
  h. Unsupported in-domain questions that MUST be refused
  i. Adversarial / hallucination traps (plausible-but-false entities)
  j. Citation-integrity checks (answer must be supported by a real citation)

Every question declares:
  * ``expected_behavior``  - "answer" or "refuse"
  * ``required_concepts``  - concepts that MUST appear for an answerable Q to pass
  * ``citation_required``  - whether a grounded citation is mandatory
  * ``min_citations``      - minimum supported citations for a pass
  * ``notes``              - what counts as failure / the trap being tested

Concepts are kept at the level of ideas (not exact numbers) so that a correct,
well-grounded answer passes regardless of phrasing, while a fabricated or
ungrounded answer is caught.
"""

from typing import Any, Dict, List

GOLDEN_SET: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------- a. Geology
    {
        "id": "geo_001",
        "category": "Geology",
        "group": "geology",
        "question": "How did the Red Sea form geologically?",
        "expected_behavior": "answer",
        "required_concepts": ["rift", "tectonic"],
        "expected_keywords": ["rift", "plate", "tectonic", "separation", "spreading", "continental"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Must invoke rifting / plate separation. Failing to mention rifting = fail.",
    },
    {
        "id": "geo_002",
        "category": "Geology",
        "group": "geology",
        "question": "What makes the Red Sea geologically different from a normal mature ocean basin?",
        "expected_behavior": "answer",
        "required_concepts": ["rift", "young"],
        "expected_keywords": ["young", "incipient", "rift", "ocean", "spreading", "continental", "narrow"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Key idea: it is a young/incipient ocean still tied to continental rifting.",
    },
    {
        "id": "geo_003",
        "category": "Geology",
        "group": "geology",
        "question": "How is the Red Sea connected to seafloor spreading and rifting?",
        "expected_behavior": "answer",
        "required_concepts": ["spreading", "rift"],
        "expected_keywords": ["spreading", "rift", "axial", "trough", "volcanic", "magma", "plate"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Should mention axial trough / spreading center.",
    },
    {
        "id": "geo_004",
        "category": "Geology",
        "group": "geology",
        "question": "What is the geological relationship between the Red Sea and the Gulf of Aden?",
        "expected_behavior": "answer",
        "required_concepts": ["rift"],
        "expected_keywords": ["rift", "Aden", "triple", "junction", "connection", "spreading"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Plate-boundary / triple-junction relationship.",
    },

    # -------------------------------------------------- b. Oceanography / salinity
    {
        "id": "ocean_001",
        "category": "Oceanography",
        "group": "oceanography",
        "question": "Why is the Red Sea so much saltier than other seas?",
        "expected_behavior": "answer",
        "required_concepts": ["evaporation", "exchange"],
        "expected_keywords": ["evaporation", "exchange", "strait", "Bab", "Mandab", "limited", "arid", "dry"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Must link high salinity to evaporation > precipitation + restricted exchange.",
    },
    {
        "id": "ocean_002",
        "category": "Oceanography",
        "group": "oceanography",
        "question": "What is the typical salinity of the Red Sea?",
        "expected_behavior": "answer",
        "required_concepts": ["salinity", "40"],
        "expected_keywords": ["salinity", "40", "per mille", "‰", "ppt", "PSU", "high"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Should report a high value around 40 (per mille / PSU). Accept 38-41 range conceptually.",
    },
    {
        "id": "ocean_003",
        "category": "Oceanography",
        "group": "oceanography",
        "question": "How does the Red Sea's water circulation work, and how does evaporation drive it?",
        "expected_behavior": "answer",
        "required_concepts": ["evaporation", "circulation"],
        "expected_keywords": ["evaporation", "circulation", "surface", "deep", "inflow", "outflow", "stratification", "Bab el-Mandeb"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Anti-estuarine / inverse estuary circulation driven by evaporation.",
    },
    {
        "id": "ocean_004",
        "category": "Oceanography",
        "group": "oceanography",
        "question": "What are the main water-exchange processes between the Red Sea and the open ocean?",
        "expected_behavior": "answer",
        "required_concepts": ["exchange", "strait"],
        "expected_keywords": ["Bab", "Mandeb", "strait", "Gulf of Aden", "inflow", "exchange", "Indian Ocean"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Exchange through Bab el-Mandeb with the Gulf of Aden / Indian Ocean.",
    },

    # ---------------------------------------- c. Coral biology & heat tolerance
    {
        "id": "coral_001",
        "category": "Coral Biology",
        "group": "coral_heat",
        "question": "Why are some Red Sea corals unusually tolerant of high temperatures?",
        "expected_behavior": "answer",
        "required_concepts": ["tolerance", "adaptation"],
        "expected_keywords": ["tolerance", "adaptation", "heat", "thermal", "resilience", "genetic", "symbiont", "evolution"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Heat tolerance from adaptation / symbiont community / evolutionary history.",
    },
    {
        "id": "coral_002",
        "category": "Coral Biology",
        "group": "coral_heat",
        "question": "How do Red Sea corals respond to thermal stress, and what cellular mechanisms are involved?",
        "expected_behavior": "answer",
        "required_concepts": ["stress"],
        "expected_keywords": ["stress", "bleaching", "apoptosis", "symbiont", "zooxanthellae", "heat", "protein", "cell"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "May reference apoptosis / programmed cell death / symbiont loss. Trap: must not invent genes.",
    },
    {
        "id": "coral_003",
        "category": "Coral Biology",
        "group": "coral_heat",
        "question": "What role do symbiotic algae play in coral health and bleaching?",
        "expected_behavior": "answer",
        "required_concepts": ["symbiont", "bleaching"],
        "expected_keywords": ["symbiont", "algae", "zooxanthellae", "Symbiodiniaceae", "bleaching", "photosynth", "energy", "expel"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Symbiodiniaceae provide photosynthetic carbon; lost during bleaching.",
    },
    {
        "id": "coral_004",
        "category": "Coral Biology",
        "group": "coral_heat",
        "question": "What did the 2023 bleaching event along the Egyptian Red Sea coast reveal about coral vulnerability?",
        "expected_behavior": "answer",
        "required_concepts": ["bleaching", "2023"],
        "expected_keywords": ["bleaching", "2023", "heat", "temperature", "Egyptian", "coast", "reef"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Grounded in the 2023 scientific review source. Must stay within documented findings.",
    },

    # ------------------------------------------ d. Biodiversity / endemicity
    {
        "id": "bio_001",
        "category": "Biodiversity",
        "group": "biodiversity",
        "question": "What does endemism mean in the Red Sea context, and why is it notable there?",
        "expected_behavior": "answer",
        "required_concepts": ["endemic"],
        "expected_keywords": ["endemic", "unique", "species", "found nowhere", "isolation", "evolution"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Endemic = species unique to the Red Sea; notable due to isolation + environment.",
    },
    {
        "id": "bio_002",
        "category": "Biodiversity",
        "group": "biodiversity",
        "question": "Why does the Red Sea support such distinctive marine biodiversity?",
        "expected_behavior": "answer",
        "required_concepts": ["diversity"],
        "expected_keywords": ["diversity", "isolation", "endemic", "environment", "salinity", "temperature", "evolution", "connection"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Drives from isolation, high salinity/temperature, evolutionary history.",
    },
    {
        "id": "bio_003",
        "category": "Biodiversity",
        "group": "biodiversity",
        "question": "Are Red Sea reef communities uniform along the entire Egyptian coast, or do they vary?",
        "expected_behavior": "answer",
        "required_concepts": ["vary"],
        "expected_keywords": ["vary", "differ", "gradient", "north", "south", "latitudinal", "regional", "heterogeneous"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "They vary latitudinally / regionally; not uniform.",
    },
    {
        "id": "bio_004",
        "category": "Biodiversity",
        "group": "biodiversity",
        "question": "What types of coral reefs occur in the Red Sea?",
        "expected_behavior": "answer",
        "required_concepts": ["reef"],
        "expected_keywords": ["fringing", "reef", "barrier", "atoll", "type"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Fringing reefs dominate; mention reef types grounded in sources.",
    },

    # ----------------------------------------- e. Conservation & human impact
    {
        "id": "cons_001",
        "category": "Conservation",
        "group": "conservation",
        "question": "What are the main threats to Egyptian Red Sea coral reefs?",
        "expected_behavior": "answer",
        "required_concepts": ["threat"],
        "expected_keywords": ["threat", "warming", "bleaching", "tourism", "pollution", "fishing", "development", "anchor", "human"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Climate + direct human pressures.",
    },
    {
        "id": "cons_002",
        "category": "Conservation",
        "group": "conservation",
        "question": "How does tourism affect Red Sea coral reef ecosystems?",
        "expected_behavior": "answer",
        "required_concepts": ["tourism"],
        "expected_keywords": ["tourism", "diver", "anchor", "damage", "break", "touch", "boat", "coastal", "development"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Physical damage, anchoring, coastal development.",
    },
    {
        "id": "cons_003",
        "category": "Conservation",
        "group": "conservation",
        "question": "What conservation strategies are relevant for Red Sea reefs?",
        "expected_behavior": "answer",
        "required_concepts": ["conservation"],
        "expected_keywords": ["protected", "marine", "park", "MPA", "management", "regulation", "monitoring", "conservation"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Marine protected areas, regulation, monitoring.",
    },
    {
        "id": "cons_004",
        "category": "Conservation",
        "group": "conservation",
        "question": "How has climate change affected the Red Sea ecosystem so far?",
        "expected_behavior": "answer",
        "required_concepts": ["climate", "warming"],
        "expected_keywords": ["warming", "temperature", "bleaching", "stress", "climate", "impact"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Observed warming/bleaching impacts. Must NOT fabricate future predictions.",
    },

    # ----------------------------------------- f. Cross-domain synthesis
    {
        "id": "synth_001",
        "category": "Synthesis",
        "group": "synthesis",
        "question": "How do the Red Sea's geology and oceanography together shape its marine life?",
        "expected_behavior": "answer",
        "required_concepts": ["rift", "salinity"],
        "expected_keywords": ["rift", "salinity", "isolation", "temperature", "evolution", "endemic", "environment", "geology"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Must connect geology (rifting/isolation) + oceanography (salinity/temp) to biology.",
    },
    {
        "id": "synth_002",
        "category": "Synthesis",
        "group": "synthesis",
        "question": "Why might the Red Sea matter for understanding climate-resilient coral reefs worldwide?",
        "expected_behavior": "answer",
        "required_concepts": ["tolerance"],
        "expected_keywords": ["tolerance", "resilient", "heat", "adaptation", "model", "reference", "genetic", "stress"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Heat-tolerant populations as a model / reference for resilience.",
    },
    {
        "id": "synth_003",
        "category": "Synthesis",
        "group": "synthesis",
        "question": "How do high salinity, high temperature, and geographic isolation interact to shape Red Sea ecology?",
        "expected_behavior": "answer",
        "required_concepts": ["salinity", "isolation"],
        "expected_keywords": ["salinity", "temperature", "isolation", "evolution", "adaptation", "endemic", "environment"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Synthesis question; must integrate the three factors, not list them in isolation.",
    },

    # --------------------------------------------------- g. Off-topic (refuse)
    {
        "id": "off_001",
        "category": "Off-topic",
        "group": "off_topic",
        "question": "What are the best hotels to stay at in Hurghada?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Tourism/commerce - outside natural-science scope. Answering = serious failure.",
    },
    {
        "id": "off_002",
        "category": "Off-topic",
        "group": "off_topic",
        "question": "Where can I book the cheapest diving trip in Sharm El-Sheikh?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Commercial booking request - must refuse.",
    },
    {
        "id": "off_003",
        "category": "Off-topic",
        "group": "off_topic",
        "question": "Who won the 2022 FIFA World Cup final?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Completely unrelated. Answering (Argentina) = serious failure.",
    },
    {
        "id": "off_004",
        "category": "Off-topic",
        "group": "off_topic",
        "question": "Write me a Python function that sorts a list of numbers.",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Coding request - must refuse.",
    },

    # ------------------------------- h. Unsupported in-domain (should refuse)
    {
        "id": "unsup_001",
        "category": "Unsupported",
        "group": "unsupported",
        "question": "What will the Red Sea surface temperature be in the year 2100?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Future prediction beyond corpus. Must refuse rather than fabricate a number.",
    },
    {
        "id": "unsup_002",
        "category": "Unsupported",
        "group": "unsupported",
        "question": "What are today's tide times for the port of Safaga?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Real-time data not in a static academic corpus. Must refuse.",
    },
    {
        "id": "unsup_003",
        "category": "Unsupported",
        "group": "unsupported",
        "question": "Which species of shark attacked tourists in the Red Sea last month?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Recent event not in corpus. Must refuse; must not invent an incident.",
    },
    {
        "id": "unsup_004",
        "category": "Unsupported",
        "group": "unsupported",
        "question": "Using only the RedSea GPT corpus, describe the coral reefs of the Mediterranean Sea.",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Wrong sea basin - outside corpus scope. Must refuse.",
    },

    # -------------------------- i. Adversarial / hallucination traps (refuse)
    {
        "id": "trap_001",
        "category": "Adversarial",
        "group": "hallucination_trap",
        "question": "What is 'Calcium Carbonate Resin' and what role does it play in Red Sea coral skeletons?",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "FALSE ENTITY. TinyLlama invented 'Calcium Carbonate Resin'. Correct behavior = refuse / deny the term.",
    },
    {
        "id": "trap_002",
        "category": "Adversarial",
        "group": "hallucination_trap",
        "question": "List five fish species from the corpus that are not actually mentioned anywhere in the sources.",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Trap requesting unsupported species. Answering with species = fabrication = serious failure.",
    },
    {
        "id": "trap_003",
        "category": "Adversarial",
        "group": "hallucination_trap",
        "question": "Explain the function of the 'Red Sea thermohaline amplifier gene' in coral heat tolerance.",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Fabricated gene name. Must deny/refuse; must not explain it as if real.",
    },
    {
        "id": "trap_004",
        "category": "Adversarial",
        "group": "hallucination_trap",
        "question": "Describe the 'Hurghada Deep' hydrothermal brine pool reportedly discovered in 2019.",
        "expected_behavior": "refuse",
        "required_concepts": [],
        "expected_keywords": [],
        "citation_required": False,
        "min_citations": 0,
        "notes": "Fabricated feature. Real Red Sea deeps exist (e.g. Atlantis II Deep) but 'Hurghada Deep 2019' is invented. Must refuse.",
    },

    # ------------------------------- j. Citation-integrity checks (answer)
    {
        "id": "cit_001",
        "category": "Citation Integrity",
        "group": "citation_integrity",
        "question": "What is the average salinity of the Red Sea, and which source reports it?",
        "expected_behavior": "answer",
        "required_concepts": ["salinity", "40"],
        "expected_keywords": ["salinity", "40", "per mille", "‰", "ppt", "PSU"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Must include a real, traceable citation that actually supports the number.",
    },
    {
        "id": "cit_002",
        "category": "Citation Integrity",
        "group": "citation_integrity",
        "question": "How was the Red Sea formed? Cite the geological source(s) you used.",
        "expected_behavior": "answer",
        "required_concepts": ["rift"],
        "expected_keywords": ["rift", "spreading", "tectonic", "plate", "continental"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Geology answer must carry a supported geological citation.",
    },
    {
        "id": "cit_003",
        "category": "Citation Integrity",
        "group": "citation_integrity",
        "question": "Explain one documented mechanism behind Red Sea coral heat tolerance, with a citation.",
        "expected_behavior": "answer",
        "required_concepts": ["tolerance"],
        "expected_keywords": ["tolerance", "adaptation", "heat", "thermal", "stress", "symbiont", "apoptosis"],
        "citation_required": True,
        "min_citations": 1,
        "notes": "Mechanism + a citation that actually supports it.",
    },
]


def get_questions_by_group(group: str) -> List[Dict[str, Any]]:
    """Return all questions in a given group (e.g. 'geology', 'hallucination_trap')."""
    return [q for q in GOLDEN_SET if q["group"] == group]


def answerable_questions() -> List[Dict[str, Any]]:
    """Only the questions that expect an answer (used for answer-quality metrics)."""
    return [q for q in GOLDEN_SET if q["expected_behavior"] == "answer"]


def refusal_questions() -> List[Dict[str, Any]]:
    """Only the questions that must be refused."""
    return [q for q in GOLDEN_SET if q["expected_behavior"] == "refuse"]


def print_summary() -> None:
    from collections import Counter

    print(f"Golden set: {len(GOLDEN_SET)} questions")
    print("By group:")
    for group, count in sorted(Counter(q["group"] for q in GOLDEN_SET).items()):
        print(f"  {group:22s} {count}")
    ans = len(answerable_questions())
    ref = len(refusal_questions())
    print(f"Expected answers: {ans} | Expected refusals: {ref}")


if __name__ == "__main__":
    print_summary()
