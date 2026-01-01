"""
Test Questions for RedSea GPT Evaluation

Defines a comprehensive set of test questions covering different
aspects of Red Sea knowledge and query types.
"""

from typing import List, Dict, Any


# Test questions covering different categories and difficulty levels
TEST_QUESTIONS: List[Dict[str, Any]] = [
    # Category: Oceanography
    {
        "id": "ocean_001",
        "category": "Oceanography",
        "type": "factual",
        "difficulty": "easy",
        "question": "What is the average salinity of the Red Sea?",
        "expected_keywords": ["salinity", "40", "high", "ppt", " PSU"],
    },
    {
        "id": "ocean_002",
        "category": "Oceanography",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "Why is the Red Sea more saline than other seas?",
        "expected_keywords": ["evaporation", "temperature", "circulation", "limited", "exchange"],
    },
    {
        "id": "ocean_003",
        "category": "Oceanography",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "How does water circulation work in the Red Sea?",
        "expected_keywords": ["stratification", "thermocline", "winter", "summer", "circulation"],
    },
    {
        "id": "ocean_004",
        "category": "Oceanography",
        "type": "factual",
        "difficulty": "easy",
        "question": "What are the temperature ranges in the Red Sea?",
        "expected_keywords": ["temperature", "warm", "tropical", "range"],
    },

    # Category: Coral Reefs
    {
        "id": "coral_001",
        "category": "Coral Reefs",
        "type": "factual",
        "difficulty": "easy",
        "question": "What types of coral reefs are found in the Red Sea?",
        "expected_keywords": ["fringing", "reefs", "barrier", "atoll", "types"],
    },
    {
        "id": "coral_002",
        "category": "Coral Reefs",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "Why are Red Sea corals more resistant to bleaching?",
        "expected_keywords": ["temperature", "tolerance", "adaptation", "thermal", "resilience"],
    },
    {
        "id": "coral_003",
        "category": "Coral Reefs",
        "type": "factual",
        "difficulty": "medium",
        "question": "What are the main coral species in the Red Sea?",
        "expected_keywords": ["Acropora", "Porites", "species", "coral"],
    },
    {
        "id": "coral_004",
        "category": "Coral Reefs",
        "type": "analytical",
        "difficulty": "hard",
        "question": "How do coral communities differ between the northern and southern Red Sea?",
        "expected_keywords": ["northern", "southern", "difference", "species", "diversity", "gradient"],
    },

    # Category: Marine Life
    {
        "id": "marine_001",
        "category": "Marine Life",
        "type": "factual",
        "difficulty": "easy",
        "question": "What is the endemism rate in the Red Sea?",
        "expected_keywords": ["endemic", "species", "percentage", "unique"],
    },
    {
        "id": "marine_002",
        "category": "Marine Life",
        "type": "factual",
        "difficulty": "medium",
        "question": "What are some endemic species found in the Red Sea?",
        "expected_keywords": ["fish", "coral", "endemic", "species", "example"],
    },
    {
        "id": "marine_003",
        "category": "Marine Life",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "Why does the Red Sea have high biodiversity?",
        "expected_keywords": ["endemic", "connection", "Indian Ocean", "conditions", "diversity"],
    },
    {
        "id": "marine_004",
        "category": "Marine Life",
        "type": "factual",
        "difficulty": "easy",
        "question": "What large marine animals are found in the Red Sea?",
        "expected_keywords": ["dolphin", "shark", "turtle", "whale", "dugong"],
    },

    # Category: Geology
    {
        "id": "geo_001",
        "category": "Geology",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "How was the Red Sea formed?",
        "expected_keywords": ["rift", "separation", "tectonic", "plate", "formation"],
    },
    {
        "id": "geo_002",
        "category": "Geology",
        "type": "factual",
        "difficulty": "easy",
        "question": "What is the geological connection between the Red Sea and the Gulf of Aden?",
        "expected_keywords": ["rift", "connection", "spreading", "triple junction"],
    },
    {
        "id": "geo_003",
        "category": "Geology",
        "type": "explanatory",
        "difficulty": "hard",
        "question": "What are the main geological features of the Red Sea rift?",
        "expected_keywords": ["axial trough", "deeps", "sediments", "spreading center", "volcanic"],
    },

    # Category: Conservation
    {
        "id": "cons_001",
        "category": "Conservation",
        "type": "explanatory",
        "difficulty": "medium",
        "question": "What are the main threats to Red Sea coral reefs?",
        "expected_keywords": ["bleaching", "warming", "human", "pollution", "development", "fishing"],
    },
    {
        "id": "cons_002",
        "category": "Conservation",
        "type": "factual",
        "difficulty": "medium",
        "question": "What conservation efforts exist in the Egyptian Red Sea?",
        "expected_keywords": ["protected areas", "marine parks", "regulation", "conservation"],
    },
    {
        "id": "cons_003",
        "category": "Conservation",
        "type": "analytical",
        "difficulty": "hard",
        "question": "How has climate change affected the Red Sea ecosystem?",
        "expected_keywords": ["temperature", "bleaching", "warming", "impact", "stress"],
    },

    # Category: Regional Differences
    {
        "id": "regional_001",
        "category": "Regional Differences",
        "type": "comparative",
        "difficulty": "medium",
        "question": "How does the northern Red Sea differ from the southern Red Sea?",
        "expected_keywords": ["temperature", "salinity", "coral", "biodiversity", "different"],
    },
    {
        "id": "regional_002",
        "category": "Regional Differences",
        "type": "comparative",
        "difficulty": "hard",
        "question": "Compare coral reef health between the Egyptian and Saudi coasts",
        "expected_keywords": ["Egyptian", "Saudi", "coast", "health", "difference", "development"],
    },
]


def get_questions_by_category(category: str = None) -> List[Dict[str, Any]]:
    """
    Get test questions filtered by category.

    Args:
        category: Category name (e.g., 'Oceanography', 'Coral Reefs')
                 If None, returns all questions

    Returns:
        List of question dictionaries

    Examples:
        >>> ocean_q = get_questions_by_category("Oceanography")
        >>> all_q = get_questions_by_category()
    """
    if category is None:
        return TEST_QUESTIONS

    return [q for q in TEST_QUESTIONS if q["category"] == category]


def get_questions_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """
    Get test questions filtered by difficulty.

    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')

    Returns:
        List of question dictionaries
    """
    return [q for q in TEST_QUESTIONS if q["difficulty"] == difficulty]


def get_questions_by_type(question_type: str) -> List[Dict[str, Any]]:
    """
    Get test questions filtered by type.

    Args:
        question_type: Question type ('factual', 'explanatory', 'analytical', 'comparative')

    Returns:
        List of question dictionaries
    """
    return [q for q in TEST_QUESTIONS if q["type"] == question_type]


def print_question_summary() -> None:
    """Print a summary of the test question set."""
    from collections import Counter

    total = len(TEST_QUESTIONS)

    print(f"📊 Test Question Summary")
    print(f"=" * 50)
    print(f"Total Questions: {total}")
    print()

    # By category
    print("By Category:")
    categories = Counter(q["category"] for q in TEST_QUESTIONS)
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
    print()

    # By difficulty
    print("By Difficulty:")
    difficulties = Counter(q["difficulty"] for q in TEST_QUESTIONS)
    for diff, count in difficulties.items():
        print(f"  {diff}: {count}")
    print()

    # By type
    print("By Question Type:")
    types = Counter(q["type"] for q in TEST_QUESTIONS)
    for qtype, count in types.items():
        print(f"  {qtype}: {count}")


if __name__ == "__main__":
    print_question_summary()
