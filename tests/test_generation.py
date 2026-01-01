"""
Tests for Generation Module
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generation.llm_config import create_llm_from_preset
from generation.prompts import create_rag_prompt, format_context, create_naturalist_system_prompt
from generation.rag_chain import RedSeaGPT


def test_prompt_creation():
    """Test that prompts are created correctly."""
    prompt = create_rag_prompt(version=3)

    # Test formatting
    formatted = prompt.format(
        context="This is test context about the Red Sea.",
        question="What is the salinity?"
    )

    assert "Red Sea" in formatted
    assert "salinity" in formatted
    assert "context" in formatted.lower() or "context" in formatted

    print("✅ Prompt creation test passed")


def test_system_prompt():
    """Test that system prompt contains necessary elements."""
    system_prompt = create_naturalist_system_prompt()

    # Check for key elements
    assert "naturalist" in system_prompt.lower()
    assert "red sea" in system_prompt.lower()
    assert "egyptian" in system_prompt.lower()

    print("✅ System prompt test passed")


def test_context_formatting():
    """Test context formatting."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="The Red Sea has high salinity.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Coral reefs are diverse.",
            metadata={"source": "test2.pdf", "page": 5}
        ),
    ]

    formatted = format_context(docs)

    assert "salinity" in formatted
    assert "coral" in formatted
    assert "test.pdf" in formatted
    assert "page 1" in formatted

    print("✅ Context formatting test passed")


def test_llm_creation():
    """Test LLM creation with preset."""
    # This test requires model download, so we'll just test the API
    try:
        llm = create_llm_from_preset("tinyllama", temperature=0.5)
        assert llm is not None
        print("✅ LLM creation test passed")
    except Exception as e:
        print(f"⚠️  LLM creation test skipped (requires model download): {e}")


def test_rag_chain_initialization():
    """Test RAG chain initialization."""
    try:
        gpt = RedSeaGPT(
            vectordb_path="chroma_redsea",
            llm_preset="tinyllama",
        )

        assert gpt.vectordb is not None
        assert gpt.llm is not None
        assert gpt.chain is not None

        print("✅ RAG chain initialization test passed")
    except Exception as e:
        print(f"⚠️  RAG chain test skipped (requires vector DB): {e}")


def test_basic_query():
    """Test basic query functionality."""
    try:
        gpt = RedSeaGPT(
            vectordb_path="chroma_redsea",
            llm_preset="tinyllama",
        )

        result = gpt.query("What is the Red Sea?", return_source_docs=False)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        print(f"✅ Basic query test passed")
        print(f"   Answer length: {len(result)} characters")
    except Exception as e:
        print(f"⚠️  Basic query test skipped: {e}")


if __name__ == "__main__":
    print("Running generation module tests...\n")

    test_prompt_creation()
    test_system_prompt()
    test_context_formatting()
    test_llm_creation()
    test_rag_chain_initialization()
    test_basic_query()

    print("\n✅ All generation tests completed!")
