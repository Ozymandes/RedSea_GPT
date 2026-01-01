"""
Interactive CLI Interface for RedSea GPT

Provides a command-line interface for querying the RedSea GPT system.
"""

import sys
from typing import Optional
from generation.rag_chain import RedSeaGPT, create_rag_chain


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def print_welcome_message() -> None:
    """Print welcome message and instructions."""
    print_separator()
    print("🐠 RedSea GPT - Your Personal Naturalist for the Egyptian Red Sea 🐠")
    print_separator()
    print("\nAsk me anything about:")
    print("  • Coral reefs and marine life")
    print("  • Oceanography and water conditions")
    print("  • Geology and formation of the Red Sea")
    print("  • Conservation and environmental issues")
    print("  • Biodiversity and endemic species")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("  'sources' - Toggle source display")
    print("\n" + "-" * 80 + "\n")


def print_answer(answer: str, sources: Optional[list] = None, metadata: Optional[dict] = None) -> None:
    """
    Print the answer and optionally show sources with metadata.

    Args:
        answer: Generated answer
        sources: Optional list of source documents
        metadata: Optional metadata (confidence, refusal, hallucination check)
    """
    # Print metadata if available
    if metadata:
        if metadata.get('refusal'):
            print(f"\n⚠️  {metadata.get('confidence', 0):.2f} confidence - REFUSED")

    print("\n📝 Answer:")
    print_separator("-")
    print(answer)
    print_separator("-")

    # Show confidence and hallucination check
    if metadata and not metadata.get('refusal'):
        confidence = metadata.get('confidence', 0)
        print(f"\n📊 Confidence: {confidence:.2%}")

        hallucination = metadata.get('hallucination_check', {})
        if hallucination.get('has_hallucination'):
            print(f"⚠️  Grounding: {hallucination['grounding_rate']:.1%} ({hallucination['grounded_sentences']}/{hallucination['total_sentences']} sentences grounded)")

    if sources:
        print("\n📚 Sources:")
        for source in sources:
            cit_id = source.get('citation_id', '?')
            print(f"\n  [{cit_id}] {source['source']}, page {source['page']}")
            print(f"      {source['content']}")
    else:
        print()

    print()


def run_interactive_cli(
    vectordb_path: str = "chroma_redsea",
    retrieval_k: int = 5,
    show_sources: bool = True,
    use_mmr: bool = True,
    refusal_threshold: float = 0.2,
    structured_citations: bool = True,
) -> None:
    """
    Run the interactive CLI.

    Args:
        vectordb_path: Path to vector database
        retrieval_k: Number of documents to retrieve
        show_sources: Whether to show sources by default
        use_mmr: Use MMR for diverse retrieval
        refusal_threshold: Confidence threshold for answering (0-1)
        structured_citations: Use [1], [2] citation format
    """
    print("\n⏳ Initializing RedSea GPT...")
    print(f"   Model: Llama 70B (via Grok API)")
    print(f"   Retrieval: k={retrieval_k}, {'MMR' if use_mmr else 'similarity'}")
    print(f"   Citations: {'Structured [1], [2]' if structured_citations else 'Narrative'}")
    print(f"   Refusal threshold: {refusal_threshold}")
    print(f"   Vector DB: {vectordb_path}")

    try:
        gpt = RedSeaGPT(
            vectordb_path=vectordb_path,
            retrieval_k=retrieval_k,
            use_mmr=use_mmr,
            refusal_threshold=refusal_threshold,
            structured_citations=structured_citations,
        )
        print("✅ Ready!\n")
    except Exception as e:
        print(f"❌ Error initializing RedSea GPT: {e}")
        sys.exit(1)

    print_welcome_message()

    conversation_count = 0

    while True:
        try:
            # Get user input
            question = input("🤔 Your question: ").strip()

            # Handle empty input
            if not question:
                continue

            # Handle commands
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thanks for using RedSea GPT! Goodbye!")
                print_separator()
                break

            if question.lower() == "help":
                print_welcome_message()
                continue

            if question.lower() == "sources":
                show_sources = not show_sources
                print(f"\n{'✅' if show_sources else '❌'} Source display: {'enabled' if show_sources else 'disabled'}\n")
                continue

            # Process the question
            conversation_count += 1
            print(f"\n⏳ Thinking... (Question #{conversation_count})")

            result = gpt.query(question, return_source_docs=True)
            metadata = {
                'confidence': result.get('confidence'),
                'refusal': result.get('refusal', False),
                'hallucination_check': result.get('hallucination_check', {}),
                'retrieval_method': result.get('retrieval_method'),
            }

            sources = result["sources"] if show_sources else None
            print_answer(result["answer"], sources, metadata)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            print_separator()
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_single_query(
    question: str,
    vectordb_path: str = "chroma_redsea",
    retrieval_k: int = 5,
    show_sources: bool = True,
    use_mmr: bool = True,
    refusal_threshold: float = 0.2,
    structured_citations: bool = True,
) -> None:
    """
    Run a single query and print the result.

    Args:
        question: Question to ask
        vectordb_path: Path to vector database
        retrieval_k: Number of documents to retrieve
        show_sources: Whether to show sources
        use_mmr: Use MMR for retrieval
        refusal_threshold: Confidence threshold for refusal
        structured_citations: Use structured citations
    """
    print(f"\n⏳ Initializing RedSea GPT...")

    gpt = RedSeaGPT(
        vectordb_path=vectordb_path,
        retrieval_k=retrieval_k,
        use_mmr=use_mmr,
        refusal_threshold=refusal_threshold,
        structured_citations=structured_citations,
    )

    print(f"❓ Question: {question}\n")
    print_separator()

    result = gpt.query(question, return_source_docs=True)
    metadata = {
        'confidence': result.get('confidence'),
        'refusal': result.get('refusal', False),
        'hallucination_check': result.get('hallucination_check', {}),
    }

    if show_sources:
        print_answer(result["answer"], result["sources"], metadata)
    else:
        print_answer(result["answer"], None, metadata)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RedSea GPT - Interactive CLI"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode (ask one question and exit)",
    )
    parser.add_argument(
        "--retrieval-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--vectordb", "-v",
        type=str,
        default="chroma_redsea",
        help="Path to vector database (default: chroma_redsea)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents",
    )
    parser.add_argument(
        "--no-mmr",
        action="store_true",
        help="Disable MMR (use simple similarity search)",
    )
    parser.add_argument(
        "--refusal-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for answering (0-1, default: 0.2)",
    )
    parser.add_argument(
        "--no-structured-citations",
        action="store_true",
        help="Use narrative citations instead of [1], [2] format",
    )

    args = parser.parse_args()

    if args.query:
        run_single_query(
            question=args.query,
            vectordb_path=args.vectordb,
            retrieval_k=args.retrieval_k,
            show_sources=not args.no_sources,
            use_mmr=not args.no_mmr,
            refusal_threshold=args.refusal_threshold,
            structured_citations=not args.no_structured_citations,
        )
    else:
        run_interactive_cli(
            vectordb_path=args.vectordb,
            retrieval_k=args.retrieval_k,
            show_sources=not args.no_sources,
            use_mmr=not args.no_mmr,
            refusal_threshold=args.refusal_threshold,
            structured_citations=not args.no_structured_citations,
        )


if __name__ == "__main__":
    main()
