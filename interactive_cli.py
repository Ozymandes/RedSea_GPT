"""
Interactive CLI Interface for RedSea GPT

Provides a command-line interface for querying the RedSea GPT system.
"""

import sys
from typing import Optional
from generation.rag_chain import RedSeaGPT, create_rag_chain
from generation.llm_config import describe_active_provider
from generation.memory import ConversationMemory, Turn
from logging_wrapper import LoggedRedSeaGPT  # Logging wrapper


def _build_engine(agent: bool, **kwargs):
    """Build either the baseline RAG or the agentic CRAG engine.

    Both expose the same ``.query(question, return_source_docs=True)`` contract.
    """
    if agent:
        from generation.agent import RedSeaAgent
        return RedSeaAgent(**kwargs)
    return RedSeaGPT(**kwargs)


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def print_welcome_message() -> None:
    """Print welcome message and instructions."""
    print_separator()
    print(" RedSea GPT - Your Personal Naturalist for the Egyptian Red Sea 🐠")
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
            print(f"\n  {metadata.get('confidence', 0):.2f} confidence - REFUSED")

    print("\n Answer:")
    print_separator("-")
    print(answer)
    print_separator("-")

    # Show confidence and hallucination check
    if metadata and not metadata.get('refusal'):
        confidence = metadata.get('confidence', 0)
        print(f"\n Confidence: {confidence:.2%}")

        hallucination = metadata.get('hallucination_check', {})
        if hallucination.get('has_hallucination'):
            print(f"  Grounding: {hallucination['grounding_rate']:.1%} ({hallucination['grounded_sentences']}/{hallucination['total_sentences']} sentences grounded)")

    if sources:
        print("\n Sources:")
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
    agent: bool = False,
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
    provider_info = describe_active_provider()
    print("\n Initializing RedSea GPT...")
    if provider_info.get("configured"):
        print(f"   Provider: {provider_info['provider']} | Model: {provider_info['model']}")
        print(f"   Endpoint: {provider_info['base_url']}")
    else:
        print(f"   Provider: not configured ({provider_info.get('error', 'see .env.example')})")
    print(f"   Retrieval: k={retrieval_k}, {'MMR' if use_mmr else 'similarity'}")
    print(f"   Citations: {'Structured [1], [2]' if structured_citations else 'Narrative'}")
    print(f"   Refusal threshold: {refusal_threshold}")
    print(f"   Vector DB: {vectordb_path}")

    try:
        # Initialize the engine (baseline RAG or agentic CRAG)
        raw_gpt = _build_engine(
            agent=agent,
            vectordb_path=vectordb_path,
            retrieval_k=retrieval_k,
            **({"use_mmr": use_mmr, "refusal_threshold": refusal_threshold,
                "structured_citations": structured_citations} if not agent else {}),
        )

        mode = "agentic CRAG (LangGraph)" if agent else "baseline RAG"
        print(f"   Engine: {mode}")
        # Wrap with logging (works for both since they share the .query contract)
        gpt = LoggedRedSeaGPT(raw_gpt, enable_logging=True)
        print(" Ready! (Logging enabled - logs stored in ./logs/)\n")
    except Exception as e:
        print(f" Error initializing RedSea GPT: {e}")
        sys.exit(1)

    print_welcome_message()

    conversation_count = 0
    # Multiturn memory: lets the user ask follow-ups with pronouns / implicit
    # references ("how deep is it?", "and its salinity?"). The engine rewrites
    # the latest message into a self-contained question before retrieval.
    memory = ConversationMemory(max_turns=6)

    while True:
        try:
            # Get user input
            question = input(" Your question: ").strip()

            # Handle empty input
            if not question:
                continue

            # Handle commands
            if question.lower() in ["quit", "exit", "q"]:
                print("\n Thanks for using RedSea GPT! Goodbye!")
                print_separator()
                break

            if question.lower() == "help":
                print_welcome_message()
                continue

            if question.lower() == "sources":
                show_sources = not show_sources
                print(f"\n{'✅' if show_sources else '❌'} Source display: {'enabled' if show_sources else 'disabled'}\n")
                continue

            # Multiturn commands
            if question.lower() in ("/clear", "clear memory"):
                memory.clear()
                print("\n Conversation memory cleared.\n")
                continue
            if question.lower() in ("/history", "/mem"):
                if memory.is_empty:
                    print("\n (no conversation history yet)\n")
                else:
                    print(f"\n Conversation history ({memory.num_turns} turn{'s' if memory.num_turns != 1 else ''}):")
                    for i, t in enumerate(memory.turns, 1):
                        rq = f"  -> resolved: {t.resolved_question}" if t.resolved_question and t.resolved_question != t.question else ""
                        print(f"   {i}. You: {t.question[:70]}")
                        if rq:
                            print(rq)
                    print("")
                continue

            # Process the question
            conversation_count += 1
            print(f"\n Thinking... (Question #{conversation_count})")

            result = gpt.query(question, return_source_docs=True, memory=memory)
            # Record this turn so the next question can reference it.
            memory.add(Turn(
                question=question,
                answer=str(result.get("answer", "")),
                sources=result.get("sources", []) or [],
                resolved_question=result.get("resolved_question"),
            ))
            metadata = {
                'confidence': result.get('confidence'),
                'refusal': result.get('refusal', False),
                'hallucination_check': result.get('hallucination_check', {}),
                'retrieval_method': result.get('retrieval_method'),
            }

            sources = result["sources"] if show_sources else None
            print_answer(result["answer"], sources, metadata)

        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!")
            print_separator()
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


def run_single_query(
    question: str,
    vectordb_path: str = "chroma_redsea",
    retrieval_k: int = 5,
    show_sources: bool = True,
    use_mmr: bool = True,
    refusal_threshold: float = 0.2,
    structured_citations: bool = True,
    agent: bool = False,
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

    # Initialize the engine
    raw_gpt = _build_engine(
        agent=agent,
        vectordb_path=vectordb_path,
        retrieval_k=retrieval_k,
        **({"use_mmr": use_mmr, "refusal_threshold": refusal_threshold,
            "structured_citations": structured_citations} if not agent else {}),
    )

    # Wrap with logging
    gpt = LoggedRedSeaGPT(raw_gpt, enable_logging=True)

    print(f" Question: {question}\n")
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
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the agentic LangGraph CRAG pipeline (hybrid retrieval + query "
             "rewriting + document grading + self-correction) instead of baseline RAG",
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
            agent=args.agent,
        )
    else:
        run_interactive_cli(
            vectordb_path=args.vectordb,
            retrieval_k=args.retrieval_k,
            show_sources=not args.no_sources,
            use_mmr=not args.no_mmr,
            refusal_threshold=args.refusal_threshold,
            structured_citations=not args.no_structured_citations,
            agent=args.agent,
        )


if __name__ == "__main__":
    main()
