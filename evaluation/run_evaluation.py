"""
Run Evaluation Suite for RedSea GPT

Main script to run comprehensive evaluation of the RAG system.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generation.rag_chain import RedSeaGPT
from evaluation.questions import TEST_QUESTIONS, get_questions_by_category
from evaluation.metrics import (
    evaluate_answer_relevance,
    evaluate_retrieval_quality,
    evaluate_faithfulness,
    calculate_evaluation_summary,
    EvaluationResult,
)


def run_evaluation(
    gpt: RedSeaGPT,
    questions: List[Dict[str, Any]] = None,
    save_results: bool = True,
    output_dir: str = "evaluation_results",
) -> List[EvaluationResult]:
    """
    Run evaluation on a set of questions.

    Args:
        gpt: RedSeaGPT instance
        questions: List of question dictionaries (default: all test questions)
        save_results: Whether to save results to file
        output_dir: Directory to save results

    Returns:
        List of EvaluationResult objects
    """
    if questions is None:
        questions = TEST_QUESTIONS

    print(f"\n🔬 Running evaluation on {len(questions)} questions...")
    print(f"=" * 80)

    results = []

    for q in tqdm(questions, desc="Evaluating"):
        try:
            # Query the system
            result = gpt.query(q["question"], return_source_docs=True)

            # Evaluate relevance
            relevance = evaluate_answer_relevance(
                answer=result["answer"],
                question=q["question"],
                expected_keywords=q["expected_keywords"],
            )

            # Evaluate retrieval
            retrieval = evaluate_retrieval_quality(
                retrieved_docs=result["sources"],
                question=q["question"],
            )

            # Evaluate faithfulness (basic)
            context = " ".join([s["content"] for s in result["sources"]])
            faithfulness = evaluate_faithfulness(
                answer=result["answer"],
                context=context,
            )

            metrics = {
                "relevance": relevance,
                "retrieval": retrieval,
                "faithfulness": faithfulness,
            }

            # Create result object
            eval_result = EvaluationResult(
                question=q,
                answer=result["answer"],
                sources=result["sources"],
                metrics=metrics,
            )

            results.append(eval_result)

        except Exception as e:
            print(f"\n❌ Error evaluating question {q['id']}: {e}")
            continue

    # Save results if requested
    if save_results:
        save_evaluation_results(results, output_dir)

    return results


def save_evaluation_results(
    results: List[EvaluationResult],
    output_dir: str,
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: List of EvaluationResult objects
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save detailed results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {results_file}")

    # Save summary
    summary = calculate_evaluation_summary(results)
    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Summary saved to {summary_file}")


def print_evaluation_report(results: List[EvaluationResult]) -> None:
    """
    Print a comprehensive evaluation report.

    Args:
        results: List of EvaluationResult objects
    """
    print("\n" + "=" * 80)
    print("📊 EVALUATION REPORT")
    print("=" * 80)

    # Calculate summary statistics
    summary = calculate_evaluation_summary(results)

    print(f"\nTotal Questions: {summary['total_questions']}")
    print(f"Average Keyword Coverage: {summary['avg_keyword_coverage']:.2%}")

    # Performance by difficulty
    print(f"\n📈 Performance by Difficulty:")
    for difficulty, avg in summary["by_difficulty"].items():
        print(f"  {difficulty.capitalize()}: {avg:.2%}")

    # Performance by category
    print(f"\n📈 Performance by Category:")
    for category, avg in summary["by_category"].items():
        print(f"  {category}: {avg:.2%}")

    # Pass/fail analysis
    threshold = 0.5
    passed = sum(1 for r in results if r.is_passed(threshold))
    print(f"\n✅ Pass Rate (@ {threshold:.0%} threshold): {passed}/{len(results)} ({passed/len(results):.2%})")

    # Best and worst performing
    print(f"\n🏆 Best Performing Questions:")
    sorted_results = sorted(
        results,
        key=lambda r: r.metrics["relevance"]["keyword_coverage"],
        reverse=True,
    )
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. [{r.question['category']}] {r.question['question'][:60]}...")
        print(f"     Coverage: {r.metrics['relevance']['keyword_coverage']:.2%}")

    print(f"\n⚠️  Worst Performing Questions:")
    for i, r in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. [{r.question['category']}] {r.question['question'][:60]}...")
        print(f"     Coverage: {r.metrics['relevance']['keyword_coverage']:.2%}")

    print("\n" + "=" * 80)


def compare_prompt_variants(
    questions: List[Dict[str, Any]] = None,
    vectordb_path: str = "chroma_redsea",
) -> None:
    """
    Compare different prompt variants.

    Args:
        questions: Questions to test (default: subset of test questions)
        vectordb_path: Path to vector database
    """
    if questions is None:
        # Use a smaller subset for variant testing
        questions = TEST_QUESTIONS[:5]

    variants = ["basic", "cited", "structured"]

    print("\n🔬 Comparing prompt variants...")
    print("=" * 80)

    variant_results = {}

    for variant in variants:
        print(f"\nTesting variant: {variant}")

        gpt = RedSeaGPT(
            vectordb_path=vectordb_path,
            prompt_variant=variant,
        )

        results = run_evaluation(
            gpt=gpt,
            questions=questions,
            save_results=False,
        )

        summary = calculate_evaluation_summary(results)
        variant_results[variant] = summary

        print(f"  Average coverage: {summary['avg_keyword_coverage']:.2%}")

    # Print comparison
    print("\n" + "=" * 80)
    print("📊 VARIANT COMPARISON")
    print("=" * 80)

    for variant, summary in variant_results.items():
        print(f"\n{variant.upper()}:")
        print(f"  Avg Coverage: {summary['avg_keyword_coverage']:.2%}")


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RedSea GPT - Evaluation Suite"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Evaluate specific category only",
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Evaluate specific difficulty only",
    )
    parser.add_argument(
        "--retrieval-k", "-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Compare different prompt variants",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Initialize RAG system
    print(f"\n⏳ Initializing RedSea GPT...")
    gpt = RedSeaGPT(
        vectordb_path="chroma_redsea",
        retrieval_k=args.retrieval_k,
    )

    # Select questions
    if args.category:
        questions = get_questions_by_category(args.category)
    elif args.difficulty:
        from evaluation.questions import get_questions_by_difficulty
        questions = get_questions_by_difficulty(args.difficulty)
    else:
        questions = TEST_QUESTIONS

    print(f"📝 Evaluating {len(questions)} questions")

    # Run comparison or normal evaluation
    if args.compare_variants:
        compare_prompt_variants(questions)
    else:
        results = run_evaluation(
            gpt=gpt,
            questions=questions,
            save_results=True,
            output_dir=args.output_dir,
        )

        print_evaluation_report(results)


if __name__ == "__main__":
    main()
