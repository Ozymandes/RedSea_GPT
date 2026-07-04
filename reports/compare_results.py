import json

# Read TinyLlama baseline
with open('evaluation_results/evaluation_results.json', 'r', encoding='utf-8') as f:
    baseline = json.load(f)

# Read Groq results
with open('evaluation_results/groq_final_results.json/evaluation_results.json', 'r', encoding='utf-8') as f:
    groq = json.load(f)

# Calculate metrics
def calc_metrics(results):
    passed = sum(1 for r in results if r['metrics']['relevance']['keyword_coverage'] >= 0.5)
    total = len(results)

    avg_coverage = sum(r['metrics']['relevance']['keyword_coverage'] for r in results) / total
    avg_faithfulness = sum(r['metrics']['faithfulness']['faithfulness'] for r in results) / total
    avg_diversity = sum(r['metrics']['retrieval']['source_diversity'] for r in results) / total

    hallucinations = sum(1 for r in results if r['metrics']['faithfulness']['grounded_sentences'] / r['metrics']['faithfulness']['total_sentences'] < 0.5)

    return {
        'pass_rate': passed / total,
        'passed': passed,
        'total': total,
        'avg_coverage': avg_coverage,
        'avg_faithfulness': avg_faithfulness,
        'avg_diversity': avg_diversity,
        'hallucinations': hallucinations
    }

baseline_metrics = calc_metrics(baseline)
groq_metrics = calc_metrics(groq)

print('=' * 80)
print('EVALUATION RESULTS: TinyLlama 1.1B vs Groq Llama 70B')
print('=' * 80)
print()
print(f"{'Metric':<30} {'TinyLlama':<20} {'Groq 70B':<20} {'Improvement':<15}")
print('-' * 80)
print(f"{'Pass Rate':<30} {baseline_metrics['pass_rate']:<20.2%} {groq_metrics['pass_rate']:<20.2%} {(groq_metrics['pass_rate'] - baseline_metrics['pass_rate']):+.1%}")
print(f"{'Passed/Total':<30} {baseline_metrics['passed']}/{baseline_metrics['total']:<15} {groq_metrics['passed']}/{groq_metrics['total']:<15} {(groq_metrics['passed'] - baseline_metrics['passed']):+d}")
print(f"{'Avg Keyword Coverage':<30} {baseline_metrics['avg_coverage']:<20.1%} {groq_metrics['avg_coverage']:<20.1%} {(groq_metrics['avg_coverage'] - baseline_metrics['avg_coverage']):+.1%}")
print(f"{'Avg Faithfulness':<30} {baseline_metrics['avg_faithfulness']:<20.1%} {groq_metrics['avg_faithfulness']:<20.1%} {(groq_metrics['avg_faithfulness'] - baseline_metrics['avg_faithfulness']):+.1%}")
print(f"{'Avg Source Diversity':<30} {baseline_metrics['avg_diversity']:<20.2f} {groq_metrics['avg_diversity']:<20.2f} {(groq_metrics['avg_diversity'] - baseline_metrics['avg_diversity']):+.2f}")
print(f"{'Severe Hallucinations':<30} {baseline_metrics['hallucinations']:<20} {groq_metrics['hallucinations']:<20} {baseline_metrics['hallucinations'] - groq_metrics['hallucinations']:+d}")
print()
print('=' * 80)
