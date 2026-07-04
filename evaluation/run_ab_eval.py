#!/usr/bin/env python3
"""
A/B evaluation: baseline RAG vs agentic RAG (LangGraph CRAG) on the same golden set.

This is the experiment that lets us make a *defensible* claim about whether the
agentic pipeline (hybrid retrieval + query rewriting + document grading + self-
correction loop) actually beats the baseline (dense retrieval + MMR + refusal
heuristics). Both systems are scored with the identical transparent metrics, on
the identical 38 questions, with the identical provider/model — so any delta is
attributable to the pipeline, not the evaluator or the LLM.

Outputs (per A/B run, under ``eval_results/ab/<timestamp>/``):
    baseline/results.json, baseline/summary.json   - baseline detail + aggregates
    agent/results.json, agent/summary.json          - agent detail + aggregates
    comparison.json                                 - metric-by-metric delta + per-question paired verdict
    comparison.md                                   - human-readable A/B report with significance notes

Usage
-----
    python evaluation/run_ab_eval.py --provider optillm --model gpt-4o-mini

    # quick 5-question smoke comparison
    python evaluation/run_ab_eval.py --smoke

    # strict post-generation verifier on (ablation: does the verify-gate help or hurt?)
    python evaluation/run_ab_eval.py --strict-verify
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.golden_set import GOLDEN_SET
from evaluation.run_golden_eval import score_one, summarize, write_artifacts  # reuse identical scoring
from generation.llm_config import create_llm, describe_active_provider


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_system(name: str, gpt, questions: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Run a question list through a system (RedSeaGPT or RedSeaAgent) and score each."""
    results: List[Dict[str, Any]] = []
    n = len(questions)
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            out = gpt.query(q["question"], return_source_docs=True)
            latency = (time.perf_counter() - t0) * 1000.0
            scored = score_one(q, out, latency)
            status = (
                "REFUSE-OK" if scored["actually_refused"] and q["expected_behavior"] == "refuse"
                else ("PASS" if scored["passed"] else "FAIL")
            )
            print(f"  [{label} {i:>2}/{n}] {status:9s} {q['group']:16s} {q['question'][:42]}")
            results.append(scored)
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            print(f"  [{label} {i:>2}/{n}] ERROR     {q['group']:16s} {q['question'][:42]} -> {exc}")
            results.append({
                "id": q["id"], "category": q["category"], "group": q["group"],
                "question": q["question"], "expected_behavior": q["expected_behavior"],
                "actually_refused": True, "passed": False, "latency_ms": round(latency, 1),
                "metrics": {}, "answer_excerpt": f"ERROR: {exc}", "num_sources": 0, "error": str(exc),
            })
    return results


def _mc_nemar_pvalue(better: int, worse: int) -> float:
    """Approximate two-sided p-value for a paired sign test (McNemar without continuity).

    Used only as a *directional* significance hint on the discordant pairs, never
    as the sole basis for a claim. With 38 questions the power is low, so we report
    the discordant counts and the p-value together and flag low-n caveats.
    """
    n = better + worse
    if n == 0:
        return 1.0
    # Exact binomial two-sided p under H0: p=0.5
    from math import comb
    k = min(better, worse)
    tail = sum(comb(n, j) for j in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def build_comparison(
    baseline: List[Dict[str, Any]],
    agent: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Metric-by-metric delta + per-question paired verdict (the core of the A/B report)."""
    b_sum = summarize(baseline)
    a_sum = summarize(agent)

    # Per-question paired verdict
    b_by_id = {r["id"]: r for r in baseline}
    a_by_id = {r["id"]: r for r in agent}
    paired = []
    agent_only_pass = 0  # agent passed where baseline failed
    baseline_only_pass = 0  # baseline passed where agent failed
    both_pass = 0
    both_fail = 0
    for q in questions:
        b = b_by_id.get(q["id"], {})
        a = a_by_id.get(q["id"], {})
        bp, ap = bool(b.get("passed")), bool(a.get("passed"))
        if bp and ap:
            both_pass += 1
            verdict = "tie_pass"
        elif not bp and not ap:
            both_fail += 1
            verdict = "tie_fail"
        elif ap and not bp:
            agent_only_pass += 1
            verdict = "agent_win"
        else:
            baseline_only_pass += 1
            verdict = "baseline_win"
        paired.append({
            "id": q["id"], "group": q["group"], "question": q["question"],
            "baseline_passed": bp, "agent_passed": ap, "verdict": verdict,
            "baseline_latency_ms": b.get("latency_ms"),
            "agent_latency_ms": a.get("latency_ms"),
        })

    n_discordant = agent_only_pass + baseline_only_pass
    pval = _mc_nemar_pvalue(agent_only_pass, baseline_only_pass)

    def _delta(key, *path):
        """Extract a nested numeric from both summaries and return agent - baseline."""
        cur_b, cur_a = b_sum, a_sum
        for p in path:
            cur_b = (cur_b or {}).get(p, {})
            cur_a = (cur_a or {}).get(p, {})
        try:
            return round(float(cur_a) - float(cur_b), 4)
        except (TypeError, ValueError):
            return None

    return {
        "headline": {
            "baseline_pass_rate": b_sum.get("pass_rate"),
            "agent_pass_rate": a_sum.get("pass_rate"),
            "pass_rate_delta": round((a_sum.get("pass_rate", 0)) - (b_sum.get("pass_rate", 0)), 4),
            "baseline_severe_hallucinations": b_sum.get("severe_hallucinations"),
            "agent_severe_hallucinations": a_sum.get("severe_hallucinations"),
            "baseline_faithfulness": (b_sum.get("answerable") or {}).get("avg_faithfulness"),
            "agent_faithfulness": (a_sum.get("answerable") or {}).get("avg_faithfulness"),
            "baseline_refusal_accuracy": (b_sum.get("refusals") or {}).get("refusal_accuracy"),
            "agent_refusal_accuracy": (a_sum.get("refusals") or {}).get("refusal_accuracy"),
            "baseline_latency_mean_ms": (b_sum.get("latency_ms") or {}).get("mean"),
            "agent_latency_mean_ms": (a_sum.get("latency_ms") or {}).get("mean"),
        },
        "paired": {
            "both_pass": both_pass,
            "both_fail": both_fail,
            "agent_only_pass": agent_only_pass,
            "baseline_only_pass": baseline_only_pass,
            "discordant_pairs": n_discordant,
            "sign_test_pvalue": pval,
            "low_power_caveat": n_discordant < 8,  # sign test is weak with few discordant pairs
        },
        "per_question": paired,
    }


def write_comparison_md(out_dir: Path, comp: Dict[str, Any], b_sum, a_sum, meta):
    h = comp["headline"]
    p = comp["paired"]
    lines = []
    lines.append(f"# A/B Evaluation — Baseline vs Agentic RAG\n")
    lines.append(f"- **Generated:** {meta['timestamp']}")
    lines.append(f"- **Provider:** `{meta['provider']}` · **Model:** `{meta['model']}`")
    lines.append(f"- **Golden set:** {meta['golden_set_size']} questions\n")

    lines.append("## Headline (agent − baseline)\n")
    lines.append("| Metric | Baseline | Agent | Δ |")
    lines.append("|---|---:|---:|---:|")
    d = h["pass_rate_delta"]
    arrow = "🔺" if d > 0 else ("🔻" if d < 0 else "→")
    lines.append(f"| **Pass rate** | {h['baseline_pass_rate']:.1%} | {h['agent_pass_rate']:.1%} | {arrow} {d:+.1%} |")
    lines.append(f"| Severe hallucinations | {h['baseline_severe_hallucinations']} | {h['agent_severe_hallucinations']} | {h['agent_severe_hallucinations']-h['baseline_severe_hallucinations']:+d} |")
    bf = h["baseline_faithfulness"] or 0
    af = h["agent_faithfulness"] or 0
    lines.append(f"| Avg faithfulness (answerable) | {bf:.1%} | {af:.1%} | {af-bf:+.1%} |")
    br = h["baseline_refusal_accuracy"] or 0
    ar = h["agent_refusal_accuracy"] or 0
    lines.append(f"| Refusal accuracy | {br:.1%} | {ar:.1%} | {ar-br:+.1%} |")
    bl = h["baseline_latency_mean_ms"] or 0
    al = h["agent_latency_mean_ms"] or 0
    lines.append(f"| Latency mean (ms) | {bl:.0f} | {al:.0f} | {al-bl:+.0f} |\n")

    lines.append("## Paired statistical signal\n")
    lines.append(f"- Agent passed where baseline failed: **{p['agent_only_pass']}**")
    lines.append(f"- Baseline passed where agent failed: **{p['baseline_only_pass']}**")
    lines.append(f"- Both pass: {p['both_pass']} · Both fail: {p['both_fail']}")
    lines.append(f"- Discordant pairs: {p['discordant_pairs']} · two-sided sign-test p ≈ **{p['sign_test_pvalue']}**")
    if p["low_power_caveat"]:
        lines.append(f"- ⚠️ **Low-power caveat:** only {p['discordant_pairs']} discordant pairs; treat the "
                     f"direction as suggestive, not statistically conclusive.")
    lines.append("")

    lines.append("## Per-question verdicts (discordant only)\n")
    lines.append("| ID | Group | Question | Verdict |")
    lines.append("|---|---|---|---|")
    for q in comp["per_question"]:
        if q["verdict"] in ("agent_win", "baseline_win"):
            tag = "🟢 agent" if q["verdict"] == "agent_win" else "🔴 baseline"
            lines.append(f"| `{q['id']}` | {q['group']} | {q['question'][:60]} | {tag} |")
    lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="A/B compare baseline RAG vs agentic RAG.")
    ap.add_argument("--smoke", action="store_true", help="5-question smoke comparison")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrieval-k", type=int, default=5)
    ap.add_argument("--strict-verify", action="store_true",
                    help="Enable the agent's post-generation verify-gate (ablation)")
    ap.add_argument("--out-dir", default="eval_results/ab")
    args = ap.parse_args()

    info = describe_active_provider()
    if not info.get("configured"):
        print(f"ERROR: provider not configured: {info.get('error')}", file=sys.stderr)
        sys.exit(2)
    print(f"Provider: {info['provider']} | model: {info['model']} | base: {info['base_url']}\n")

    questions = GOLDEN_SET
    if args.smoke:
        seen, qs = set(), []
        for q in GOLDEN_SET:
            if q["group"] not in seen:
                qs.append(q); seen.add(q["group"])
            if len(qs) >= 5:
                break
        questions = qs

    # --- build both systems with the SAME provider/model ---
    print("Building baseline RedSeaGPT ...")
    from generation.rag_chain import RedSeaGPT
    baseline = RedSeaGPT(
        llm_config={"provider": args.provider, "model": args.model},
        retrieval_k=args.retrieval_k,
    )

    print("Building agentic RedSeaAgent (LangGraph CRAG) ...")
    from generation.agent import RedSeaAgent
    agent = RedSeaAgent(
        llm_config={"provider": args.provider, "model": args.model},
        retrieval_k=args.retrieval_k,
        strict_verify=args.strict_verify,
    )

    # --- run both on the same questions ---
    print(f"\n=== BASELINE ({len(questions)} Q) ===")
    b_results = _run_system("baseline", baseline, questions, "B")
    print(f"\n=== AGENT ({len(questions)} Q) ===")
    a_results = _run_system("agent", agent, questions, "A")

    b_sum = summarize(b_results)
    a_sum = summarize(a_results)
    comp = build_comparison(b_results, a_results, questions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # write per-system artifacts (reuse the standard writer)
    meta_b = {"run_name": "baseline", "timestamp": _now(), "provider": info,
              "config": {"retrieval_k": args.retrieval_k, "system": "baseline_RAG"}, "golden_set_size": len(questions)}
    meta_a = {"run_name": "agent", "timestamp": _now(), "provider": info,
              "config": {"retrieval_k": args.retrieval_k, "system": "agentic_C-RAG",
                         "strict_verify": args.strict_verify}, "golden_set_size": len(questions)}
    write_artifacts(out_dir / "baseline", b_results, b_sum, meta_b)
    write_artifacts(out_dir / "agent", a_results, a_sum, meta_a)

    with (out_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)
    write_comparison_md(out_dir, comp, b_sum, a_sum,
                        {"timestamp": _now(), "provider": info["provider"], "model": info["model"],
                         "golden_set_size": len(questions)})

    # --- console summary ---
    h = comp["headline"]
    p = comp["paired"]
    print("\n" + "=" * 64)
    print("A/B RESULT  (baseline → agent)")
    print(f"  Pass rate:       {h['baseline_pass_rate']:.1%} → {h['agent_pass_rate']:.1%}   (Δ {h['pass_rate_delta']:+.1%})")
    print(f"  Severe halluc.:  {h['baseline_severe_hallucinations']} → {h['agent_severe_hallucinations']}")
    print(f"  Faithfulness:    {h['baseline_faithfulness']:.1%} → {h['agent_faithfulness']:.1%}")
    print(f"  Refusal accuracy:{h['baseline_refusal_accuracy']:.1%} → {h['agent_refusal_accuracy']:.1%}")
    print(f"  Latency mean:    {h['baseline_latency_mean_ms']:.0f}ms → {h['agent_latency_mean_ms']:.0f}ms")
    print(f"  Discordant pairs: agent+{p['agent_only_pass']} / baseline+{p['baseline_only_pass']} "
          f"(sign-test p≈{p['sign_test_pvalue']})")
    print("=" * 64)
    print(f"Artifacts: {out_dir}/")


if __name__ == "__main__":
    main()
