#!/usr/bin/env python3
"""
Reproducible evaluation runner for RedSea GPT.

Runs the golden set (``evaluation/golden_set.py``) through the full RAG pipeline
and scores every question with transparent, auditable metrics
(``evaluation/metrics_v2.py``). Writes machine- and human-readable artifacts to
``eval_results/<run_name>/``.

Usage
-----
    # full golden set against the env-configured provider (e.g. OptoLLM)
    python evaluation/run_golden_eval.py

    # small smoke run (5 questions, no LLM judge)
    python evaluation/run_golden_eval.py --smoke

    # compare mistral-small vs gpt-4o-mini on OptiLLM
    python evaluation/run_golden_eval.py --run-name mistral --provider optillm --model mistral-small
    python evaluation/run_golden_eval.py --run-name gpt4omini --provider optillm --model gpt-4o-mini

    # opt-in LLM-as-judge for clarity (writes judge prompts + verdicts)
    python evaluation/run_golden_eval.py --judge

Outputs (per run)
-----------------
    eval_results/<run_name>/results.json      - per-question detail
    eval_results/<run_name>/summary.json      - aggregate metrics
    eval_results/<run_name>/results.csv       - flat table
    eval_results/<run_name>/report.md         - human-readable report
    eval_results/<run_name>/judge.jsonl       - (only if --judge) auditable LLM judge I/O
    eval_results/<run_name>/run_meta.json     - provider/model/timestamp (no secrets)

This runner never prints API keys. If the configured provider is unreachable,
it fails fast with a secret-free message.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.golden_set import GOLDEN_SET, answerable_questions, refusal_questions
from evaluation.metrics_v2 import (
    evaluate_citation_presence,
    evaluate_citation_support,
    evaluate_concept_coverage,
    evaluate_faithfulness,
    evaluate_hallucination,
    evaluate_refusal_correctness,
    evaluate_retrieval,
    is_refusal,
    judge_with_llm,
)
from generation.llm_config import create_llm, describe_active_provider
from generation.rag_chain import RedSeaGPT


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(0.95 * (len(s) - 1)))
    return s[idx]


def score_one(question: Dict[str, Any], result: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
    """Compute all metrics for a single question. Pure function -> auditable."""
    answer = result.get("answer", "") or ""
    sources = result.get("sources", []) or []
    # Use the FULL retrieved chunk text (the actual context the model saw),
    # not the 300-char truncated source excerpts, for faithfulness scoring.
    chunks = result.get("retrieved_chunks", []) or sources
    context = " ".join(c.get("page_content", c.get("content", "")) for c in chunks)

    retrieval = evaluate_retrieval(sources)
    correctness = evaluate_concept_coverage(answer, question.get("required_concepts", []))
    cit_presence = evaluate_citation_presence(answer, question.get("min_citations", 1))
    cit_support = evaluate_citation_support(answer, sources)
    faith = evaluate_faithfulness(answer, context)
    refusal = evaluate_refusal_correctness(answer, question["expected_behavior"])
    halluc = evaluate_hallucination(question, answer, sources)

    # Overall pass logic (deliberately strict):
    #   - refusal questions pass iff correctly refused.
    #   - answerable questions pass iff NOT a serious failure AND concept coverage ok.
    if question["expected_behavior"] == "refuse":
        passed = refusal["correct"]
    else:
        passed = (
            not halluc["severe_hallucination"]
            and correctness["ok"]
            and (not question.get("citation_required") or cit_support["ok"])
        )

    return {
        "id": question["id"],
        "category": question["category"],
        "group": question["group"],
        "question": question["question"],
        "expected_behavior": question["expected_behavior"],
        "actually_refused": refusal["actually_refused"],
        "passed": passed,
        "latency_ms": round(latency_ms, 1),
        "metrics": {
            "retrieval": retrieval,
            "concept_coverage": correctness,
            "citation_presence": cit_presence,
            "citation_support": cit_support,
            "faithfulness": faith,
            "refusal_correctness": refusal,
            "hallucination": halluc,
        },
        "answer_excerpt": answer[:600],
        "num_sources": len(sources),
    }


def run(
    gpt: RedSeaGPT,
    questions: List[Dict[str, Any]],
    use_judge: bool = False,
    judge_llm=None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    n = len(questions)
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            out = gpt.query(q["question"], return_source_docs=True)
            latency = (time.perf_counter() - t0) * 1000.0
            scored = score_one(q, out, latency)
            status = "REFUSE-OK" if scored["actually_refused"] and q["expected_behavior"] == "refuse" else (
                "PASS" if scored["passed"] else "FAIL"
            )
            print(f"  [{i:>2}/{n}] {status:9s} {q['group']:18s} {q['question'][:50]}")

            if use_judge and judge_llm is not None and q["expected_behavior"] == "answer":
                ctx = " ".join(s.get("content", "") for s in out.get("sources", []))
                scored["judge"] = judge_with_llm(
                    judge_llm, q["question"], out.get("answer", ""), ctx, q["expected_behavior"]
                )
            results.append(scored)
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            print(f"  [{i:>2}/{n}] ERROR     {q['group']:18s} {q['question'][:50]} -> {exc}")
            results.append({
                "id": q["id"], "category": q["category"], "group": q["group"],
                "question": q["question"], "expected_behavior": q["expected_behavior"],
                "actually_refused": True, "passed": False, "latency_ms": round(latency, 1),
                "metrics": {}, "answer_excerpt": f"ERROR: {exc}", "num_sources": 0, "error": str(exc),
            })
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if not total:
        return {"total": 0}

    passed = sum(1 for r in results if r["passed"])
    severe = sum(1 for r in results if r.get("metrics", {}).get("hallucination", {}).get("severe_hallucination"))

    ans = [r for r in results if r["expected_behavior"] == "answer"]
    ref = [r for r in results if r["expected_behavior"] == "refuse"]

    # metrics that only make sense on answerable questions
    coverages = [r["metrics"]["concept_coverage"]["coverage"] for r in ans if "metrics" in r and "concept_coverage" in r.get("metrics", {})]
    faiths = [r["metrics"]["faithfulness"]["faithfulness"] for r in ans if "metrics" in r and "faithfulness" in r.get("metrics", {})]
    cit_ok = [r["metrics"]["citation_support"]["ok"] for r in ans if "metrics" in r and "citation_support" in r.get("metrics", {})]
    cit_present = [r["metrics"]["citation_presence"]["num_distinct"] > 0 for r in ans if "metrics" in r and "citation_presence" in r.get("metrics", {})]

    refusal_correct = sum(1 for r in ref if r.get("metrics", {}).get("refusal_correctness", {}).get("correct"))

    latencies = [r["latency_ms"] for r in results]

    by_group: Dict[str, Dict[str, int]] = {}
    for r in results:
        g = r["group"]
        d = by_group.setdefault(g, {"total": 0, "passed": 0})
        d["total"] += 1
        if r["passed"]:
            d["passed"] += 1

    return {
        "total": total,
        "pass_rate": round(passed / total, 4),
        "passed": passed,
        "severe_hallucinations": severe,
        "answerable": {
            "count": len(ans),
            "avg_concept_coverage": round(statistics.mean(coverages), 4) if coverages else 0.0,
            "avg_faithfulness": round(statistics.mean(faiths), 4) if faiths else 0.0,
            "citation_coverage": round(sum(cit_ok) / len(cit_ok), 4) if cit_ok else 0.0,
            "citation_presence_rate": round(sum(cit_present) / len(cit_present), 4) if cit_present else 0.0,
        },
        "refusals": {
            "count": len(ref),
            "correct": refusal_correct,
            "refusal_accuracy": round(refusal_correct / len(ref), 4) if ref else 0.0,
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
            "median": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": round(_p95(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "by_group": by_group,
    }


def write_artifacts(out_dir: Path, results: List[Dict[str, Any]], summary: Dict[str, Any], meta: Dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # flat CSV
    cols = ["id", "category", "group", "expected_behavior", "actually_refused", "passed",
            "latency_ms", "num_sources", "concept_coverage", "faithfulness",
            "citations_present", "citations_supported", "severe_hallucination"]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            m = r.get("metrics", {})
            w.writerow([
                r.get("id"), r.get("category"), r.get("group"),
                r.get("expected_behavior"), r.get("actually_refused"), r.get("passed"),
                r.get("latency_ms"), r.get("num_sources"),
                m.get("concept_coverage", {}).get("coverage", ""),
                m.get("faithfulness", {}).get("faithfulness", ""),
                m.get("citation_presence", {}).get("num_distinct", 0),
                len(m.get("citation_support", {}).get("supported", [])),
                m.get("hallucination", {}).get("severe_hallucination", ""),
            ])

    # judge artifact
    judged = [r for r in results if "judge" in r]
    if judged:
        with (out_dir / "judge.jsonl").open("w", encoding="utf-8") as f:
            for r in judged:
                f.write(json.dumps({"id": r["id"], **r["judge"]}, ensure_ascii=False) + "\n")

    # human report
    rep = []
    rep.append(f"# Evaluation Report — {meta.get('run_name', 'run')}\n")
    rep.append(f"- **Generated:** {meta.get('timestamp')}")
    rep.append(f"- **Provider:** `{meta['provider'].get('provider')}` ({meta['provider'].get('protocol')})")
    rep.append(f"- **Model:** `{meta['provider'].get('model')}`")
    rep.append(f"- **Base URL:** `{meta['provider'].get('base_url')}`\n")
    rep.append("## Headline metrics\n")
    rep.append(f"- Total questions: **{summary['total']}**")
    rep.append(f"- Pass rate: **{summary['pass_rate']:.1%}** ({summary['passed']}/{summary['total']})")
    rep.append(f"- Severe hallucinations: **{summary['severe_hallucinations']}**")
    a = summary["answerable"]
    rep.append(f"- Avg concept coverage (answerable): **{a['avg_concept_coverage']:.1%}**")
    rep.append(f"- Avg faithfulness (answerable): **{a['avg_faithfulness']:.1%}**")
    rep.append(f"- Citation support (answerable): **{a['citation_coverage']:.1%}**")
    rf = summary["refusals"]
    rep.append(f"- Refusal accuracy: **{rf['refusal_accuracy']:.1%}** ({rf['correct']}/{rf['count']})")
    lat = summary["latency_ms"]
    rep.append(f"- Latency: mean **{lat['mean']:.0f}ms** · p95 **{lat['p95']:.0f}ms** · max **{lat['max']:.0f}ms**\n")
    rep.append("## By group\n")
    rep.append("| Group | Pass | Total |")
    rep.append("|---|---:|---:|")
    for g, d in sorted(summary["by_group"].items()):
        rep.append(f"| {g} | {d['passed']} | {d['total']} |")
    rep.append("\n## Per-question detail\n")
    for r in results:
        flag = "✅" if r["passed"] else "❌"
        rep.append(f"### {flag} `{r['id']}` — {r['question']}")
        rep.append(f"- Group: `{r['group']}` · Expected: `{r['expected_behavior']}` · Refused: `{r['actually_refused']}` · Latency: {r['latency_ms']:.0f}ms")
        m = r.get("metrics", {})
        if m:
            cc = m.get("concept_coverage", {})
            fh = m.get("faithfulness", {})
            cs = m.get("citation_support", {})
            rep.append(f"- Concept coverage: {cc.get('coverage')} · Faithfulness: {fh.get('faithfulness')} · Citations supported: {cs.get('supported')}")
            hl = m.get("hallucination", {})
            if hl.get("severe_hallucination"):
                rep.append(f"- ⚠️ **Severe hallucination:** {hl.get('reason')}")
        rep.append("")
    (out_dir / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"\nArtifacts written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Run the RedSea GPT golden evaluation.")
    ap.add_argument("--smoke", action="store_true", help="Small 5-question smoke run")
    ap.add_argument("--provider", default=None, help="Force provider (optillm|groq|openai|openai-compatible)")
    ap.add_argument("--model", default=None, help="Override model (e.g. mistral-small, gpt-4o-mini)")
    ap.add_argument("--retrieval-k", type=int, default=5)
    ap.add_argument("--refusal-threshold", type=float, default=0.2)
    ap.add_argument("--run-name", default=None, help="Output folder name under eval_results/")
    ap.add_argument("--judge", action="store_true", help="Enable LLM-as-judge for clarity (writes judge.jsonl)")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    provider_info = describe_active_provider()  # secret-free
    if not provider_info.get("configured"):
        print("ERROR: No LLM provider configured. Set OPTO_LLM_API_KEY / GROQ_API_KEY in .env.", file=sys.stderr)
        print(f"  detail: {provider_info.get('error')}", file=sys.stderr)
        sys.exit(2)

    # Build the LLM with optional overrides.
    llm = create_llm(provider=args.provider, model=args.model)
    info = describe_active_provider()
    print(f"Provider: {info['provider']} | model: {info['model']} | base: {info['base_url']}")

    gpt = RedSeaGPT(
        llm_config={"provider": args.provider, "model": args.model},
        retrieval_k=args.retrieval_k,
        refusal_threshold=args.refusal_threshold,
    )

    questions = GOLDEN_SET
    if args.smoke:
        # one per group + keep some refusal/trap ones
        seen = set()
        qs = []
        for q in GOLDEN_SET:
            if q["group"] not in seen:
                qs.append(q); seen.add(q["group"])
            if len(qs) >= 5:
                break
        questions = qs

    print(f"\nEvaluating {len(questions)} questions...\n")
    judge_llm = None
    if args.judge:
        judge_llm = create_llm(provider=args.provider, model=args.model)

    results = run(gpt, questions, use_judge=args.judge, judge_llm=judge_llm)
    summary = summarize(results)

    run_name = args.run_name or f"{info['provider']}_{info['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = PROJECT_ROOT / args.out_dir / run_name
    meta = {
        "run_name": run_name,
        "timestamp": _now(),
        "provider": info,
        "config": {
            "retrieval_k": args.retrieval_k,
            "refusal_threshold": args.refusal_threshold,
            "use_mmr": True,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "smoke": args.smoke,
        },
        "golden_set_size": len(GOLDEN_SET),
    }
    write_artifacts(out_dir, results, summary, meta)

    print("\n" + "=" * 60)
    print(f"PASS RATE: {summary['pass_rate']:.1%} ({summary['passed']}/{summary['total']})")
    print(f"SEVERE HALLUCINATIONS: {summary['severe_hallucinations']}")
    print(f"REFUSAL ACCURACY: {summary['refusals']['refusal_accuracy']:.1%}")
    print(f"FAITHFULNESS (answerable): {summary['answerable']['avg_faithfulness']:.1%}")
    print(f"LATENCY mean/p95: {summary['latency_ms']['mean']:.0f}/{summary['latency_ms']['p95']:.0f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
