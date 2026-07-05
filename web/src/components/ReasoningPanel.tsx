import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronIcon, SparkIcon } from "./Icons";
import type { ReasoningTrace } from "../lib/types";

interface Props {
  reasoning: ReasoningTrace;
  rawUser: string; // to show "you asked X -> understood Y"
}

function Stat({ label, value, hint }: { label: string; value: string | null; hint?: string }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex flex-col gap-0.5" title={hint}>
      <span className="mono-label text-[9.5px]">{label}</span>
      <span className="font-mono text-[12px] text-ink-950">{value}</span>
    </div>
  );
}

export function ReasoningPanel({ reasoning, rawUser }: Props) {
  const [open, setOpen] = useState(false);

  const r = reasoning;
  const resolved = r.resolved_question && r.resolved_question.trim() !== rawUser.trim();
  const method = r.retrieval_method === "graph_crag" ? "CRAG graph" : r.retrieval_method ?? null;
  const confidence = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : null;

  const hasAny = Boolean(
    resolved || method || r.route || r.retrieval_rounds != null || r.verification || r.node_trace
  );

  // Collapsed summary line
  const summaryParts: string[] = [];
  if (method) summaryParts.push(method);
  if (r.retrieval_rounds != null) summaryParts.push(`${r.retrieval_rounds} round${r.retrieval_rounds === 1 ? "" : "s"}`);
  if (confidence) summaryParts.push(`${confidence} relevance`);
  const summary = summaryParts.join(" · ");

  return (
    <div className="rounded-lg border hairline bg-bone-50">
      <button
        onClick={() => hasAny && setOpen((o) => !o)}
        className={`pressable flex w-full items-center justify-between gap-2 px-3.5 py-2.5 text-left ${
          hasAny ? "cursor-pointer" : "cursor-default"
        }`}
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-coral">
            <SparkIcon width={14} height={14} />
          </span>
          <span className="mono-label">Reasoning</span>
          {summary && (
            <span className="truncate font-mono text-[11px] text-muted">{summary}</span>
          )}
        </div>
        {hasAny && (
          <motion.span
            animate={{ rotate: open ? 90 : 0 }}
            transition={{ duration: 0.2, ease: [0.23, 1, 0.32, 1] }}
            className="text-muted"
          >
            <ChevronIcon width={14} height={14} />
          </motion.span>
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && hasAny && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.26, ease: [0.23, 1, 0.32, 1] }}
            className="overflow-hidden"
          >
            <div className="border-t hairline px-3.5 py-3.5">
              {resolved && (
                <div className="mb-3.5 rounded-md border border-coral-soft bg-coral-soft/30 p-2.5">
                  <div className="mono-label mb-1 text-[9.5px] text-coral-deep">
                    Conversation memory
                  </div>
                  <div className="text-[12.5px] leading-snug text-ink-950">
                    <span className="text-muted">You asked:</span>{" "}
                    <span className="italic">&ldquo;{rawUser}&rdquo;</span>
                    <br />
                    <span className="text-muted">Understood as:</span>{" "}
                    <span className="font-medium">&ldquo;{r.resolved_question}&rdquo;</span>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Route" value={r.route ?? null} />
                <Stat label="Retrieval" value={method} />
                <Stat
                  label="Rounds"
                  value={r.retrieval_rounds != null ? String(r.retrieval_rounds) : null}
                />
                <Stat
                  label="Source relevance"
                  value={confidence}
                  hint="Average match between the question and the retrieved passages (cosine similarity). This is a retrieval-match signal, not a guarantee of correctness."
                />
              </div>

              {r.verification &&
                "ok" in r.verification &&
                (r.verification as Record<string, unknown>).faithfulness != null && (
                  <div className="mt-3 border-t hairline pt-3">
                    <div className="mono-label mb-1 text-[9.5px]">Verification (claim entailment)</div>
                    <p className="font-mono text-[12px] text-ink-950">
                      faithfulness{" "}
                      {Math.round(
                        ((r.verification as Record<string, number>).faithfulness ?? 0) * 100
                      )}
                      % · {(r.verification as Record<string, number>).n_claims ?? 0} claims
                    </p>
                  </div>
                )}

              {r.node_trace && Object.keys(r.node_trace).length > 0 && (
                <div className="mt-3 border-t hairline pt-3">
                  <div className="mono-label mb-1.5 text-[9.5px]">Graph trace (per-node, seconds)</div>
                  <div className="flex flex-wrap gap-x-3 gap-y-1">
                    {Object.entries(r.node_trace).map(([k, v]) => (
                      <span key={k} className="font-mono text-[11px] text-muted">
                        <span className="text-ink-950">{k}</span> {v.toFixed(2)}s
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
