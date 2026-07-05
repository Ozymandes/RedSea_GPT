import { useState } from "react";
import { motion } from "framer-motion";
import type { ChatMessage } from "../lib/types";
import { ReasoningPanel } from "./ReasoningPanel";
import { CitationCard } from "./CitationCard";
import { ShieldIcon } from "./Icons";
import { AnswerBody } from "./AnswerBody";

interface Props {
  message: ChatMessage;
}

// Render the assistant answer: turn [n] markers into interactive chips that
// scroll to / highlight the matching citation card. We do NOT render arbitrary
// markdown HTML from the model for safety; we render paragraphs + inline chips.

export function MessageView({ message }: Props) {
  const [activeCitation, setActiveCitation] = useState<number | null>(null);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <motion.div
          initial={{ opacity: 0, transform: "translateY(6px)" }}
          animate={{ opacity: 1, transform: "translateY(0)" }}
          transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
          className="max-w-[85%] rounded-2xl rounded-br-md bg-ink px-3.5 py-2.5 text-[14.5px] leading-relaxed text-bone"
        >
          {message.text}
        </motion.div>
      </div>
    );
  }

  // Assistant
  const scrollToCitation = (id: number) => {
    setActiveCitation(id);
    const el = document.getElementById(`cite-${message.id}-${id}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    window.setTimeout(() => setActiveCitation((cur) => (cur === id ? id : cur)), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateY(8px)" }}
      animate={{ opacity: 1, transform: "translateY(0)" }}
      transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
      className="flex flex-col gap-3"
    >
      {/* Refusal — styled as a confident "not in sources", never an error red */}
      {message.refused && !message.error ? (
        <div className="flex items-start gap-2.5 rounded-xl border border-ink/15 bg-ink/[0.03] px-4 py-3">
          <span className="mt-0.5 text-ink/60">
            <ShieldIcon width={15} height={15} />
          </span>
          <div>
            <div className="mono-label mb-0.5 text-[10px] text-ink/60">
              Not covered in the sources
            </div>
            <p className="text-[14.5px] leading-relaxed text-ink-950">
              {message.text || "I can't answer that from the available sources."}
            </p>
          </div>
        </div>
      ) : message.error ? (
        <div className="rounded-xl border border-coral-soft bg-coral-soft/20 px-4 py-3">
          <div className="mono-label mb-0.5 text-[10px] text-coral-deep">Error</div>
          <p className="font-mono text-[12.5px] text-ink-950">
            {message.error}
          </p>
        </div>
      ) : (
        <AnswerBody text={message.text} onChip={scrollToCitation} />
      )}

      {/* Reasoning — collapsible, shows resolved-question + trace */}
      {!message.error && (
        <ReasoningPanel reasoning={message.reasoning} rawUser={message.raw_user} />
      )}

      {/* Citations — always visible (the proof). Hidden on pure refusals with no sources. */}
      {message.citations.length > 0 && (
        <div className="mt-1">
          <div className="mono-label mb-2 flex items-center gap-1.5">
            <span className="h-1 w-1 rounded-full bg-coral" />
            Sources · {message.citations.length} passages
          </div>
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            {message.citations.map((c, i) => (
              <div key={c.citation_id} id={`cite-${message.id}-${c.citation_id}`}>
                <CitationCard citation={c} active={activeCitation === c.citation_id} index={i} />
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
