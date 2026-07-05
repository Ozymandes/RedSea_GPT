import { motion } from "framer-motion";
import { DocIcon } from "./Icons";
import type { Citation } from "../lib/types";

interface Props {
  citation: Citation;
  active: boolean;
  index: number;
}

export function CitationCard({ citation, active, index }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, transform: "translateY(8px)" }}
      animate={{ opacity: 1, transform: "translateY(0)" }}
      transition={{
        duration: 0.4,
        delay: index * 0.05,
        ease: [0.23, 1, 0.32, 1],
      }}
      data-active={active}
      className={`rounded-lg border bg-white p-3.5 transition-colors duration-200 ${
        active ? "border-ink/40 ring-1 ring-ink/10" : "border-line hover:border-ink/25"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="shrink-0 text-ink/40">
            <DocIcon width={15} height={15} />
          </span>
          <span className="truncate font-mono text-[11px] font-medium text-ink-950">
            {citation.source}
          </span>
        </div>
        <span className="shrink-0 rounded-md bg-bone-100 px-1.5 py-0.5 font-mono text-[10px] text-muted">
          p.{citation.page ?? "?"}
        </span>
      </div>
      <p className="mt-2.5 line-clamp-4 text-[12.5px] leading-relaxed text-ink/70">
        {citation.content}
      </p>
      <div className="mt-2.5 flex items-center gap-1.5">
        <span className="inline-flex h-4 min-w-[1rem] items-center justify-center rounded bg-bone-100 px-1 font-mono text-[10px] font-medium text-ink/80">
          {citation.citation_id}
        </span>
        <span className="mono-label text-[10px]">source passage</span>
      </div>
    </motion.div>
  );
}
