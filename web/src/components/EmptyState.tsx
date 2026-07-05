import { motion } from "framer-motion";
import { ShieldIcon } from "./Icons";
import type { Suggestion } from "../lib/types";

const SUGGESTIONS: Suggestion[] = [
  { label: "How did the Red Sea form?", prompt: "How did the Red Sea form geologically?" },
  { label: "Why is it so salty?", prompt: "Why is the Red Sea so much saltier than other seas?" },
  { label: "Heat-tolerant corals", prompt: "Why are some Red Sea corals unusually tolerant of high temperatures?" },
  { label: "Gulf of Aqaba", prompt: "Tell me about the Gulf of Aqaba and its depth." },
];

interface Props {
  onPick: (prompt: string) => void;
}

export function EmptyState({ onPick }: Props) {
  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center px-5 pb-24 pt-[12vh] text-center">
      {/* Ambient depth halo — very low opacity, evokes looking into deep water */}
      <div
        aria-hidden
        className="pointer-events-none fixed left-1/2 top-[18%] -z-10 h-[420px] w-[620px] -translate-x-1/2 rounded-full opacity-[0.06] blur-3xl"
        style={{
          background:
            "radial-gradient(circle, #0E3B43 0%, #155059 40%, transparent 70%)",
        }}
      />

      <motion.div
        initial={{ opacity: 0, transform: "translateY(12px)" }}
        animate={{ opacity: 1, transform: "translateY(0)" }}
        transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
      >
        <span className="mono-label">Egyptian Red Sea · Grounded Q&amp;A</span>
        <h1 className="font-editorial mt-5 text-[44px] font-medium leading-[1.02] text-ink-950 sm:text-[56px]">
          Ask the Red Sea.
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
          A retrieval-augmented assistant built on the marine science of the
          Egyptian Red Sea. Every claim cites its source — and when the sources
          don&rsquo;t cover it, it says so.
        </p>

        <div className="mt-9 flex flex-wrap items-center justify-center gap-2">
          {SUGGESTIONS.map((s, i) => (
            <motion.button
              key={s.label}
              initial={{ opacity: 0, transform: "translateY(8px)" }}
              animate={{ opacity: 1, transform: "translateY(0)" }}
              transition={{
                duration: 0.5,
                delay: 0.15 + i * 0.06,
                ease: [0.23, 1, 0.32, 1],
              }}
              onClick={() => onPick(s.prompt)}
              className="pressable rounded-lg border hairline bg-white px-3.5 py-2 text-[13px] font-medium text-ink/80 hover:text-ink-950 hover:bg-bone-50"
            >
              {s.label}
            </motion.button>
          ))}
        </div>

        <div className="mt-12 flex items-center justify-center gap-2 text-[12px] text-muted/80">
          <ShieldIcon width={13} height={13} />
          <span>
            gpt-4o-mini · 13 sources · refuses off-topic &amp; fabricated claims
          </span>
        </div>
      </motion.div>
    </div>
  );
}
