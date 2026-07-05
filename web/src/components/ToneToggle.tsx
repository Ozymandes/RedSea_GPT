import { motion } from "framer-motion";

export type Tone = "intuitive" | "technical";

interface Props {
  tone: Tone;
  onChange: (t: Tone) => void;
  disabled?: boolean;
}

// A two-option segmented control. The active segment slides behind the label
// (layout animation), the inactive fades. Calm, scientific — not a flashy switch.
export function ToneToggle({ tone, onChange, disabled }: Props) {
  return (
    <div
      className={`relative flex items-center rounded-lg border hairline bg-white p-0.5 ${
        disabled ? "opacity-50" : ""
      }`}
      role="group"
      aria-label="Answer tone"
    >
      {/* Sliding indicator */}
      <motion.span
        layout
        transition={{ duration: 0.26, ease: [0.23, 1, 0.32, 1] }}
        className="absolute top-0.5 bottom-0.5 rounded-[7px] bg-ink"
        style={{
          left: tone === "intuitive" ? "2px" : "calc(50% + 0px)",
          right: tone === "intuitive" ? "calc(50% + 0px)" : "2px",
        }}
      />
      <button
        onClick={() => onChange("intuitive")}
        disabled={disabled}
        aria-pressed={tone === "intuitive"}
        className={`pressable relative z-10 flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11.5px] font-medium transition-colors duration-200 ${
          tone === "intuitive" ? "text-bone" : "text-ink/65 hover:text-ink-950"
        }`}
      >
        <span className="hidden sm:inline">Naturalist</span>
        <span className="sm:hidden">Simple</span>
      </button>
      <button
        onClick={() => onChange("technical")}
        disabled={disabled}
        aria-pressed={tone === "technical"}
        className={`pressable relative z-10 flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11.5px] font-medium transition-colors duration-200 ${
          tone === "technical" ? "text-bone" : "text-ink/65 hover:text-ink-950"
        }`}
      >
        <span className="hidden sm:inline">University</span>
        <span className="sm:hidden">Technical</span>
      </button>
    </div>
  );
}
