import { motion } from "framer-motion";

export type Tone = "intuitive" | "technical";

interface Props {
  tone: Tone;
  onChange: (t: Tone) => void;
  disabled?: boolean;
}

const OPTIONS: { value: Tone; label: string }[] = [
  { value: "intuitive", label: "Educational" },
  { value: "technical", label: "Expert" },
];

// Two-option segmented control. The active pill uses framer-motion `layoutId`,
// so it slides between segments AND sizes itself to the active label's natural
// width. That fixes the earlier overflow where "Educational" (wider than
// "Expert") bled out of a hard-coded 50/50 split.
export function ToneToggle({ tone, onChange, disabled }: Props) {
  return (
    <div
      className={`relative flex items-center rounded-lg border hairline bg-white p-0.5 ${
        disabled ? "opacity-50" : ""
      }`}
      role="group"
      aria-label="Answer tone"
    >
      {OPTIONS.map((opt) => {
        const active = tone === opt.value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            disabled={disabled}
            aria-pressed={active}
            className={`pressable relative rounded-md px-3 py-1 text-[11.5px] font-medium transition-colors duration-200 ${
              active ? "text-bone" : "text-ink/65 hover:text-ink-950"
            }`}
          >
            {/* Active pill — animated, auto-sized to this label */}
            {active && (
              <motion.span
                layoutId="tone-pill"
                className="absolute inset-0 rounded-[7px] bg-ink"
                transition={{ duration: 0.26, ease: [0.23, 1, 0.32, 1] }}
              />
            )}
            <span className="relative z-10 whitespace-nowrap">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
