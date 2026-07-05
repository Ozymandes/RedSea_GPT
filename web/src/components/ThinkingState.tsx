import { useEffect, useState } from "react";
import { motion } from "framer-motion";

// Rotating status lines give honest perceived progress during the ~15s OptiLLM
// latency. Calm pulse, not a frantic spinner — matches the "scientific" mood.
const STEPS = [
  "Searching the corpus…",
  "Retrieving passages…",
  "Grading relevance…",
  "Reading sources…",
  "Composing the answer…",
];

export function ThinkingState() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((x) => (x + 1) % STEPS.length), 2600);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-2.5 py-1">
      <motion.span
        className="inline-block h-1.5 w-1.5 rounded-full bg-coral"
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.span
        key={i}
        initial={{ opacity: 0, transform: "translateY(3px)" }}
        animate={{ opacity: 1, transform: "translateY(0)" }}
        transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
        className="mono-label text-ink/70"
      >
        {STEPS[i]}
      </motion.span>
    </div>
  );
}
