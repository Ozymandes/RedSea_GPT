import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Header } from "./components/Header";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { MessageView } from "./components/Message";
import { ThinkingState } from "./components/ThinkingState";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, loading, send, reset, error } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [booted, setBooted] = useState(false);

  // Gentle "warming up" state for cold starts (Render free tier sleeps).
  useEffect(() => {
    const t = window.setTimeout(() => setBooted(true), 500);
    return () => window.clearTimeout(t);
  }, []);

  // Auto-scroll to the latest message on new content.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const hasMessages = messages.length > 0;
  const engine = messages.find((m) => m.role === "assistant")?.engine ?? "baseline";

  return (
    <div className="flex h-full flex-col">
      <Header engine={engine} onReset={reset} hasMessages={hasMessages} />

      <main ref={scrollRef} className="flex-1 overflow-y-auto">
        {!hasMessages ? (
          <EmptyState onPick={send} />
        ) : (
          <div className="mx-auto max-w-3xl px-5 py-7">
            <div className="flex flex-col gap-7">
              <AnimatePresence initial={false}>
                {messages.map((m) => (
                  <MessageView key={m.id} message={m} />
                ))}
              </AnimatePresence>

              {loading && (
                <motion.div
                  initial={{ opacity: 0, transform: "translateY(6px)" }}
                  animate={{ opacity: 1, transform: "translateY(0)" }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
                >
                  <ThinkingState />
                </motion.div>
              )}
            </div>
          </div>
        )}
      </main>

      <Composer onSend={send} disabled={loading || !booted} />

      {error && (
        <div className="pointer-events-none fixed bottom-24 left-1/2 -translate-x-1/2 rounded-lg border border-coral-soft bg-white px-3.5 py-2 text-[12px] text-coral-deep shadow-sm">
          {error}
        </div>
      )}
    </div>
  );
}
