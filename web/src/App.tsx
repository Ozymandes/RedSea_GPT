import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Header } from "./components/Header";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { MessageView } from "./components/Message";
import { ThinkingState } from "./components/ThinkingState";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import type { Tone } from "./components/ToneToggle";
import {
  type StoredChat,
  loadChats,
  upsertChat,
  deleteChat,
  getActiveId,
  setActiveId,
  newChatId,
  deriveTitle,
} from "./lib/history";

export default function App() {
  const [tone, setTone] = useState<Tone>("intuitive");

  // --- Persistent chat sessions -------------------------------------------
  const [chats, setChats] = useState<StoredChat[]>([]);
  const [activeId, setActiveChatId] = useState<string | null>(null);

  // The active chat's messages, loaded into the useChat hook.
  const { messages, loading, send, reset, error, setMessages } = useChat(
    tone,
    activeId ? chats.find((c) => c.id === activeId)?.messages ?? [] : []
  );

  // Sidebar open/closed. On desktop (md+) it's persistent; on mobile it's a drawer.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [booted, setBooted] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Hydrate from localStorage on mount.
  useEffect(() => {
    const stored = loadChats();
    setChats(stored);
    const aid = getActiveId();
    if (aid && stored.some((c) => c.id === aid)) {
      setActiveChatId(aid);
    } else if (stored.length > 0) {
      // auto-open the most recent
      setActiveChatId(stored[0].id);
    }
    setBooted(true);
  }, []);

  // Open sidebar by default on desktop.
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    const apply = () => setSidebarOpen(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  // Persist the active chat whenever its messages change.
  useEffect(() => {
    if (!booted || !activeId || messages.length === 0) return;
    const existing = chats.find((c) => c.id === activeId);
    const firstUser = messages.find((m) => m.role === "user");
    const updated: StoredChat = {
      id: activeId,
      title: existing?.title || (firstUser ? deriveTitle(firstUser.text) : "New chat"),
      createdAt: existing?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
      tone,
      messages,
    };
    setChats(upsertChat(updated));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, activeId, booted]);

  // Auto-scroll on new content.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const hasMessages = messages.length > 0;
  const engine = messages.find((m) => m.role === "assistant")?.engine ?? "baseline";

  const startNewChat = useCallback(() => {
    const id = newChatId();
    setActiveChatId(id);
    setActiveId(id);
    reset();
    if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
  }, [reset]);

  const selectChat = useCallback(
    (id: string) => {
      setActiveChatId(id);
      setActiveId(id);
      const found = chats.find((c) => c.id === id);
      if (found) setMessages(found.messages);
      if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
    },
    [chats, setMessages]
  );

  const removeChat = useCallback(
    (id: string) => {
      const next = deleteChat(id);
      setChats(next);
      if (activeId === id) {
        if (next.length > 0) {
          selectChat(next[0].id);
        } else {
          setActiveChatId(null);
          setActiveId(null);
          reset();
        }
      }
    },
    [activeId, selectChat, reset]
  );

  const handleSend = useCallback(
    (text: string) => {
      // If no active chat, start one with this first message.
      if (!activeId) {
        const id = newChatId();
        setActiveChatId(id);
        setActiveId(id);
      }
      send(text);
    },
    [activeId, send]
  );

  return (
    <div className="flex h-full">
      <Sidebar
        chats={chats}
        activeId={activeId}
        open={sidebarOpen}
        onNew={startNewChat}
        onSelect={selectChat}
        onDelete={removeChat}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Header
          engine={engine}
          onReset={startNewChat}
          hasMessages={hasMessages}
          tone={tone}
          onToneChange={setTone}
          onOpenSidebar={() => setSidebarOpen(true)}
          sidebarOpen={sidebarOpen}
        />

        <main ref={scrollRef} className="flex-1 overflow-y-auto">
          {!hasMessages ? (
            <EmptyState onPick={handleSend} />
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

        <Composer onSend={handleSend} disabled={loading || !booted} />

        {error && (
          <div className="pointer-events-none fixed bottom-24 left-1/2 -translate-x-1/2 rounded-lg border border-coral-soft bg-white px-3.5 py-2 text-[12px] text-coral-deep shadow-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
