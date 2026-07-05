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
  deleteChat as deleteChatFromStore,
  getActiveId,
  setActiveId,
  newChatId,
  deriveTitle,
} from "./lib/history";

export default function App() {
  const [tone, setTone] = useState<Tone>("intuitive");

  // --- Persistent chat sessions -------------------------------------------
  // `chats` is the SIDEBAR list (what to render). `messages` is the CURRENT
  // conversation, owned independently by useChat. These are deliberately NOT
  // derived from each other — deriving messages from chats created a render
  // loop (persist wrote fresh objects → find() returned a new array ref →
  // messages "reset" → persist fired again) that broke scrolling and made
  // deletes flicker. Loading a past chat is now an EXPLICIT action.
  const [chats, setChats] = useState<StoredChat[]>([]);
  const [activeId, setActiveChatId] = useState<string | null>(null);

  const { messages, loading, send, error, loadMessages, clearMessages } = useChat(tone);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [booted, setBooted] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevMsgLen = useRef(0);

  // Hydrate from localStorage on mount.
  useEffect(() => {
    const stored = loadChats();
    setChats(stored);
    const aid = getActiveId();
    if (aid && stored.some((c) => c.id === aid)) {
      setActiveChatId(aid);
      const found = stored.find((c) => c.id === aid);
      if (found) loadMessages(found.messages);
    } else if (stored.length > 0) {
      setActiveChatId(stored[0].id);
      setActiveId(stored[0].id);
      loadMessages(stored[0].messages);
    }
    setBooted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
  // Reads a FRESH list from localStorage (not the React `chats` closure, which
  // can be stale) so a concurrent delete elsewhere is never clobbered, and so
  // we never resurrect a chat that was removed.
  useEffect(() => {
    if (!booted || !activeId || messages.length === 0) return;
    const fresh = loadChats();
    const existing = fresh.find((c) => c.id === activeId);
    // If the active chat was deleted from storage, do NOT re-create it.
    // (Shouldn't happen now, but this is the definitive guard.)
    if (!existing) return;
    const firstUser = messages.find((m) => m.role === "user");
    const updated: StoredChat = {
      id: activeId,
      title: existing.title || (firstUser ? deriveTitle(firstUser.text) : "New chat"),
      createdAt: existing.createdAt,
      updatedAt: Date.now(),
      tone,
      messages,
    };
    setChats(upsertChat(updated));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, activeId, booted]);

  // Auto-scroll — ONLY when a message was added, and ONLY if the user is
  // already near the bottom. Never yanks the user down while they're reading
  // earlier messages.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const grew = messages.length > prevMsgLen.current;
    prevMsgLen.current = messages.length;
    if (!grew) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 180;
    if (nearBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages, loading]);

  const hasMessages = messages.length > 0;
  const engine = messages.find((m) => m.role === "assistant")?.engine ?? "baseline";

  const startNewChat = useCallback(() => {
    const id = newChatId();
    setActiveChatId(id);
    setActiveId(id);
    clearMessages();
    if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
  }, [clearMessages]);

  const selectChat = useCallback(
    (id: string) => {
      // Read fresh from storage so we always load the canonical messages.
      const found = loadChats().find((c) => c.id === id);
      setActiveChatId(id);
      setActiveId(id);
      if (found) loadMessages(found.messages);
      if (!window.matchMedia("(min-width: 768px)").matches) setSidebarOpen(false);
    },
    [loadMessages]
  );

  const removeChat = useCallback(
    (id: string) => {
      const next = deleteChatFromStore(id); // removes from localStorage
      setChats(next);
      if (activeId === id) {
        if (next.length > 0) {
          const fallback = next[0];
          setActiveChatId(fallback.id);
          setActiveId(fallback.id);
          loadMessages(fallback.messages);
        } else {
          setActiveChatId(null);
          setActiveId(null);
          clearMessages();
        }
      }
    },
    [activeId, loadMessages, clearMessages]
  );

  const handleSend = useCallback(
    (text: string) => {
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
