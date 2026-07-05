import { motion, AnimatePresence } from "framer-motion";
import { WaveIcon, ResetIcon } from "./Icons";
import type { StoredChat } from "../lib/history";

interface Props {
  chats: StoredChat[];
  activeId: string | null;
  open: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function Sidebar({ chats, activeId, open, onNew, onSelect, onDelete, onClose }: Props) {
  return (
    <>
      {/* Mobile scrim — click to dismiss */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            className="fixed inset-0 z-30 bg-ink-950/20 backdrop-blur-[1px] md:hidden"
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {open && (
          <motion.aside
            initial={{ transform: "translateX(-100%)" }}
            animate={{ transform: "translateX(0)" }}
            exit={{ transform: "translateX(-100%)" }}
            transition={{ duration: 0.28, ease: [0.23, 1, 0.32, 1] }}
            className="fixed left-0 top-0 z-40 flex h-full w-[280px] flex-col border-r hairline bg-bone-50 md:static md:z-0 md:translate-x-0"
          >
            {/* Brand row */}
            <div className="flex h-14 items-center justify-between border-b hairline px-4">
              <div className="flex items-center gap-2">
                <span className="text-ink">
                  <WaveIcon width={18} height={18} />
                </span>
                <span className="font-editorial text-[16px] font-medium text-ink-950">
                  RedSea
                </span>
                <span className="mono-label">GPT</span>
              </div>
              <button
                onClick={onClose}
                className="pressable rounded-md p-1.5 text-muted hover:bg-bone-100 hover:text-ink"
                aria-label="Collapse sidebar"
                title="Collapse"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M15 6l-6 6 6 6" />
                </svg>
              </button>
            </div>

            {/* New chat */}
            <div className="p-3">
              <button
                onClick={onNew}
                className="pressable flex w-full items-center justify-center gap-2 rounded-lg bg-ink px-3 py-2.5 text-[13px] font-medium text-bone hover:bg-ink-700"
              >
                <ResetIcon width={14} height={14} />
                New chat
              </button>
            </div>

            {/* Chat list */}
            <div className="flex-1 overflow-y-auto px-2 pb-3">
              <div className="mono-label px-2 py-2">Recent</div>
              {chats.length === 0 ? (
                <p className="px-2 py-6 text-center text-[12px] text-muted/70">
                  No conversations yet.
                </p>
              ) : (
                <ul className="flex flex-col gap-0.5">
                  {chats.map((c) => {
                    const active = c.id === activeId;
                    return (
                      <li key={c.id} className="group relative">
                        <button
                          onClick={() => onSelect(c.id)}
                          className={`pressable block w-full rounded-md px-2.5 py-2 pr-8 text-left ${
                            active ? "bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)]" : "hover:bg-white/60"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span
                              className={`truncate text-[13px] ${
                                active ? "font-medium text-ink-950" : "text-ink/75"
                              }`}
                            >
                              {c.title}
                            </span>
                          </div>
                          <div className="mt-0.5 flex items-center gap-1.5">
                            <span className="font-mono text-[10px] text-muted">
                              {timeAgo(c.updatedAt)}
                            </span>
                            <span className="rounded bg-bone-100 px-1 font-mono text-[9px] uppercase tracking-wide text-muted">
                              {c.tone === "technical" ? "expert" : "edu"}
                            </span>
                          </div>
                        </button>
                        {/* Delete — always visible (faint), reliable on touch/desktop */}
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            onDelete(c.id);
                          }}
                          className="pressable absolute right-1 top-1/2 z-20 -translate-y-1/2 rounded p-1.5 text-muted/50 transition-colors hover:bg-coral-soft/50 hover:text-coral-deep"
                          aria-label={`Delete conversation: ${c.title}`}
                          title="Delete"
                        >
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />
                          </svg>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <div className="border-t hairline px-4 py-3">
              <p className="text-[10.5px] leading-relaxed text-muted/70">
                History is stored in your browser only.
              </p>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}
