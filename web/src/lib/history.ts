// Persistence layer for chat history, backed by localStorage.
//
// Designed for a demo: per-browser, no backend, no auth. Each chat is stored
// under REDSEA_CHATS as a record. A separate REDSEA_ACTIVE_ID tracks the
// currently-open chat so a refresh restores the right conversation.
//
// Why localStorage (not IndexedDB): the payload is small (text messages +
// citations, no media), access is synchronous which keeps the UI simple, and
// it survives across tabs of the same origin. For a production app you'd move
// this to a backend; for a portfolio demo this is the right weight.

import type { ChatMessage } from "./types";

const CHATS_KEY = "redsea_chats_v1";
const ACTIVE_KEY = "redsea_active_chat_v1";
const MAX_CHATS = 30; // bound it so localStorage never grows unbounded

export interface StoredChat {
  id: string;
  title: string; // derived from the first user message
  createdAt: number; // epoch ms
  updatedAt: number; // epoch ms
  tone: string; // the tone the chat was started in
  messages: ChatMessage[];
}

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function loadChats(): StoredChat[] {
  if (typeof window === "undefined") return [];
  const all = safeParse<StoredChat[]>(localStorage.getItem(CHATS_KEY), []);
  // newest first; drop any malformed entries
  return all
    .filter((c) => c && typeof c.id === "string" && Array.isArray(c.messages))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export function saveChats(chats: StoredChat[]): void {
  if (typeof window === "undefined") return;
  try {
    // bound: keep only the N most recent
    const bounded = [...chats].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS);
    localStorage.setItem(CHATS_KEY, JSON.stringify(bounded));
  } catch {
    // quota exceeded or storage disabled — fail silently; the demo still works
    // in-session, history just won't persist.
  }
}

export function upsertChat(chat: StoredChat): StoredChat[] {
  const chats = loadChats();
  const idx = chats.findIndex((c) => c.id === chat.id);
  const updated = { ...chat, updatedAt: Date.now() };
  if (idx >= 0) {
    chats[idx] = updated;
  } else {
    chats.unshift(updated);
  }
  saveChats(chats);
  return loadChats();
}

export function deleteChat(id: string): StoredChat[] {
  const chats = loadChats().filter((c) => c.id !== id);
  saveChats(chats);
  return chats;
}

export function getActiveId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveId(id: string | null): void {
  if (typeof window === "undefined") return;
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

// Derive a short human title from the first user message.
export function deriveTitle(firstMessage: string): string {
  const clean = firstMessage.replace(/\s+/g, " ").trim();
  if (clean.length <= 48) return clean;
  // try to cut at a word boundary near 48 chars
  const slice = clean.slice(0, 48);
  const lastSpace = slice.lastIndexOf(" ");
  return (lastSpace > 20 ? slice.slice(0, lastSpace) : slice) + "…";
}

export function newChatId(): string {
  return `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}
