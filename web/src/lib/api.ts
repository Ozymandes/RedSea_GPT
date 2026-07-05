import type { ChatResponse } from "./types";

const API_BASE = "/api";

export async function postChat(
  message: string,
  sessionId: string | null,
  signal?: AbortSignal
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as ChatResponse;
}

export async function postReset(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function getHealth(signal?: AbortSignal): Promise<{
  ok: boolean;
  engine: string;
  engine_loaded: boolean;
}> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
