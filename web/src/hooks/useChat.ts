import { useCallback, useEffect, useRef, useState } from "react";
import { postChat, postReset } from "../lib/api";
import type { ChatMessage } from "../lib/types";

let _idc = 0;
const uid = () => `m${Date.now().toString(36)}${(_idc++).toString(36)}`;

export interface UseChat {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  sessionId: string | null;
  send: (text: string) => Promise<void>;
  reset: () => Promise<void>;
  setMessages: (m: ChatMessage[]) => void;
}

export function useChat(tone: string, initialMessages: ChatMessage[] = []): UseChat {
  const [messages, setMessagesState] = useState<ChatMessage[]>(initialMessages);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // When the caller passes a new set of initial messages (e.g. loading a past
  // chat from history), replace the current view. This lets us reuse one hook
  // instance across active-chat switches.
  useEffect(() => {
    setMessagesState(initialMessages);
    setSessionId(null); // server session resets when we load a different chat
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialMessages]);

  const setMessages = useCallback((m: ChatMessage[]) => {
    setMessagesState(m);
    setSessionId(null);
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;
      setError(null);
      const userMsg: ChatMessage = { id: uid(), role: "user", text: trimmed };
      setMessagesState((m) => [...m, userMsg]);
      setLoading(true);

      const ctl = new AbortController();
      abortRef.current = ctl;
      try {
        const res = await postChat(trimmed, sessionId, tone, ctl.signal);
        setSessionId(res.session_id);
        const asst: ChatMessage = {
          id: uid(),
          role: "assistant",
          text: res.answer || (res.error ? "" : ""),
          refused: res.refused,
          citations: res.citations,
          reasoning: res.reasoning,
          engine: res.engine,
          error: res.error ?? null,
          raw_user: trimmed,
        };
        setMessagesState((m) => [...m, asst]);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setError((e as Error).message || "Something went wrong");
        setMessagesState((m) => [
          ...m,
          {
            id: uid(),
            role: "assistant",
            text: "",
            refused: false,
            citations: [],
            reasoning: {},
            engine: "",
            error: (e as Error).message,
            raw_user: trimmed,
          },
        ]);
      } finally {
        setLoading(false);
        abortRef.current = null;
      }
    },
    [loading, sessionId, tone]
  );

  const reset = useCallback(async () => {
    if (sessionId) {
      try {
        await postReset(sessionId);
      } catch {
        /* ignore */
      }
    }
    setMessagesState([]);
    setSessionId(null);
    setError(null);
  }, [sessionId]);

  return { messages, loading, error, sessionId, send, reset, setMessages };
}
