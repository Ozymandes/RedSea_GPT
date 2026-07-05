// Types mirroring the FastAPI response contract (api/main.py).

export interface Citation {
  citation_id: number;
  source: string;
  page: number | string | null;
  content: string;
}

export interface ReasoningTrace {
  route?: string | null;
  retrieval_method?: string | null;
  retrieval_rounds?: number | null;
  confidence?: number | null;
  resolved_question?: string | null;
  verification?: Record<string, unknown> | null;
  node_trace?: Record<string, number> | null;
  reason?: string | null;
}

export interface ChatResponse {
  answer: string;
  refused: boolean;
  citations: Citation[];
  reasoning: ReasoningTrace;
  session_id: string;
  engine: string;
  tone: string;
  error?: string | null;
}

export type ChatMessage =
  | {
      id: string;
      role: "user";
      text: string; // what the user typed
    }
  | {
      id: string;
      role: "assistant";
      text: string;
      refused: boolean;
      citations: Citation[];
      reasoning: ReasoningTrace;
      engine: string;
      error?: string | null;
      raw_user: string; // the user message this answers (for "you asked X -> understood Y")
    };

export interface Suggestion {
  label: string;
  prompt: string;
}
