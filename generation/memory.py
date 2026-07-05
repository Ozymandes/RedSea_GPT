"""Conversation memory and history-aware query resolution for multiturn chat.

Design goals
------------
1. Let the user ask follow-ups with pronouns/implicit references
   ("how deep is *it*?", "and what about its salinity?") and have the system
   retrieve and answer correctly.

2. Do it the way production conversational-RAG systems do: before retrieval,
   rewrite the latest user turn into a *self-contained* question using a small
   LLM call. This is sometimes called "standalone question" or "query
   contextualization" (it is what ChatGPT/Claude do under the hood). Then
   retrieve against the rewritten question and pass the raw conversation to the
   generator so it can phrase the answer naturally ("As I mentioned...").

3. Degrade gracefully: if there is no history or the resolution LLM call fails,
   fall back to the raw question so the system never breaks because of memory.

The memory object is intentionally small and pickle/JSON-serializable so a
future server layer could persist it per-session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single conversational turn."""

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # The resolved (self-contained) question used for retrieval. Stored for
    # audit/debug so you can see how a follow-up was interpreted.
    resolved_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationMemory:
    """A bounded conversation buffer.

    Keeps the last ``max_turns`` turns (most-recent last). Bounded to keep the
    prompt size and retrieval resolution latency predictable.
    """

    max_turns: int = 6
    turns: List[Turn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        # Trim oldest turns beyond the window. We never trim below 0.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()

    @property
    def is_empty(self) -> bool:
        return len(self.turns) == 0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def format_for_prompt(self, max_turns: Optional[int] = None) -> str:
        """Render the conversation as a compact transcript for the generator.

        Only the question/answer are shown (sources are too long and the model
        has the freshly retrieved chunks for the current turn). Returns an empty
        string when there is nothing to show.
        """
        turns = self.turns
        if max_turns is not None:
            turns = turns[-max_turns:]
        if not turns:
            return ""
        lines = ["=== PREVIOUS CONVERSATION (use for context; cite only freshly retrieved sources) ==="]
        for i, t in enumerate(turns, 1):
            # Bound answer length per turn, but generously enough to preserve real
            # conversational context (the previous answer's specific claims / numbers
            # a follow-up may reference). ~800 chars ~ 2-3 substantive sentences.
            ans = t.answer.strip().replace("\n", " ")
            if len(ans) > 800:
                ans = ans[:800].rstrip() + "…"
            lines.append(f"Turn {i}")
            lines.append(f"  User: {t.question}")
            lines.append(f"  Assistant: {ans}")
        lines.append("=== END OF PREVIOUS CONVERSATION ===\n")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"max_turns": self.max_turns, "turns": [t.to_dict() for t in self.turns]}


# ---------------------------------------------------------------------------
# History-aware query resolution (query contextualization)
# ---------------------------------------------------------------------------

_RESOLVE_PROMPT = """You are rewriting a user's latest message into a self-contained question.

The user is chatting with a scientific assistant about the Egyptian Red Sea. Their latest message may
use pronouns ("it", "they", "this"), implicit references, or build on an earlier topic. Rewrite it into
ONE clear, specific, standalone question that could be answered without seeing the conversation history.

Rules:
- Preserve the user's intent. Do NOT introduce new topics or answer the question yourself.
- Resolve pronouns and implicit references using the prior turns (e.g. "how deep is it?" ->
  "How deep is the Gulf of Aqaba?").
- If the latest message is already self-contained, return it unchanged.
- Output ONLY the rewritten question on a single line. No preamble, no quotes, no explanation.
- If the latest message is small talk, a greeting, or not a question about the Red Sea, output it
  essentially unchanged so the assistant can refuse or respond.

Conversation so far:
{history}

Latest user message:
{question}

Standalone question:"""


def resolve_query_with_history(
    llm,
    question: str,
    memory: ConversationMemory,
    max_history_turns: int = 3,
) -> str:
    """Rewrite ``question`` into a self-contained query using recent history.

    Returns the original question unchanged when there is no history to use or
    if the LLM call fails. Never raises.
    """
    # Fast path: nothing to resolve against.
    if memory.is_empty:
        return question
    try:
        history = memory.format_for_prompt(max_turns=max_history_turns)
        if not history.strip():
            return question
        prompt = _RESOLVE_PROMPT.format(history=history, question=question)
        raw = llm.invoke(prompt)
        # LangChain BaseLLM.invoke returns a string for these providers; be defensive.
        resolved = (raw.content if hasattr(raw, "content") else str(raw)).strip()
        # Strip surrounding quotes the model sometimes adds.
        resolved = resolved.strip("`\"' \n")
        # Reject obviously broken outputs and fall back.
        if not resolved or len(resolved) > 4 * len(question) + 200:
            logger.warning("Query resolution returned suspicious output; falling back to raw question.")
            return question
        return resolved
    except Exception as exc:  # noqa: BLE001 - never let memory break the pipeline
        logger.warning("Query resolution failed (%s); falling back to raw question.", exc)
        return question
