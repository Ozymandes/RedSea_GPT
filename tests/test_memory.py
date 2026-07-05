"""Tests for conversation memory and history-aware query resolution.

Covers: bounded buffer trimming, prompt rendering, query contextualization
(pronoun resolution), and the graceful fallback when the resolution LLM fails
or there is no history.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation.memory import ConversationMemory, Turn, resolve_query_with_history


# ---------------------------------------------------------------------------
# A tiny fake LLM so we can test resolve_query_with_history deterministically
# without hitting the network.
# ---------------------------------------------------------------------------
class _FakeLLM:
    def __init__(self, response: str, raise_exc=None):
        self._response = response
        self._raise = raise_exc
        self.calls = 0

    def invoke(self, prompt):  # noqa: D401 - mimics LangChain BaseLLM.invoke
        self.calls += 1
        if self._raise:
            raise self._raise
        return self._response


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------
def test_memory_starts_empty():
    mem = ConversationMemory()
    assert mem.is_empty
    assert mem.num_turns == 0
    assert mem.format_for_prompt() == ""


def test_memory_add_and_format():
    mem = ConversationMemory()
    mem.add(Turn(question="How did the Red Sea form?", answer="Through rifting..."))
    mem.add(Turn(question="How fast is that?", answer="About 10-15 mm/yr."))
    assert not mem.is_empty
    assert mem.num_turns == 2
    out = mem.format_for_prompt()
    assert "PREVIOUS CONVERSATION" in out
    assert "How did the Red Sea form?" in out
    assert "How fast is that?" in out
    # Both turns rendered
    assert out.count("Turn ") == 2


def test_memory_trims_to_max_turns():
    mem = ConversationMemory(max_turns=3)
    for i in range(6):
        mem.add(Turn(question=f"Q{i}", answer=f"A{i}"))
    assert mem.num_turns == 3
    # Oldest are dropped, most recent kept
    assert mem.turns[0].question == "Q3"
    assert mem.turns[-1].question == "Q5"


def test_memory_format_truncates_long_answers():
    mem = ConversationMemory()
    long_ans = "word " * 200  # ~1000 chars
    mem.add(Turn(question="q", answer=long_ans))
    out = mem.format_for_prompt()
    assert "…" in out
    # Truncated well under the original length
    assert len(out) < len(long_ans)


def test_memory_format_respects_max_turns_arg():
    mem = ConversationMemory()
    for i in range(5):
        mem.add(Turn(question=f"Q{i}", answer=f"A{i}"))
    out = mem.format_for_prompt(max_turns=2)
    assert "Q3" in out and "Q4" in out
    assert "Q0" not in out and "Q1" not in out


def test_memory_clear():
    mem = ConversationMemory()
    mem.add(Turn(question="q", answer="a"))
    mem.clear()
    assert mem.is_empty
    assert mem.format_for_prompt() == ""


def test_memory_rejects_invalid_max_turns():
    try:
        ConversationMemory(max_turns=0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_memory_serializable():
    mem = ConversationMemory()
    mem.add(Turn(question="q", answer="a", resolved_question="resolved"))
    d = mem.to_dict()
    assert d["max_turns"] == mem.max_turns
    assert d["turns"][0]["resolved_question"] == "resolved"


# ---------------------------------------------------------------------------
# resolve_query_with_history
# ---------------------------------------------------------------------------
def test_resolve_no_history_returns_question_unchanged():
    llm = _FakeLLM("should not be called")
    mem = ConversationMemory()
    out = resolve_query_with_history(llm, "How deep is the Gulf of Aqaba?", mem)
    assert out == "How deep is the Gulf of Aqaba?"
    assert llm.calls == 0  # fast path skips the LLM entirely


def test_resolve_resolves_pronoun():
    llm = _FakeLLM("How fast is the seafloor spreading in the Red Sea?")
    mem = ConversationMemory()
    mem.add(Turn(question="How did the Red Sea form?", answer="Via seafloor spreading and rifting."))
    out = resolve_query_with_history(llm, "how fast is that happening?", mem)
    assert out == "How fast is the seafloor spreading in the Red Sea?"
    assert llm.calls == 1


def test_resolve_strips_surrounding_quotes():
    llm = _FakeLLM('  "How deep is the Gulf of Aqaba?"  \n')
    mem = ConversationMemory()
    mem.add(Turn(question="Tell me about the Gulf of Aqaba.", answer="It is a deep narrow gulf."))
    out = resolve_query_with_history(llm, "how deep is it?", mem)
    assert out == "How deep is the Gulf of Aqaba?"


def test_resolve_falls_back_on_llm_exception():
    llm = _FakeLLM("", raise_exc=RuntimeError("provider down"))
    mem = ConversationMemory()
    mem.add(Turn(question="How did the Red Sea form?", answer="Via rifting."))
    # Must NOT raise; falls back to the raw question so the pipeline never breaks.
    out = resolve_query_with_history(llm, "how fast is that?", mem)
    assert out == "how fast is that?"


def test_resolve_falls_back_on_empty_output():
    llm = _FakeLLM("   \n  ")  # empty after strip
    mem = ConversationMemory()
    mem.add(Turn(question="q", answer="a"))
    out = resolve_query_with_history(llm, "follow up", mem)
    assert out == "follow up"


def test_resolve_falls_back_on_absurdly_long_output():
    llm = _FakeLLM("x " * 5000)  # way longer than a reasonable question
    mem = ConversationMemory()
    mem.add(Turn(question="q", answer="a"))
    out = resolve_query_with_history(llm, "follow up", mem)
    assert out == "follow up"


def test_resolve_handles_llm_returning_object_with_content():
    """Some LangChain invoke() paths return an object with .content, not a str."""
    class _Obj:
        def __init__(self, c):
            self.content = c
    llm = _FakeLLM(_Obj("How deep is the Gulf of Aqaba?"))
    mem = ConversationMemory()
    mem.add(Turn(question="Tell me about the Gulf of Aqaba.", answer="Deep narrow gulf."))
    out = resolve_query_with_history(llm, "how deep is it?", mem)
    assert out == "How deep is the Gulf of Aqaba?"
