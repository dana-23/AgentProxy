"""Tests for agent subgraphs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agentproxy.agent.email_agent import EmailAgent
from agentproxy.graph.state import Task

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="mocked response")
    return llm


@pytest.fixture
def email_agent(mock_llm):
    return EmailAgent(llm=mock_llm)


def _make_state(message: str = "Hello", goal: str = "Test goal") -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "task": {"agent": "email", "goal": goal},
        "result": {},
    }


# ── Unit tests ───────────────────────────────────────────────────────


class TestEmailAgent:
    def test_returns_result(self, email_agent, mock_llm):
        result = email_agent(state=_make_state())

        assert "result" in result
        assert result["result"]["summary"] == "mocked response"
        mock_llm.invoke.assert_called_once()

    def test_passes_messages_to_llm(self, email_agent, mock_llm):
        state = _make_state(message="Write a haiku about AI")
        email_agent(state=state)

        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].content == "Write a haiku about AI"

    def test_graph_compiles(self, email_agent):
        compiled = email_agent.compile()
        assert compiled is not None

    def test_different_llm_responses(self, mock_llm):
        mock_llm.invoke.return_value = AIMessage(content="a different response")
        agent = EmailAgent(llm=mock_llm)

        result = agent(state=_make_state())
        assert result["result"]["summary"] == "a different response"


# ── Integration tests (real API calls) ───────────────────────────────


@pytest.mark.integration
class TestEmailAgentLive:
    def test_llm_response(self):
        from agentproxy.config.settings import get_llm

        llm = get_llm()
        agent = EmailAgent(llm=llm)

        result = agent(
            state=_make_state(
                message="Write a haiku about AI",
                goal="Write a haiku",
            )
        )

        assert result["result"]["summary"]
        assert len(result["result"]["summary"]) > 10
