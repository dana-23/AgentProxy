"""Tests all agents before deployment"""

from __future__ import annotations

import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.email import EmailAgent
from config.settings import get_llm
from graph.state import AgentState

if __name__ == "__main__":
    llm = get_llm()
    state = AgentState(messages="Write a haiku on being a LLM.")

    email = EmailAgent(llm=llm)
    email.compile()

    result = email(state=state)
    print(result)