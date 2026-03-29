"""EmailAgent — email triage agent with Gmail read tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.base import BaseAgent
from config.settings import BASE_DIR
from graph.state import AgentState

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_PATH = BASE_DIR / "token.json"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"


def _get_gmail_service():
    """Authenticate and return a Gmail API service instance."""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


@tool
def fetch_emails(max_results: int = 10) -> list[dict[str, str]]:
    """Fetch recent emails from Gmail.

    Returns a list of emails, each with 'from', 'subject', and 'snippet' fields.

    Args:
        max_results: Number of emails to fetch (default 10, max 50).
    """
    max_results = min(max_results, 50)
    service = _get_gmail_service()

    results = service.users().messages().list(
        userId="me", maxResults=max_results
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        return []

    emails = []
    for message in messages:
        msg = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="metadata",
            metadataHeaders=["Subject", "From"],
        ).execute()

        headers = msg.get("payload", {}).get("headers", [])
        subject = "No Subject"
        sender = "Unknown Sender"

        for header in headers:
            if header["name"] == "Subject":
                subject = header["value"]
            if header["name"] == "From":
                sender = header["value"]

        emails.append({
            "from": sender,
            "subject": subject,
            "snippet": msg.get("snippet", ""),
        })

    return emails


TOOLS = [fetch_emails]


class EmailAgent(BaseAgent):
    """
    START -> LLM -> SHOULD_USE_TOOL -> END
              |          |
              |<- TOOL <-|
    """
    name: str = "email"

    def build_graph(self) -> StateGraph:
        llm_with_tools = self.llm.bind_tools(TOOLS)

        system = SystemMessage(content=(
            "You are an email assistant. When given email data, provide a concise "
            "summary of each email in 1-2 sentences — describe what it's about, not "
            "just repeat the raw fields. Categorize each as: ACTION NEEDED, FYI, or IGNORE."
        ))

        def call_llm(state: AgentState) -> dict:
            messages = [system] + list(state["messages"])
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_use_tool(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "done"

        graph = StateGraph(AgentState)

        graph.add_node("call_llm", call_llm)
        graph.add_node("tools", ToolNode(TOOLS))

        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges("call_llm", should_use_tool, {
            "tools": "tools",
            "done": END,
        })
        graph.add_edge("tools", "call_llm")

        return graph

if __name__ == "__main__":
    from config.settings import get_llm

    llm = get_llm()
    email_agent = EmailAgent(llm=llm)
    result = email_agent(state={"messages": "Summarize my last received email."})

    # Print only the final AI response
    last_msg = result["messages"][-1]
    content = last_msg.content
    if isinstance(content, list):
        text = next((block["text"] for block in content if block.get("type") == "text"), "")
    else:
        text = content
    print(text)