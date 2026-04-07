""" A Finance Agent - Help me with daily/weekly/monthly expenses."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, List
import yaml

import aiofiles

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from agent.base import BaseAgent
from config.settings import BASE_DIR
from graph.state import AgentState
from graph.state import Task

prompts_path = Path(__file__).parent.parent / Path("prompts.yaml")

LOGS_DIR = str(BASE_DIR / "data" / "finance" / "logs")

class ExpenseEntry(BaseModel):
    date: str = Field(
        ...,
        description="Date of the expense in YYYY-MM-DD format. e.g. '2026-03-15'"
    )
    vendor: str = Field(
        ...,
        description="Vendor or store name. e.g. 'Woolies', 'Shell'"
    )
    category: str = Field(
        ...,
        description="Expense category. e.g. 'Groceries', 'Transport'"
    )
    amount: float = Field(
        ...,
        description="Amount spent as a number. e.g. 45.50"
    )

@tool(args_schema=ExpenseEntry)
async def make_entry(date: str, vendor: str, category: str, amount: float) -> str:
    """
    Adds a new expense entry to the log file for that month.
    Creates the file with frontmatter and table header if it doesn't exist.

    Returns:
        A confirmation message with the entry that was added.
    """
    # Derive period and filename from the date
    period = date[:7]  # "2026-03-15" -> "2026-03"
    month, year = period.split("-")[1], period.split("-")[0]
    filename = f"{month}-{year}.md"
    filepath = os.path.join(LOGS_DIR, filename)

    logger.info("Adding expense: %s | %s | %s | %.2f to %s", date, vendor, category, amount, filepath)

    if not os.path.exists(filepath):
        logger.info("Creating new expense log: %s", filepath)
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(
                f"---\ntype: expense-log\nperiod: \"{period}\"\nstatus: active\n---\n"
                f"# {period} Expenses\n"
                f"| Date          | Vendor    | Category  | Amount |\n"
                f"| :---          | :---      | :---      | :---   |\n"
            )

    # Append the new row
    row = f"| {date}    | {vendor}   | {category} | {amount:.2f}  |\n"
    async with aiofiles.open(filepath, 'a') as f:
        await f.write(row)

    logger.info("Entry added successfully")
    return f"Added: {date} | {vendor} | {category} | ${amount:.2f}"


@tool
async def find_file_by_period(directory: str, target_period: str) -> str | None:
    """
    Scans markdown file frontmatter in a directory to find the expense log matching a given period.

    Args:
        directory: Absolute path to the folder containing expense-log markdown files.
                   e.g. "/Users/.../data/finance/logs"
        target_period: The period string to match in YYYY-MM format. e.g. "2026-03" for March 2026.

    Returns:
        The filepath of the matching markdown file, or None if no file matches.
    """
    logger.info("Searching for period=%s in %s", target_period, directory)
    for filename in os.listdir(directory):
        if not filename.endswith(".md"): continue

        filepath = os.path.join(directory, filename)
        async with aiofiles.open(filepath, 'r') as file:
            content = await file.read()

            # Extract text between the first two '---' markers
            frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
            if frontmatter_match:
                frontmatter = frontmatter_match.group(1)
                # Look for the period key in the frontmatter
                if f'period: "{target_period}"' in frontmatter:
                    logger.info("Found match: %s", filepath)
                    return filepath # Match found, stop searching!
    logger.warning("No file found for period=%s", target_period)
    return None

@tool
async def get_expenses(filepath: str) -> list[dict[str, str]]:
    """
    Reads a markdown expense-log file and returns all expense rows.

    Args:
        filepath: Absolute path to the expense-log markdown file (returned by find_file_by_period).
                  e.g. "/Users/.../data/finance/logs/03-2026.md"

    Returns:
        A list of dicts, each with keys: date, vendor, category, amount.
    """
    logger.info("Reading expenses from %s", filepath)
    rows = []
    async with aiofiles.open(filepath, 'r') as file:
        lines = await file.readlines()

        for line in lines:
            if line.strip().startswith('|'):
                if '---' in line:
                    continue

                columns = [col.strip() for col in line.split('|')[1:-1]]

                if len(columns) >= 4 and columns[0] != 'Date':
                    rows.append({
                        "date": columns[0],
                        "vendor": columns[1],
                        "category": columns[2],
                        "amount": columns[3],
                    })
    logger.info("Found %d expense rows", len(rows))
    return rows

@tool
def get_today() -> str:
    """
    Returns today's date in YYYY-MM-DD format. Call this when the user says
    "today", "yesterday", "this week", or doesn't specify a date.

    Returns:
        Today's date as a string, e.g. "2026-04-07"
    """
    from datetime import date
    today = date.today().isoformat()
    logger.info("Today's date: %s", today)
    return today

@tool
def calculate_total(amounts: list[float]) -> float:
    """
    Sums a list of amounts. Use this after selecting the relevant rows from get_expenses.

    Args:
        amounts: List of numeric amounts to sum. e.g. [45.50, 12.00, 78.00]

    Returns:
        The total sum as a float.
    """
    total = sum(amounts)
    logger.info("Calculated total: %.2f from %d amounts", total, len(amounts))
    return total

TOOLS: list = [find_file_by_period, get_expenses, get_today, calculate_total, make_entry]

class FinanceAgent(BaseAgent):

    name: str = "finance"

    def build_graph(self) -> StateGraph:
        logger.info("Building FinanceAgent graph")
        llm_with_tools = self.llm.bind_tools(TOOLS)

        with open(prompts_path, 'r') as file:
            data = yaml.safe_load(file)

        prompt_template = data.get("agents", {}).get(f"{self.name}", {}).get('system')
        system = SystemMessage(
            content=prompt_template.format(data_dir=str(BASE_DIR / "data"))
        )

        async def call_llm(state: AgentState) -> dict:
            logger.debug("Calling LLM with %d messages", len(state["messages"]))
            messages = [system] + list(state["messages"])
            response = await llm_with_tools.ainvoke(messages)
            logger.debug("LLM responded (tool_calls=%s)", bool(getattr(response, "tool_calls", None)))
            return {"messages": [response]}

        def should_use_tool(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                logger.info("Routing to tools: %s", [tc["name"] for tc in last.tool_calls])
                return "tools"
            return "done"

        graph = StateGraph(AgentState)

        graph.add_node("call_llm", call_llm)
        graph.add_node("tools", ToolNode(TOOLS))

        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges(
            "call_llm", should_use_tool, {
                "tools": "tools",
                "done": END
                }
            )
        graph.add_edge("tools", "call_llm")

        return graph


if __name__ == "__main__":
    import asyncio
    from config.settings import get_llm
    from config.logger import setup_logging
    setup_logging()

    async def main():
        finance_agent = FinanceAgent(llm=get_llm())
        result = await finance_agent(
            state={
                "messages": "I had coffee for $5.5 today from starbucks."
            }
        )

        # Print only the final AI response
        last_msg = result["messages"][-1]
        content = last_msg.content
        if isinstance(content, list):
            text = next((block["text"] for block in content if block.get("type") == "text"), "")
        else:
            text = content
        print(text)
    
    asyncio.run(main())
    