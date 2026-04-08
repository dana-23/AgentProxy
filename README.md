# AgentProxy

An AI-powered agent that automates daily tasks, acting as a personal proxy to handle routine workflows so you can focus on what matters.

## Features

- **Task Automation** — Automate repetitive daily tasks like email triage, calendar management, and data entry
- **Intelligent Scheduling** — Prioritize and schedule tasks based on deadlines and importance
- **Multi-Service Integration** — Connect with APIs and services you already use
- **Natural Language Interface** — Describe tasks in plain English and let the agent handle execution
- **Extensible Plugin System** — Add custom task handlers for domain-specific workflows

## Getting Started

### Prerequisites

- Python 3.12+
- An API key for your chosen LLM provider

### Installation

```bash
git clone https://github.com/yourusername/AgentProxy.git
cd AgentProxy
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

### Usage

```bash
# Interactive terminal chat
agentproxy chat

# Use a specific provider/model
agentproxy chat --provider anthropic --model claude-sonnet-4

# Start the Discord bot
agentproxy discord

# Show current config
agentproxy config
```