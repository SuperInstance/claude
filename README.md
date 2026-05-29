# claude — OpenClaw Agent Workspace

**The Claude agent workspace for the Cocapn Fleet. An OpenClaw-configured Claude agent with SDK utilities for request building, conversation management, tool definitions, and cost tracking.**

## What This Gives You

- **ClaudeClient** — constructs API request payloads for the Claude Messages API (no HTTP calls — build and inspect before sending)
- **Conversation** — manages message history with context window awareness
- **Tool definitions** — structured tool/function definitions for Claude
- **Cost tracking** — track token usage and estimated costs across conversations
- **Message types** — typed content blocks (text, image, tool_use, tool_result)

## Quick Start

```python
from claude_sdk import ClaudeClient, Conversation, Message, ToolDef, CostTracker

# Build a client
client = ClaudeClient(
    api_key="your-key",
    model="claude-3.5-sonnet",
    max_tokens=4096,
)

# Create a conversation
conv = Conversation()
conv.add(Message(role="user", content="Explain the fleet architecture"))

# Build the request (no HTTP call)
request = client.build_request(conv)
print(request["model"])  # "claude-3.5-sonnet"
print(request["messages"])  # [{role: "user", content: "Explain..."}]

# Define tools
tools = [
    ToolDef(name="deploy", description="Deploy a service", input_schema={...}),
]
client.tools = tools

# Track costs
tracker = CostTracker()
tracker.record(model="claude-3.5-sonnet", input_tokens=100, output_tokens=200)
print(tracker.total_cost())
```

## API Reference

### `ClaudeClient(api_key, model, max_tokens, temperature, system, tools, cost_tracker)`
### `Conversation` — `add(message)`, `to_api_messages()`, `truncate(max_tokens)`
### `Message(role, content)` — `ContentBlock` types: text, image, tool_use, tool_result
### `ToolDef(name, description, input_schema)`
### `CostTracker` — `record(model, input_tokens, output_tokens)`, `total_cost()`

## How It Fits

The OpenClaw workspace for a Claude-powered agent in the [SuperInstance fleet](https://github.com/SuperInstance). Uses the Claude SDK for structured API interactions.

- **[cocapn-sdk](https://github.com/SuperInstance/cocapn-sdk)** — Multi-model SDK
- **[cocapn-py](https://github.com/SuperInstance/cocapn-py)** — Python SDK
- **[claude-code-vessel](https://github.com/SuperInstance/claude-code-vessel)** — Containerized Claude execution

## Testing

```bash
pytest tests/
```

MIT license.
