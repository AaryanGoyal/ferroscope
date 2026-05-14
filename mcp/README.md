# ferroscope MCP server

A TypeScript MCP server that reads from `ferroscope.db` and exposes LLM observability data to Claude Desktop.

## Tools

| Tool | Description |
|------|-------------|
| `get_recent_calls(n)` | Last N LLM calls (model, tokens, latency, cost, classifier tag) |
| `get_detections(since?)` | Classifier detections, optionally filtered by ISO timestamp |
| `get_session_summary()` | Total calls, cost, avg latency, detections by classifier, most expensive call |
| `explain_detection(detection_id)` | Full detection record + all implicated call rows |

## Build

```bash
cd mcp
npm install
npm run build
```

## Claude Desktop integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ferroscope": {
      "command": "node",
      "args": ["/absolute/path/to/ferroscope/mcp/dist/index.js"],
      "env": {
        "FERROSCOPE_DB": "/absolute/path/to/ferroscope/ferroscope.db"
      }
    }
  }
}
```

Replace `/absolute/path/to/ferroscope` with your actual repo path. Restart Claude Desktop after editing.

## Usage

Run ferroscope in one terminal:

```bash
ferroscope --tui
# or with a child process:
ferroscope run -- python3 my_agent.py
```

Then ask Claude Desktop questions like:
- "What LLM calls has ferroscope logged?"
- "Are there any cost inflation detections?"
- "Explain detection 3 — which calls triggered it?"
