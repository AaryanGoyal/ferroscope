# ferroscope

An HTTP proxy that intercepts LLM API calls and runs loop-pattern classifiers across the call history.

---

## Why

LangSmith and Helicone record individual traces. They don't run logic across sequential calls. That's where most agent waste lives — a retry loop, a model escalation, an oscillating tool call — none of it is visible in a single span. Ferroscope sits in the call path and checks for those patterns after every call.

---

## Quickstart

```bash
cargo install --path .

# run the demo to see all 4 classifiers fire
ferroscope run --tui python demo/loop_agent.py

# wrap any agent — zero code changes
ferroscope run python my_agent.py

# or set base URLs manually
export ANTHROPIC_BASE_URL=http://localhost:8080
export OPENAI_BASE_URL=http://localhost:8080
python my_agent.py
```

Open the live TUI without wrapping a subprocess:

```bash
ferroscope --tui
```

### Classifiers

- **retry_storm** — three or more near-identical prompts (≥ 85% similarity) within 60 seconds
- **cost_inflation** — same prompt re-sent with an escalated model tier
- **self_correction** — correction phrase in a response followed by the same prompt re-submitted
- **ping_pong** — output fingerprints alternating in an A-B-A-B pattern across consecutive calls

### Claude Desktop

Build the MCP server:

```bash
cd mcp && npm install && npm run build
```

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ferroscope": {
      "command": "node",
      "args": ["/path/to/ferroscope/mcp/dist/index.js"],
      "env": {
        "FERROSCOPE_DB": "/path/to/ferroscope/ferroscope.db"
      }
    }
  }
}
```

---

## Architecture

```
agent → ferroscope (:8080) → anthropic / openai
              ↓
           sqlite
              ↓
        claude desktop (mcp)
```

---

## Configuration

```
--addr          listen address          default: 0.0.0.0:8080
--db            sqlite database path    default: ./ferroscope.db
--tui           enable terminal UI
--anthropic-upstream    upstream Anthropic URL    default: https://api.anthropic.com/v1/messages
--openai-upstream       upstream OpenAI URL       default: https://api.openai.com/v1/chat/completions
```

Environment variables injected by `ferroscope run`:

```
ANTHROPIC_BASE_URL    set to http://127.0.0.1:<port>
OPENAI_BASE_URL       set to http://127.0.0.1:<port>
```

MCP server:

```
FERROSCOPE_DB    path to sqlite database    default: ./ferroscope.db
```
