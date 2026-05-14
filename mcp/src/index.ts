import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { DatabaseSync } from "node:sqlite";
import path from "node:path";

const DB_PATH = process.env.FERROSCOPE_DB ?? path.join(process.cwd(), "ferroscope.db");
const db = new DatabaseSync(DB_PATH, { open: true });

// ── helpers ────────────────────────────────────────────────────────────────

function getRecentCalls(n: number): unknown[] {
  return db
    .prepare(`SELECT * FROM calls ORDER BY id DESC LIMIT ?`)
    .all(n);
}

function getDetections(since?: string): unknown[] {
  if (since) {
    return db
      .prepare(
        `SELECT id, timestamp, classifier, call_ids, detail, suggested_fix, cost_usd
         FROM detections
         WHERE timestamp >= ?
         ORDER BY id DESC`
      )
      .all(since);
  }
  return db
    .prepare(
      `SELECT id, timestamp, classifier, call_ids, detail, suggested_fix, cost_usd
       FROM detections
       ORDER BY id DESC`
    )
    .all();
}

function getSessionSummary(): object {
  const totals = db
    .prepare(
      `SELECT COUNT(*) as total_calls,
              COALESCE(SUM(cost_usd), 0) as total_cost_usd,
              COALESCE(AVG(latency_ms), 0) as avg_latency_ms
       FROM calls`
    )
    .get() as { total_calls: number; total_cost_usd: number; avg_latency_ms: number };

  const detectionsByClassifier = db
    .prepare(
      `SELECT classifier, COUNT(*) as count
       FROM detections
       GROUP BY classifier
       ORDER BY count DESC`
    )
    .all() as { classifier: string; count: number }[];

  const mostExpensive = db
    .prepare(
      `SELECT id, timestamp, model, prompt_tokens, output_tokens, cost_usd
       FROM calls
       ORDER BY cost_usd DESC
       LIMIT 1`
    )
    .get() as object | undefined;

  return {
    total_calls: totals.total_calls,
    total_cost_usd: Number(totals.total_cost_usd.toFixed(6)),
    avg_latency_ms: Number(totals.avg_latency_ms.toFixed(1)),
    detections_by_classifier: detectionsByClassifier,
    most_expensive_call: mostExpensive ?? null,
  };
}

function explainDetection(detectionId: number): object | null {
  const detection = db
    .prepare(
      `SELECT id, timestamp, classifier, call_ids, detail, suggested_fix, cost_usd
       FROM detections WHERE id = ?`
    )
    .get(detectionId) as { id: number; call_ids: string; [k: string]: unknown } | undefined;

  if (!detection) return null;

  const ids: number[] = String(detection.call_ids)
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !isNaN(n));

  const calls =
    ids.length > 0
      ? db
          .prepare(
            `SELECT * FROM calls
             WHERE id IN (${ids.map(() => "?").join(",")})
             ORDER BY id ASC`
          )
          .all(...ids)
      : [];

  return { detection, implicated_calls: calls };
}

// ── tool definitions ───────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: "get_recent_calls",
    description:
      "Return the last N LLM calls logged by ferroscope, newest first. " +
      "Each row includes model, token counts, latency, cost, and any classifier tag.",
    inputSchema: {
      type: "object",
      properties: {
        n: {
          type: "number",
          description: "Number of calls to return (default 20, max 200)",
        },
      },
    },
  },
  {
    name: "get_detections",
    description:
      "Return ferroscope classifier detections (retry_storm, cost_inflation, " +
      "self_correction, ping_pong). Optionally filter by ISO 8601 timestamp.",
    inputSchema: {
      type: "object",
      properties: {
        since: {
          type: "string",
          description: "ISO 8601 timestamp — return only detections after this time",
        },
      },
    },
  },
  {
    name: "get_session_summary",
    description:
      "Return aggregate statistics for the entire ferroscope session: total calls, " +
      "total cost, average latency, detections by classifier, and the most expensive call.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "explain_detection",
    description:
      "Return full details for a single detection (by id) including the detection " +
      "record and all implicated call rows.",
    inputSchema: {
      type: "object",
      required: ["detection_id"],
      properties: {
        detection_id: {
          type: "number",
          description: "The integer id of the detection to explain",
        },
      },
    },
  },
];

// ── server ─────────────────────────────────────────────────────────────────

const server = new Server(
  { name: "ferroscope", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;

  try {
    let result: unknown;

    if (name === "get_recent_calls") {
      const n = Math.min(Math.max(1, Number(a.n ?? 20)), 200);
      result = getRecentCalls(n);
    } else if (name === "get_detections") {
      result = getDetections(typeof a.since === "string" ? a.since : undefined);
    } else if (name === "get_session_summary") {
      result = getSessionSummary();
    } else if (name === "explain_detection") {
      const id = Number(a.detection_id);
      if (!Number.isInteger(id) || id <= 0) {
        return {
          content: [{ type: "text", text: "detection_id must be a positive integer" }],
          isError: true,
        };
      }
      result = explainDetection(id);
      if (result === null) {
        return {
          content: [{ type: "text", text: `No detection found with id ${id}` }],
          isError: true,
        };
      }
    } else {
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }

    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { content: [{ type: "text", text: `Error: ${msg}` }], isError: true };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write(`ferroscope MCP server running — db: ${DB_PATH}\n`);
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err}\n`);
  process.exit(1);
});
