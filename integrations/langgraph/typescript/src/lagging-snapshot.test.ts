/**
 * The checkpoint read by getStateAndMessagesSnapshots lags the stream, so a
 * mid-run MESSAGES_SNAPSHOT can be strictly *behind* what was already
 * dispatched — carrying nothing new while omitting messages currently being
 * streamed. Dispatching it makes strict-replace clients delete in-flight
 * messages. See CopilotKit/CopilotKit#6301.
 */

import { describe, it, expect, vi } from "vitest";
import { EventType } from "@ag-ui/core";
import { LangGraphAgent } from "./agent";
import type { LangGraphAgentConfig } from "./agent";
import type { Message as LangGraphMessage } from "@langchain/langgraph-sdk";

function msg(id: string, role: "human" | "ai", content: string): LangGraphMessage {
  return { id, type: role === "human" ? "human" : "ai", content } as LangGraphMessage;
}

function makeConfig(checkpointMessages: LangGraphMessage[]): LangGraphAgentConfig {
  return {
    deploymentUrl: "http://localhost:2024",
    graphId: "test-graph",
    client: {
      threads: {
        getState: vi.fn().mockResolvedValue({
          values: { messages: checkpointMessages },
          tasks: [],
          next: [],
          metadata: {},
        }),
      },
      runs: { cancel: vi.fn() },
      assistants: {
        search: vi.fn().mockResolvedValue([]),
        getGraph: vi.fn().mockResolvedValue({ nodes: [], edges: [] }),
      },
    } as any,
  };
}

/**
 * Agent wired so the real dispatchEvent runs (that is where dispatched
 * message ids are tracked) while the test still observes every event.
 */
function makeAgent(checkpointMessages: LangGraphMessage[]) {
  const agent = new LangGraphAgent(makeConfig(checkpointMessages));
  const dispatched: any[] = [];
  (agent as any).subscriber = { next: (e: any) => dispatched.push(e) };
  (agent as any).activeRun = { id: "run-1" };
  (agent as any).getStateSnapshot = vi.fn().mockReturnValue({});
  (agent as any).knownMessageIds = new Set(["u1"]);
  return { agent, dispatched };
}

const snapshots = (dispatched: any[]) =>
  dispatched.filter((e) => e?.type === EventType.MESSAGES_SNAPSHOT);

describe("getStateAndMessagesSnapshots skips a snapshot that lags the stream", () => {
  it("does not dispatch a snapshot that adds nothing and omits a streamed message", async () => {
    const { agent, dispatched } = makeAgent([msg("u1", "human", "hi")]);

    agent.dispatchEvent({
      type: EventType.TEXT_MESSAGE_START,
      messageId: "ai-1",
      role: "assistant",
    } as any);

    await (agent as any).getStateAndMessagesSnapshots("thread-1");

    expect(snapshots(dispatched)).toHaveLength(0);
    // The state snapshot is unrelated and must still go out.
    expect(dispatched.some((e) => e?.type === EventType.STATE_SNAPSHOT)).toBe(true);
  });

  it("does not dispatch when only a streamed tool call is in flight", async () => {
    const { agent, dispatched } = makeAgent([msg("u1", "human", "hi")]);

    agent.dispatchEvent({
      type: EventType.TOOL_CALL_START,
      toolCallId: "tc-1",
      toolCallName: "search",
      parentMessageId: "ai-1",
    } as any);

    await (agent as any).getStateAndMessagesSnapshots("thread-1");

    expect(snapshots(dispatched)).toHaveLength(0);
  });

  it("still dispatches when the checkpoint carries a message the client has not seen", async () => {
    // The subgraph-ordering case: the checkpoint committed h1 while the
    // supervisor's ai-1 is still streaming. The snapshot adds information, so
    // it must go out.
    const { agent, dispatched } = makeAgent([msg("u1", "human", "hi"), msg("h1", "ai", "booked")]);

    agent.dispatchEvent({
      type: EventType.TEXT_MESSAGE_START,
      messageId: "ai-1",
      role: "assistant",
    } as any);

    await (agent as any).getStateAndMessagesSnapshots("thread-1");

    expect(snapshots(dispatched)).toHaveLength(1);
    expect(snapshots(dispatched)[0].messages.map((m: any) => m.id)).toContain("h1");
  });

  it("still dispatches when nothing was streamed yet", async () => {
    const { agent, dispatched } = makeAgent([msg("u1", "human", "hi")]);

    await (agent as any).getStateAndMessagesSnapshots("thread-1");

    expect(snapshots(dispatched)).toHaveLength(1);
  });

  it("still dispatches once the checkpoint has caught up with the stream", async () => {
    const { agent, dispatched } = makeAgent([msg("u1", "human", "hi"), msg("ai-1", "ai", "done")]);

    agent.dispatchEvent({
      type: EventType.TEXT_MESSAGE_START,
      messageId: "ai-1",
      role: "assistant",
    } as any);

    await (agent as any).getStateAndMessagesSnapshots("thread-1");

    expect(snapshots(dispatched)).toHaveLength(1);
  });
});
