import { Subject, firstValueFrom } from "rxjs";
import { toArray } from "rxjs/operators";
import { AssistantMessage, BaseEvent, EventType, Message, RunAgentInput } from "@ag-ui/core";
import { defaultApplyEvents } from "../default";
import { AbstractAgent } from "@/agent";
import { AgentStateMutation } from "@/agent/subscriber";

// defaultApplyEvents only reads agent.messages (and passes agent to subscriber
// callbacks, which are empty here), so a partial stub is sufficient.
const createAgent = (messages: Message[] = []) =>
  ({ messages: messages.map((m) => ({ ...m })), state: {} }) as unknown as AbstractAgent;

const makeInput = (messages: Message[] = []): RunAgentInput => ({
  messages,
  state: {},
  threadId: "thread-test",
  runId: "run-test",
  tools: [],
  context: [],
});

async function emitAndCollect(
  initial: Message[],
  emit: (events$: Subject<BaseEvent>) => void,
): Promise<AgentStateMutation[]> {
  const events$ = new Subject<BaseEvent>();
  const agent = createAgent(initial);
  const result$ = defaultApplyEvents(makeInput(initial), events$, agent, []);
  const updatesPromise = firstValueFrom(result$.pipe(toArray()));
  emit(events$);
  events$.complete();
  return updatesPromise;
}

/** Final messages after all emitted events. */
async function finalMessages(
  initial: Message[],
  emit: (events$: Subject<BaseEvent>) => void,
): Promise<Message[]> {
  const updates = await emitAndCollect(initial, emit);
  for (let i = updates.length - 1; i >= 0; i--) {
    if (updates[i].messages !== undefined) return updates[i].messages!;
  }
  return initial;
}

const userMessage: Message = { id: "user-1", role: "user", content: "do the thing" };

describe("MESSAGES_SNAPSHOT that lags the stream (CopilotKit#6301)", () => {
  it("does not delete an in-flight assistant message, and later deltas still apply", async () => {
    // Reporter's shape: LangGraph pins ONE messageId across text -> tool -> text,
    // then dispatches a checkpoint-derived snapshot that is behind the stream.
    const pinned = "pinned-msg-1";

    const messages = await finalMessages([userMessage], (events$) => {
      events$.next({ type: EventType.RUN_STARTED, threadId: "thread-test", runId: "run-test" });
      events$.next({ type: EventType.TEXT_MESSAGE_START, messageId: pinned, role: "assistant" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: pinned, delta: "Let me " });
      events$.next({
        type: EventType.TOOL_CALL_START,
        toolCallId: "tc-1",
        toolCallName: "search",
        parentMessageId: pinned,
      });
      events$.next({ type: EventType.TOOL_CALL_ARGS, toolCallId: "tc-1", delta: '{"q":"agui"}' });
      events$.next({ type: EventType.TOOL_CALL_END, toolCallId: "tc-1" });

      // The lagging checkpoint: only the user message.
      events$.next({ type: EventType.MESSAGES_SNAPSHOT, messages: [userMessage] });

      // Stream continues under the same pinned id.
      events$.next({
        type: EventType.TOOL_CALL_START,
        toolCallId: "tc-2",
        toolCallName: "search",
        parentMessageId: pinned,
      });
      events$.next({ type: EventType.TOOL_CALL_ARGS, toolCallId: "tc-2", delta: '{"q":"more"}' });
      events$.next({ type: EventType.TOOL_CALL_END, toolCallId: "tc-2" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: pinned, delta: "check." });
      events$.next({ type: EventType.TEXT_MESSAGE_END, messageId: pinned });
      events$.next({ type: EventType.RUN_FINISHED, threadId: "thread-test", runId: "run-test" });
    });

    const assistant = messages.find((m) => m.id === pinned) as AssistantMessage | undefined;
    expect(assistant).toBeDefined();
    expect(assistant!.content).toBe("Let me check.");
    expect(assistant!.toolCalls?.map((tc) => tc.function.arguments)).toEqual([
      '{"q":"agui"}',
      '{"q":"more"}',
    ]);
    expect(messages.map((m) => m.id)).toEqual([userMessage.id, pinned]);
  });

  it("keeps a tool result paired with a preserved in-flight assistant message", async () => {
    const pinned = "pinned-msg-2";

    const messages = await finalMessages([userMessage], (events$) => {
      events$.next({ type: EventType.RUN_STARTED, threadId: "thread-test", runId: "run-test" });
      events$.next({ type: EventType.TEXT_MESSAGE_START, messageId: pinned, role: "assistant" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: pinned, delta: "working" });
      events$.next({
        type: EventType.TOOL_CALL_START,
        toolCallId: "tc-2",
        toolCallName: "search",
        parentMessageId: pinned,
      });
      events$.next({ type: EventType.TOOL_CALL_ARGS, toolCallId: "tc-2", delta: "{}" });
      events$.next({ type: EventType.TOOL_CALL_END, toolCallId: "tc-2" });
      events$.next({
        type: EventType.TOOL_CALL_RESULT,
        messageId: "tool-result-1",
        toolCallId: "tc-2",
        content: "ok",
      });

      events$.next({ type: EventType.MESSAGES_SNAPSHOT, messages: [userMessage] });
    });

    expect(messages.map((m) => m.id)).toEqual([userMessage.id, pinned, "tool-result-1"]);
  });

  it("preserves a message whose tool call is still open even after its text ended", async () => {
    const pinned = "pinned-msg-3";

    const messages = await finalMessages([userMessage], (events$) => {
      events$.next({ type: EventType.RUN_STARTED, threadId: "thread-test", runId: "run-test" });
      events$.next({ type: EventType.TEXT_MESSAGE_START, messageId: pinned, role: "assistant" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: pinned, delta: "hi" });
      events$.next({ type: EventType.TEXT_MESSAGE_END, messageId: pinned });
      events$.next({
        type: EventType.TOOL_CALL_START,
        toolCallId: "tc-3",
        toolCallName: "search",
        parentMessageId: pinned,
      });
      events$.next({ type: EventType.MESSAGES_SNAPSHOT, messages: [userMessage] });
      events$.next({ type: EventType.TOOL_CALL_ARGS, toolCallId: "tc-3", delta: '{"a":1}' });
      events$.next({ type: EventType.TOOL_CALL_END, toolCallId: "tc-3" });
    });

    const assistant = messages.find((m) => m.id === pinned) as AssistantMessage | undefined;
    expect(assistant?.toolCalls?.[0]?.function.arguments).toBe('{"a":1}');
  });

  it("still replaces authoritatively when nothing is in flight", async () => {
    // A legitimate snapshot must keep its delete semantics: no message is open,
    // so the assistant message the backend dropped must go.
    const messages = await finalMessages(
      [userMessage, { id: "old-assistant", role: "assistant", content: "stale" }] as Message[],
      (events$) => {
        events$.next({ type: EventType.MESSAGES_SNAPSHOT, messages: [userMessage] });
      },
    );

    expect(messages.map((m) => m.id)).toEqual([userMessage.id]);
  });

  it("still replaces authoritatively for messages whose stream already closed", async () => {
    const closed = "closed-msg";

    const messages = await finalMessages([userMessage], (events$) => {
      events$.next({ type: EventType.RUN_STARTED, threadId: "thread-test", runId: "run-test" });
      events$.next({ type: EventType.TEXT_MESSAGE_START, messageId: closed, role: "assistant" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: closed, delta: "done" });
      events$.next({ type: EventType.TEXT_MESSAGE_END, messageId: closed });
      events$.next({ type: EventType.MESSAGES_SNAPSHOT, messages: [userMessage] });
    });

    expect(messages.map((m) => m.id)).toEqual([userMessage.id]);
  });

  it("applies snapshot content to an in-flight message that the snapshot does carry", async () => {
    const pinned = "pinned-msg-4";

    const messages = await finalMessages([userMessage], (events$) => {
      events$.next({ type: EventType.RUN_STARTED, threadId: "thread-test", runId: "run-test" });
      events$.next({ type: EventType.TEXT_MESSAGE_START, messageId: pinned, role: "assistant" });
      events$.next({ type: EventType.TEXT_MESSAGE_CONTENT, messageId: pinned, delta: "partial" });
      events$.next({
        type: EventType.MESSAGES_SNAPSHOT,
        messages: [userMessage, { id: pinned, role: "assistant", content: "canonical" }],
      });
    });

    const assistant = messages.find((m) => m.id === pinned) as AssistantMessage | undefined;
    expect(assistant?.content).toBe("canonical");
  });
});
