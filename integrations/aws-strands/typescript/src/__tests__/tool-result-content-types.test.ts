/**
 * Regression tests for issue #2233 — non-text tool result content.
 *
 * A Strands `ToolResultBlock` can carry `TextBlock`, `JsonBlock`,
 * `ImageBlock`, `DocumentBlock` or `VideoBlock` content. The adapter
 * previously only read text and `.json`: image/document/video fell through
 * and the TOOL_CALL_RESULT collapsed to `content: ""`, silently dropping the
 * payload.
 *
 * Every block type must produce a TOOL_CALL_RESULT whose `content` carries
 * the serialized payload. A genuinely void result (empty content array) must
 * still emit `content: ""`.
 *
 * The expected payloads mirror the Python adapter's exactly, so the two SDKs
 * stay in lockstep.
 */

import { describe, it, expect } from "vitest";
import {
  DocumentBlock,
  ImageBlock,
  JsonBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
  VideoBlock,
  type ToolResultContent,
} from "@strands-agents/sdk";
import type { AgentStreamEvent } from "@strands-agents/sdk";
import { EventType } from "@ag-ui/core";

import { collect, scriptedStrandsAgent } from "./helpers";

const PNG_BYTES = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);
const PDF_BYTES = new Uint8Array([0x25, 0x50, 0x44, 0x46]);
const MP4_BYTES = new Uint8Array([0x00, 0x00, 0x00, 0x18]);

const b64 = (u: Uint8Array) => Buffer.from(u).toString("base64");

async function resultContentFor(
  content: ToolResultContent[],
): Promise<string | undefined> {
  const events: AgentStreamEvent[] = [
    new ToolUseBlock({
      name: "backend_tool",
      toolUseId: "tu-1",
      input: {},
    }) as unknown as AgentStreamEvent,
    {
      type: "afterToolCallEvent",
      toolUse: { name: "backend_tool", toolUseId: "tu-1" },
      result: new ToolResultBlock({
        toolUseId: "tu-1",
        status: "success",
        content,
      }),
    } as unknown as AgentStreamEvent,
  ];
  const out = await collect(scriptedStrandsAgent(events));
  const result = out.find((e) => e.type === EventType.TOOL_CALL_RESULT) as
    | { content?: string; toolCallId?: string }
    | undefined;
  expect(result, "expected a TOOL_CALL_RESULT event").toBeDefined();
  expect(result?.toolCallId).toBe("tu-1");
  return result?.content;
}

describe("TOOL_CALL_RESULT content for every tool result block type", () => {
  it("forwards a text block", async () => {
    const content = await resultContentFor([new TextBlock('{"ok":true}')]);
    expect(JSON.parse(content!)).toEqual({ ok: true });
  });

  it("forwards a json block", async () => {
    const content = await resultContentFor([
      new JsonBlock({ json: { rows: [1, 2] } }),
    ]);
    expect(JSON.parse(content!)).toEqual({ rows: [1, 2] });
  });

  it("forwards an image block", async () => {
    const content = await resultContentFor([
      new ImageBlock({ format: "png", source: { bytes: PNG_BYTES } }),
    ]);
    expect(JSON.parse(content!)).toEqual({
      image: { format: "png", source: { bytes: b64(PNG_BYTES) } },
    });
  });

  it("forwards a document block", async () => {
    const content = await resultContentFor([
      new DocumentBlock({
        name: "report",
        format: "pdf",
        source: { bytes: PDF_BYTES },
      }),
    ]);
    expect(JSON.parse(content!)).toEqual({
      document: {
        name: "report",
        format: "pdf",
        source: { bytes: b64(PDF_BYTES) },
      },
    });
  });

  it("forwards a video block", async () => {
    const content = await resultContentFor([
      new VideoBlock({ format: "mp4", source: { bytes: MP4_BYTES } }),
    ]);
    expect(JSON.parse(content!)).toEqual({
      video: { format: "mp4", source: { bytes: b64(MP4_BYTES) } },
    });
  });

  it("keeps empty content for a genuinely void result", async () => {
    expect(await resultContentFor([])).toBe("");
  });
});
