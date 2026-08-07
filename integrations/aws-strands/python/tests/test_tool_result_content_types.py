"""Regression tests for issue #2233 — non-text tool result content.

A Strands ``toolResult`` can carry ``text``, ``json``, ``image``, ``document``
or ``video`` blocks. The adapter previously only read ``text``: any other
block left ``result_data = None``, which the guard turned into a *dropped*
``TOOL_CALL_RESULT`` — no event on the wire at all.

Every block type must produce a ``TOOL_CALL_RESULT`` whose ``content``
carries the serialized payload. A genuinely void result (empty content
array) must still emit ``content: ""``.

The expected payloads here are exactly what the TypeScript adapter emits for
the same blocks (media bytes base64-encoded, matching the Strands TS SDK's
``toJSON()``), so the two SDKs stay in lockstep. The TypeScript counterpart
(``src/__tests__/tool-result-content-types.test.ts``) asserts the same
payloads over the same bytes.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

from ag_ui.core import EventType, RunAgentInput, UserMessage
from strands.tools.registry import ToolRegistry

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig

# Same byte sequences as the TypeScript test, so the two suites assert
# byte-for-byte identical base64 payloads.
PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PDF_BYTES = b"%PDF"
MP4_BYTES = b"\x00\x00\x00\x18"


def _template_agent() -> MagicMock:
    mock = MagicMock()
    mock.model = MagicMock()
    mock.system_prompt = "You are helpful"
    mock.tool_registry.registry = {}
    mock.record_direct_tool_call = True
    return mock


def _stream_for(result_content: list) -> list:
    """A minimal backend tool call plus its result."""
    return [
        {
            "current_tool_use": {
                "name": "backend_tool",
                "toolUseId": "st-1",
                "input": {},
            }
        },
        {"event": {"contentBlockStop": {}}},
        {
            "message": {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "st-1",
                            "content": result_content,
                        }
                    }
                ],
            }
        },
    ]


async def _result_content_for(thread: str, result_content: list) -> str:
    agent = StrandsAgent(
        _template_agent(), name="test-agent", config=StrandsAgentConfig()
    )
    mock_inner = MagicMock()
    mock_inner.tool_registry = ToolRegistry()
    mock_inner.session_manager = None

    async def _stream(_msg):
        for event in _stream_for(result_content):
            yield event

    mock_inner.stream_async = _stream
    agent._agents_by_thread[thread] = mock_inner

    inp = RunAgentInput(
        thread_id=thread,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", content="go")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    events = [e async for e in agent.run(inp)]
    results = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
    assert results, (
        "expected a TOOL_CALL_RESULT event; the adapter dropped the tool "
        f"result for content={result_content!r}"
    )
    assert results[0].tool_call_id == "st-1"
    return results[0].content


class TestToolResultContentTypes:
    async def test_text_block(self):
        content = await _result_content_for("t-text", [{"text": '{"ok": true}'}])
        assert json.loads(content) == {"ok": True}

    async def test_json_block(self):
        content = await _result_content_for("t-json", [{"json": {"rows": [1, 2]}}])
        assert json.loads(content) == {"rows": [1, 2]}

    async def test_image_block(self):
        block = {"image": {"format": "png", "source": {"bytes": PNG_BYTES}}}
        content = await _result_content_for("t-image", [block])
        assert json.loads(content) == {
            "image": {
                "format": "png",
                "source": {"bytes": base64.b64encode(PNG_BYTES).decode("ascii")},
            }
        }

    async def test_document_block(self):
        block = {
            "document": {
                "format": "pdf",
                "name": "report",
                "source": {"bytes": PDF_BYTES},
            }
        }
        content = await _result_content_for("t-doc", [block])
        assert json.loads(content) == {
            "document": {
                "format": "pdf",
                "name": "report",
                "source": {"bytes": base64.b64encode(PDF_BYTES).decode("ascii")},
            }
        }

    async def test_video_block(self):
        block = {"video": {"format": "mp4", "source": {"bytes": MP4_BYTES}}}
        content = await _result_content_for("t-video", [block])
        assert json.loads(content) == {
            "video": {
                "format": "mp4",
                "source": {"bytes": base64.b64encode(MP4_BYTES).decode("ascii")},
            }
        }

    async def test_void_result_still_emits_empty_content(self):
        """An empty content array is a legitimate side-effect tool, not a
        dropped block: emit the event with ``content: ""``."""
        content = await _result_content_for("t-void", [])
        assert content == ""
