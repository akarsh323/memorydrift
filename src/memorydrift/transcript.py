"""Helpers for reading tool activity out of a conversation."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool


@dataclasses.dataclass(frozen=True)
class ToolExchange:
    """One tool call paired with its result.

    Attributes:
      function: Name of the tool that was called.
      arguments: Arguments of the call.
      result: The tool's reply, or `None` if the call has not been answered.
    """

    function: str
    arguments: dict[str, Any]
    result: ChatMessageTool | None

    @property
    def ok(self) -> bool:
        """Whether the call was answered without a tool error."""
        return self.result is not None and self.result.error is None

    @property
    def text(self) -> str:
        """Text of a successful reply, or the empty string."""
        return self.result.text if self.ok and self.result else ""

    @property
    def error(self) -> str | None:
        """Message of a tool error, or `None`."""
        if self.result is None or self.result.error is None:
            return None
        return self.result.error.message


def tool_exchanges(messages: Sequence[ChatMessage]) -> list[ToolExchange]:
    """Pairs every tool call in a conversation with its result.

    Args:
      messages: A conversation, in order.

    Returns:
      One exchange per tool call, in call order.
    """
    replies = {
        m.tool_call_id: m for m in messages if isinstance(m, ChatMessageTool)
    }
    return [
        ToolExchange(call.function, call.arguments, replies.get(call.id))
        for message in messages
        if isinstance(message, ChatMessageAssistant)
        for call in message.tool_calls or []
    ]
