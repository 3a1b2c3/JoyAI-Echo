"""Debug hook that prints each iteration's LLM output and context stats."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.utils.helpers import estimate_message_tokens


class DebugHook(AgentHook):
    """Prints detailed per-iteration info for debugging context management.

    Enable by passing this hook to AgentLoop or using --debug flag.
    """

    def __init__(self, *, show_full_content: bool = True, show_messages: bool = False) -> None:
        super().__init__()
        self._show_full_content = show_full_content
        self._show_messages = show_messages

    async def after_iteration(self, context: AgentHookContext) -> None:
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"[DEBUG] Iteration {context.iteration}")
        print(f"{sep}")

        # Token usage
        if context.usage:
            print(f"  Tokens — prompt: {context.usage.get('prompt_tokens', '?')}, "
                  f"completion: {context.usage.get('completion_tokens', '?')}")

        # Response content
        if context.response:
            content = context.response.content or ""
            print(f"  Response length: {len(content)} chars")
            if self._show_full_content and content:
                print(f"  --- Response content ---")
                print(f"  {content[:2000]}")
                if len(content) > 2000:
                    print(f"  ... [truncated, total {len(content)} chars]")
                print(f"  --- End response ---")

            # Reasoning/thinking
            if context.response.reasoning_content:
                rc = context.response.reasoning_content
                print(f"  Reasoning: {len(rc)} chars")

        # Tool calls
        if context.tool_calls:
            print(f"  Tool calls ({len(context.tool_calls)}):")
            for tc in context.tool_calls:
                args_preview = json.dumps(tc.arguments, ensure_ascii=False)
                if len(args_preview) > 200:
                    args_preview = args_preview[:200] + "..."
                print(f"    - {tc.name}({args_preview})")

        # Tool results
        if context.tool_results:
            print(f"  Tool results ({len(context.tool_results)}):")
            for i, result in enumerate(context.tool_results):
                result_str = str(result)
                print(f"    [{i}] {result_str[:300]}")
                if len(result_str) > 300:
                    print(f"        ... [truncated, total {len(result_str)} chars]")

        # Messages context size
        if context.messages:
            total_msgs = len(context.messages)
            estimated_tokens = estimate_prompt_tokens_simple(context.messages)
            print(f"  Messages in context: {total_msgs}, ~{estimated_tokens} tokens (estimate)")

        # Show full messages if requested
        if self._show_messages and context.messages:
            print(f"  --- Full messages ---")
            for i, msg in enumerate(context.messages[-5:]):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:200]
                else:
                    preview = str(content)[:200]
                print(f"    [{total_msgs - 5 + i}] {role}: {preview}")
            print(f"  --- End messages (showing last 5 of {total_msgs}) ---")

        # Stop reason / error
        if context.stop_reason:
            print(f"  Stop reason: {context.stop_reason}")
        if context.error:
            print(f"  ERROR: {context.error}")

        print(f"{sep}\n")


def estimate_prompt_tokens_simple(messages: list[dict[str, Any]]) -> int:
    """Quick estimate of total tokens in message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    total += len(str(text)) // 4
        # tool calls
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                total += len(str(args)) // 4
    return total
