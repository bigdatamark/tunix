# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [MODULE DOC]
# Tool parser for Qwen-family models, whose tool calls are emitted as a JSON
# object wrapped in <tool_call> ... </tool_call> tags, and whose tool results are
# wrapped in <tool_response> ... </tool_response> tags. Re-exports ToolParser,
# ToolCall and BaseTool aliases.
"""Tool parser for Qwen models.

This parser handles the specific tool call format used by Qwen models,
which involves XML-like tags around a JSON object.
"""

import json
from typing import Any

from tunix.rl.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.agentic.tools import base_tool

ToolParser = tool_parser_base.ToolParser
ToolCall = base_tool.ToolCall
BaseTool = base_tool.BaseTool


class QwenToolParser(ToolParser):
  # [CLASS DOC]
  # ToolParser for Qwen's tag-wrapped JSON tool-call format.
  # State: four string attributes holding the begin/end tags for tool calls
  # ("<tool_call>", "</tool_call>") and tool outputs ("<tool_response>",
  # "</tool_response>"). Provides parsing of calls and a Qwen-style tool prompt.
  """Tool parser for Qwen models.

  This parser handles the specific tool call format used by Qwen models,
  which involves XML-like tags around a JSON object.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Initialize the tag strings: tool_call_begin, tool_call_end,
    # tool_output_begin, tool_output_end (values listed in the class doc).
    # Inputs: none. Outputs: none. Side effects: sets instance attributes.
    # KNOWN ISSUE: docstring mentions "specified type and model" but the
    # constructor takes no arguments.
    """Initialize the parser with specified type and model."""
    self.tool_call_begin = "<tool_call>"
    self.tool_call_end = "</tool_call>"
    self.tool_output_begin = "<tool_response>"
    self.tool_output_end = "</tool_response>"

  def parse(self, model_response: str) -> list[ToolCall]:
    # [FUNCTION DOC]
    # Parse model text into ToolCall objects.
    # Inputs: model_response string.
    # Behavior: calls parse_qwen_tool_calls to get dicts with "name" and
    # "arguments", then constructs ToolCall by passing each dict's keys as keyword
    # arguments.
    # Outputs: list of ToolCall in the order they appear in the text.
    # Exceptions: propagates errors from ToolCall construction.
    """Parse tool calls from model output.

    Args:
        model_response (str): Text containing tool calls

    Returns:
        list[ToolCall]: Parsed tool calls
    """
    tool_calls_dicts = self.parse_qwen_tool_calls(model_response)
    tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls_dicts]
    return tool_calls

  def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # Extract raw tool-call dicts from text.
    # Inputs: text string.
    # Behavior: if the begin tag is absent, returns an empty list. Otherwise loops
    # while the begin tag occurs in the remaining text: takes the content between the
    # first begin tag and the first end tag (the end tag is searched from the start
    # of the remaining text, not after the begin tag), strips whitespace and decodes
    # it as JSON. On success appends a dict with keys "name" and "arguments" taken
    # from the decoded object. On JSON decode failure prints an error message with
    # the bad content to stdout and skips it. In both cases the remaining text is
    # advanced past the end tag. If no end tag is found the loop stops.
    # Outputs: list of dicts each with "name" and "arguments".
    # Exceptions: KeyError if decoded JSON lacks "name" or "arguments"; TypeError if
    # the decoded JSON is not a mapping.
    # Side effects: prints to stdout on malformed JSON.
    # KNOWN ISSUE: docstring says each dict contains 'parameters', but the key is
    # actually 'arguments'. Also, if an end tag appears before a begin tag the slice
    # is wrong and can loop on misaligned content; KeyError for missing keys is not
    # caught.
    """Parse tool calls from text using a simple token format.

    Format:
    <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

    Args:
        text (str): The input text potentially containing tool calls.

    Returns:
        list[dict]: List of parsed tool calls, each containing 'name' and
        'parameters'
    """

    tool_calls: list[dict[str, Any]] = []

    # Return empty list if no tool calls found
    if self.tool_call_begin not in text:
      return tool_calls

    # Process all tool calls in the text
    while self.tool_call_begin in text:
      start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
      end = text.find(self.tool_call_end)
      if end == -1:
        break

      # Extract and parse the JSON content
      json_content = text[start:end].strip()
      try:
        call_data = json.loads(json_content)
        # Convert to common format matching parse_tool_calls output
        tool_calls.append(
            {"name": call_data["name"], "arguments": call_data["arguments"]}
        )
      except json.JSONDecodeError:
        print(f"Error parsing tool call: {json_content}")
        text = text[end + len(self.tool_call_end) :]
        continue

      # Move to next potential tool call
      text = text[end + len(self.tool_call_end) :]

    return tool_calls

  def get_tool_prompt(
      self,
      tools: list[BaseTool],
      schema_style: str = "openai",
  ) -> str:
    # [FUNCTION DOC]
    # Build the Qwen tool-usage prompt.
    # Inputs: tools, list of BaseTool; schema_style (default "openai"; positional,
    # unlike the base class where it is keyword-only).
    # Behavior: dumps the tool schemas via the base helper, then returns a fixed
    # template that wraps the schema JSON in <tools></tools> tags and instructs the
    # model to return each call as a JSON object with "name" and "arguments" inside
    # <tool_call></tool_call> tags. Leading/trailing whitespace is stripped.
    # Outputs: the prompt string. Exceptions: those of the schema dump helper.
    tools_schema = self._tools_schema_dump(tools, schema_style=schema_style)
    return f"""
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>
""".strip()
