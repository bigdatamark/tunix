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
# Defines the abstract ToolParser interface used by agents to (1) turn raw model
# text into structured ToolCall objects and (2) render a tool-usage instruction
# prompt that lists the available tools' JSON schemas. Depends on base_tool for
# the BaseTool and ToolCall types. Exposes a module alias abstractmethod for
# abc.abstractmethod. No side effects at import time.
"""Base classes for parsing tool calls and generating tool prompts."""

import abc
import json
from typing import Any, List, Literal

from tunix.rl.agentic.tools import base_tool

abstractmethod = abc.abstractmethod


class ToolParser(abc.ABC):
  # [CLASS DOC]
  # Abstract base class (abc.ABC) for model-specific tool parsers.
  # Subclasses must implement parse and get_tool_prompt. Provides a shared helper
  # that serializes tool schemas to pretty-printed JSON and a default no-op
  # parse_tool_outputs. Holds no state of its own.
  """Abstract base class for all tool parsers.

  A ToolParser defines how to: 1. Extract structured tool calls from raw model
  responses. 2. Generate tool prompting text (tool specs / examples) for model
  input.
  """

  @abstractmethod
  def parse(self, model_response: str) -> list[base_tool.ToolCall]:
    # [FUNCTION DOC]
    # Abstract. Extract tool calls from a model response.
    # Inputs: model_response, the full generated text from the LLM.
    # Outputs: list of base_tool.ToolCall (empty if none found).
    # The base implementation does nothing and returns None; subclasses must
    # override. Exceptions/side effects are implementation-defined.
    """Parse model output and return a list of tool calls.

    Args:
        model_response (str): The full LLM output text.

    Returns:
        list[ToolCall]: Parsed tool call(s).
    """
    pass

  @abstractmethod
  def get_tool_prompt(
      self,
      tools: List[base_tool.BaseTool],
      *,
      schema_style: Literal["openai", "mcp", "gemini"] = "openai",
  ) -> str:
    # [FUNCTION DOC]
    # Abstract. Build the instruction text telling the model which tools exist and
    # how to format a call.
    # Inputs: tools, a list of BaseTool instances; keyword-only schema_style, one of
    # "openai", "mcp" or "gemini" (default "openai") selecting the schema format.
    # Outputs: prompt string containing the tool schemas.
    # The base implementation returns None; subclasses must override.
    """Generate tool-usage instruction prompt from a list of tools.

    Args:
        tools: List of tool instances (BaseTool).
        schema_style: "openai" -> use tool.json (OpenAI function-calling style)
          "mcp"    -> use tool.to_mcp_json() (MCP-compatible format) "gemini" ->
          use Gemini-compatible schema

    Returns:
        str: Prompt text to feed into the model (includes tool schemas).
    """
    pass

  def _tools_schema_dump(
      self,
      tools: List[base_tool.BaseTool],
      schema_style: str,
  ) -> str:
    # [FUNCTION DOC]
    # HELPER: Serialize the schemas of a list of tools to a JSON string.
    # Inputs: tools, list of BaseTool; schema_style string.
    # Behavior: if schema_style is "mcp", each tool's to_mcp_json() result is used;
    # for "gemini" and any other value, each tool's get_json_schema() result is used.
    # The resulting list is dumped as JSON with ensure_ascii disabled (non-ASCII kept)
    # and an indentation of 2 spaces.
    # Outputs: the JSON string (a JSON array, one entry per tool, in input order).
    # Exceptions: propagates any error from the tool schema methods or from JSON
    # serialization of non-serializable values.
    # KNOWN ISSUE: the docstring says "openai" and "gemini" use the tool's `.json`
    # property, but the code calls get_json_schema() for both; the "gemini" branch is
    # identical to the default branch.
    """Dumps a list of tool schemas to a JSON string.

    Args:
        tools: List of tool instances (BaseTool).
        schema_style: The style of schema to dump. "openai" and "gemini" use the
          tool's `.json` property, while "mcp" uses `to_mcp_json()`.

    Returns:
        A JSON string representation of the tool schemas.
    """
    if schema_style == "mcp":
      schemas = [t.to_mcp_json() for t in tools]
    elif schema_style == "gemini":
      # Gemini also uses JSON schema, same as OpenAI
      schemas = [t.get_json_schema() for t in tools]
    else:
      schemas = [t.get_json_schema() for t in tools]
    return json.dumps(schemas, ensure_ascii=False, indent=2)

  def parse_tool_outputs(self) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Optional hook for parsing tool output blocks (e.g. <tool_response>) from text.
    # Inputs: none besides self (note: it takes no text argument).
    # Outputs: always an empty dict in the base class; subclasses may override to
    # return a mapping from tool name/ID to its result.
    # Side effects: none.
    """Optional: Parse tool outputs (e.g. <tool_response> blocks).

    Override if your model uses them.

    Returns:
        dict[str, Any]: Mapping from tool name or ID to its result.
    """
    return {}
