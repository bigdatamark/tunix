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
# Placeholder tool parser for Gemini models. Gemini emits native structured
# functionCall objects rather than text-encoded tool calls, so this parser does
# no text parsing. Re-exports BaseTool, ToolCall and ToolParser aliases.
"""Tool parser for Gemini models."""

from typing import Any, List

from tunix.rl.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.agentic.tools import base_tool

BaseTool = base_tool.BaseTool
ToolCall = base_tool.ToolCall
ToolParser = tool_parser_base.ToolParser


class GeminiToolParser(ToolParser):
  # [CLASS DOC]
  # ToolParser implementation for Gemini. Stateless stub: parse always returns an
  # empty list and get_tool_prompt returns a fixed short instruction string.

  def parse(self, model_response: Any) -> list[ToolCall]:
    # [FUNCTION DOC]
    # Stub parse for Gemini output.
    # Inputs: model_response of any type (ignored).
    # Outputs: always an empty list; no tool calls are ever extracted.
    # Side effects/exceptions: none.
    # KNOWN ISSUE: never extracts any tool calls, so Gemini tool use through this
    # parser is effectively unsupported.
    return []

  def get_tool_prompt(
      self,
      tools: List[BaseTool],
      *,
      schema_style: str = "gemini",
  ) -> str:
    # [FUNCTION DOC]
    # Return a fixed tool instruction prompt for Gemini.
    # Inputs: tools (ignored); keyword-only schema_style (default "gemini", ignored).
    # Outputs: the constant string "Return a functionCall with name and args."
    # Side effects/exceptions: none. Tool schemas are not included in the text.
    return "Return a functionCall with name and args."
