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
# HELPER: Small name-to-class registry for tool-call parsers.
# Behavior: imports the Qwen and Gemini parser modules and re-exports ToolParser
# (abstract base), QwenToolParser and GeminiToolParser as module-level aliases.
# Holds a private mapping from the lowercase string keys "qwen" and "gemini" to
# the corresponding parser classes (classes, not instances).
# Side effects: none beyond importing the parser modules.
"""Registry for different tool parsers."""

from tunix.rl.agentic.parser.tool_parser import gemini_parser
from tunix.rl.agentic.parser.tool_parser import qwen_parser
from tunix.rl.agentic.parser.tool_parser import tool_parser_base

ToolParser = tool_parser_base.ToolParser
QwenToolParser = qwen_parser.QwenToolParser
GeminiToolParser = gemini_parser.GeminiToolParser
_PARSER_REGISTRY = {"qwen": QwenToolParser, "gemini": GeminiToolParser}


def get_tool_parser(parser_name: str = "qwen") -> type[ToolParser]:
  # [FUNCTION DOC]
  # HELPER: Look up a tool-parser class by name.
  # Inputs: parser_name, a string key; defaults to "qwen".
  # Outputs: the parser class (not an instance) registered under that key.
  # Exceptions: ValueError with message "Unknown parser: <name>" when the key is
  # not in the registry (lookup is exact and case-sensitive).
  # Side effects: none.
  if parser_name not in _PARSER_REGISTRY:
    raise ValueError(f"Unknown parser: {parser_name}")
  return _PARSER_REGISTRY[parser_name]
