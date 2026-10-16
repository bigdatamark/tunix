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
# Unit tests for the agentic tool layer: base_tool (ToolCall, ToolOutput
# string rendering, BaseTool abstractness, default apply/apply_async raising
# NotImplementedError, attribute immutability after init) and
# tool_manager.ToolManager (construction from a name-to-class map with
# description fallback, JSON/MCP schema export, registering tool instances,
# sync run with error capture, and concurrent async runs).
# Side effects: one async test sleeps about 1 second of wall time.
# KNOWN ISSUE: there is no main entry point, so the file only runs under an
# external test runner.

import asyncio
import time
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.tools import tool_manager


class BaseToolTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for base_tool data types and BaseTool contract.

  def test_tool_call_creation(self):
    # [FUNCTION DOC]
    # Verifies ToolCall stores name and arguments as given.
    tool_call = base_tool.ToolCall(
        name="test_tool", arguments={"arg1": "value1"}
    )
    self.assertEqual(tool_call.name, "test_tool")
    self.assertEqual(tool_call.arguments, {"arg1": "value1"})

  def test_tool_output_repr(self):
    # [FUNCTION DOC]
    # Verifies ToolOutput string form: "Error: <msg>" when error is set,
    # JSON for dict output, Python list text for list output, the string
    # itself for string output, and empty string for None output.
    # Test with error
    error_output = base_tool.ToolOutput(
        name="test", error="Something went wrong"
    )
    self.assertEqual(str(error_output), "Error: Something went wrong")

    # Test with dict output
    dict_output = base_tool.ToolOutput(name="test", output={"key": "value"})
    self.assertEqual(str(dict_output), '{"key": "value"}')

    # Test with list output
    list_output = base_tool.ToolOutput(name="test", output=[1, 2, 3])
    self.assertEqual(str(list_output), "[1, 2, 3]")

    # Test with string output
    str_output = base_tool.ToolOutput(name="test", output="Success")
    self.assertEqual(str(str_output), "Success")

    # Test with None output
    none_output = base_tool.ToolOutput(name="test", output=None)
    self.assertEqual(str(none_output), "")

  def test_base_tool_abstract_methods(self):
    # [FUNCTION DOC]
    # Verifies instantiating a BaseTool subclass lacking get_json_schema
    # raises TypeError.
    # Test that a subclass without implementing abstract methods fails.
    with self.assertRaises(TypeError):

      # pylint: disable=abstract-class-instantiated
      class IncompleteTool(base_tool.BaseTool):
        # [CLASS DOC]
        # HELPER: BaseTool subclass with no abstract methods implemented.
        pass

      IncompleteTool(name="incomplete", description="...")

  def test_base_tool_apply_not_implemented(self):
    # [FUNCTION DOC]
    # Verifies the default synchronous apply raises NotImplementedError.

    class TestTool(base_tool.BaseTool):
      # [CLASS DOC]
      # HELPER: minimal tool implementing only get_json_schema.

      def get_json_schema(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns an empty schema dict.
        return {}

    tool = TestTool(name="test", description="A test tool.")
    with self.assertRaises(NotImplementedError):
      tool.apply()

  def test_base_tool_apply_async_not_implemented(self):
    # [FUNCTION DOC]
    # Verifies the default apply_async raises NotImplementedError rather
    # than delegating to an implemented synchronous apply.

    class TestTool(base_tool.BaseTool):
      # [CLASS DOC]
      # HELPER: tool implementing schema and sync apply but not apply_async.

      def get_json_schema(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns an empty schema dict.
        return {}

      def apply(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: returns ToolOutput with this tool's name and "sync result".
        # This is implemented, but apply_async should not delegate to it.
        return base_tool.ToolOutput(name=self.name, output="sync result")

    tool = TestTool(name="test", description="A test tool.")
    with self.assertRaises(NotImplementedError):
      asyncio.run(tool.apply_async())

  def test_base_tool_immutability(self):
    # [FUNCTION DOC]
    # Verifies a tool whose subclass init sets an extra attribute exposes
    # name, description and extra, and that after init any attribute
    # modification or addition raises AttributeError "Cannot modify
    # immutable".

    class MultiAttrTool(base_tool.BaseTool):
      # [CLASS DOC]
      # HELPER: tool with an additional "extra" attribute set in init.

      def __init__(self, name: str, description: str, extra: str):
        # [FUNCTION DOC]
        # HELPER: calls the BaseTool initializer then stores extra.
        super().__init__(name, description)
        self.extra = extra

      def get_json_schema(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns an empty schema dict.
        return {}

    tool = MultiAttrTool(name="test", description="desc", extra="val")

    # Verify initial values
    self.assertEqual(tool.name, "test")
    self.assertEqual(tool.description, "desc")
    self.assertEqual(tool.extra, "val")

    # Attempt to modify existing attribute
    with self.assertRaisesRegex(AttributeError, "Cannot modify immutable"):
      tool.name = "new_name"

    with self.assertRaisesRegex(AttributeError, "Cannot modify immutable"):
      tool.extra = "new_val"

    # Attempt to add new attribute
    with self.assertRaisesRegex(AttributeError, "Cannot modify immutable"):
      tool.new_attr = 42


class ToolManagerTest(unittest.IsolatedAsyncioTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for ToolManager; supports async test methods. A shared manager
  # holding the calculator tool is created once per class.

  # Test docstring fallback
  class NoDocstringTool(base_tool.BaseTool):
    # [CLASS DOC]
    # HELPER: tool class without a docstring, used to exercise the
    # description fallback.

    def get_json_schema(self) -> dict[str, Any]:
      # [FUNCTION DOC]
      # HELPER: returns an empty schema dict.
      return {}

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # HELPER: creates cls.tool_map (calculator name to CalculatorTool class)
    # and cls.manager, a ToolManager built from it.
    super().setUpClass()
    cls.tool_map = {"calculator": calculator_tool.CalculatorTool}
    cls.manager = tool_manager.ToolManager(cls.tool_map)

  def test_initialization(self):
    # [FUNCTION DOC]
    # Verifies the manager instantiates each mapped class under its name,
    # and that a tool class lacking a docstring gets desc_fallback as its
    # description.
    self.assertIn("calculator", self.manager.names)
    self.assertIsInstance(
        self.manager._tool_dict["calculator"], calculator_tool.CalculatorTool
    )
    # Test docstring fallback
    manager = tool_manager.ToolManager(
        {"no_doc": self.NoDocstringTool}, desc_fallback="fallback"
    )
    self.assertEqual(manager._tool_dict["no_doc"].description, "fallback")

  def test_get_json_schema(self):
    # [FUNCTION DOC]
    # Verifies get_json_schema returns one schema per tool, named calculator.
    schemas = self.manager.get_json_schema()
    self.assertLen(schemas, 1)
    self.assertEqual(schemas[0]["function"]["name"], "calculator")

  def test_get_mcp_schema(self):
    # [FUNCTION DOC]
    # Verifies get_mcp_schema returns one entry whose function name is
    # calculator (same shape as the JSON schema).
    schemas = self.manager.get_mcp_schema()
    self.assertLen(schemas, 1)
    self.assertEqual(schemas[0]["function"]["name"], "calculator")

  def test_register_mcp_tool(self):
    # [FUNCTION DOC]
    # Verifies register_mcp_tool adds a tool instance under its name.
    manager = tool_manager.ToolManager(self.tool_map)
    class NewTool(base_tool.BaseTool):
      # [CLASS DOC]
      # HELPER: tool whose schema names it new_tool.

      def get_json_schema(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns a function-type schema named new_tool.
        return {"type": "function", "function": {"name": "new_tool"}}

    new_tool = NewTool(name="new_tool", description="A new tool.")
    manager.register_mcp_tool(new_tool)
    self.assertIn("new_tool", manager.names)

  def test_run_success(self):
    # [FUNCTION DOC]
    # Verifies run dispatches keyword args to the named tool (10 + 5 gives
    # "15", no error).
    result = self.manager.run("calculator", a=10, b=5, op="+")
    self.assertEqual(result.output, "15")
    self.assertIsNone(result.error)

  def test_run_tool_not_found(self):
    # [FUNCTION DOC]
    # Verifies running an unknown tool returns an error output containing
    # "not registered" instead of raising.
    result = self.manager.run("non_existent_tool")
    self.assertIsNotNone(result.error)
    self.assertIn("not registered", result.error)

  def test_run_tool_exception(self):
    # [FUNCTION DOC]
    # Verifies an exception raised by a tool's apply is caught by run and
    # reported as a non-None error on the returned output.
    manager = tool_manager.ToolManager(self.tool_map)
    mock_tool = mock.Mock(spec=base_tool.BaseTool)
    mock_tool.name = "buggy_tool"
    mock_tool.apply.side_effect = ValueError("Something broke")
    manager.register_mcp_tool(mock_tool)
    result = manager.run("buggy_tool", arg=1)
    self.assertIsNotNone(result.error)

  async def test_run_async_parallel(self):
    # [FUNCTION DOC]
    # Verifies two run_async calls on tools that each sleep 1s, awaited
    # together, both return "done" and finish in under 2s total (concurrent).
    # Side effects: about 1s of real sleep.
    """Tests that multiple async tools can run concurrently via run_async."""

    class SleepTool(base_tool.BaseTool):
      # [CLASS DOC]
      # HELPER: async tool that sleeps then returns "done".
      """A simple tool that sleeps asynchronously."""

      async def apply_async(
          self, sleep_duration: float = 0.01
      ) -> base_tool.ToolOutput:
        # [FUNCTION DOC]
        # HELPER: awaits sleep_duration seconds, returns ToolOutput "done".
        await asyncio.sleep(sleep_duration)
        return base_tool.ToolOutput(name=self.name, output="done")

      def get_json_schema(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: returns an empty schema dict.
        return {}

    manager = tool_manager.ToolManager({})
    manager.register_mcp_tool(SleepTool(name="sleep1", description="..."))
    manager.register_mcp_tool(SleepTool(name="sleep2", description="..."))

    sleep_time = 1  # 1s

    start_time = time.perf_counter()
    results = await asyncio.gather(
        manager.run_async("sleep1", sleep_duration=sleep_time),
        manager.run_async("sleep2", sleep_duration=sleep_time),
    )
    end_time = time.perf_counter()
    duration = end_time - start_time

    self.assertLen(results, 2)
    self.assertEqual(results[0].output, "done")
    self.assertEqual(results[1].output, "done")
    self.assertLess(duration, sleep_time * 2)
