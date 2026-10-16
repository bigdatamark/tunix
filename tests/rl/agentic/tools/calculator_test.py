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
# Unit tests for tunix.rl.agentic.tools.calculator_tool.CalculatorTool.
# Checks the generated JSON function schema and the apply method for the four
# arithmetic operators, division by zero, and an unsupported operator.
# apply returns a result object with output (string or None) and error
# (string or None). No side effects.

from absl.testing import absltest
from tunix.rl.agentic.tools import calculator_tool


class CalculatorToolTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for CalculatorTool; a fresh tool named "calculator" is built per test.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture creating self.tool as a CalculatorTool with name
    # "calculator" and description "A calculator tool.".
    super().setUp()
    self.tool = calculator_tool.CalculatorTool(
        name="calculator", description="A calculator tool."
    )

  def test_get_json_schema(self):
    # [FUNCTION DOC]
    # Verifies the schema has type "function", function name "calculator",
    # a description, properties a, b, op, and required list exactly a, b, op.
    schema = self.tool.get_json_schema()
    self.assertEqual(schema["type"], "function")
    self.assertEqual(schema["function"]["name"], "calculator")
    self.assertIn("description", schema["function"])
    self.assertIn("a", schema["function"]["parameters"]["properties"])
    self.assertIn("b", schema["function"]["parameters"]["properties"])
    self.assertIn("op", schema["function"]["parameters"]["properties"])
    self.assertEqual(
        schema["function"]["parameters"]["required"], ["a", "b", "op"]
    )

  def test_add(self):
    # [FUNCTION DOC]
    # Verifies 5 + 3 yields output "8" and no error.
    result = self.tool.apply(a=5, b=3, op="+")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "8")

  def test_subtract(self):
    # [FUNCTION DOC]
    # Verifies 5 - 3 yields output "2" and no error.
    result = self.tool.apply(a=5, b=3, op="-")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "2")

  def test_multiply(self):
    # [FUNCTION DOC]
    # Verifies 5 * 3 yields output "15" and no error.
    result = self.tool.apply(a=5, b=3, op="*")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "15")

  def test_divide(self):
    # [FUNCTION DOC]
    # Verifies 6 / 3 uses true division, yielding output "2.0" and no error.
    result = self.tool.apply(a=6, b=3, op="/")
    self.assertIsNone(result.error)
    self.assertEqual(result.output, "2.0")

  def test_divide_by_zero(self):
    # [FUNCTION DOC]
    # Verifies dividing by zero returns an error containing
    # "Division by zero" and a None output (no exception propagates).
    result = self.tool.apply(a=5, b=0, op="/")
    self.assertIsNotNone(result.error)
    self.assertIn("Division by zero", result.error)
    self.assertIsNone(result.output)

  def test_unsupported_operator(self):
    # [FUNCTION DOC]
    # Verifies operator "^" returns an error containing
    # "Unsupported operator" and a None output.
    result = self.tool.apply(a=5, b=3, op="^")
    self.assertIsNotNone(result.error)
    self.assertIn("Unsupported operator", result.error)
    self.assertIsNone(result.output)


if __name__ == "__main__":
  absltest.main()
