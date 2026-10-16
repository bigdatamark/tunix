# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [MODULE DOC]
# Unit tests for tunix.rl.function_registry.FunctionRegistry, a registry of
# named functions grouped by category (defaults include policy_loss_fn,
# advantage_estimator, loss_agg and a fourth default category). Covers
# category listing, decorator-based registration and lookup, duplicate
# registration (warning and overwrite), and errors for invalid categories.
# Side effects: none; each test uses its own registry instance.

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import function_registry

# from my_module import function_registry # Or include class def above


# --- Dummy functions for testing ---
def dummy_func_a(x):
  # [FUNCTION DOC]
  # HELPER: test payload function; returns x plus 1.
  return x + 1


def dummy_func_b(x, y):
  # [FUNCTION DOC]
  # HELPER: test payload function; returns x times y.
  return x * y


class FunctionRegistryTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests for FunctionRegistry construction, registration, lookup and errors.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates self.registry with the default allowed categories.
    super().setUp()
    # Initialize a registry with default categories for most tests
    self.registry = function_registry.FunctionRegistry()

  def test_default_categories_instance(self):
    # [FUNCTION DOC]
    # Verifies a default registry lists exactly DEFAULT_ALLOWED_CATEGORIES.
    self.assertCountEqual(
        self.registry.list_categories(),
        function_registry.FunctionRegistry.DEFAULT_ALLOWED_CATEGORIES,
    )

  def test_custom_categories_instance(self):
    # [FUNCTION DOC]
    # Verifies passing allowed_categories cat1, cat2 makes list_categories
    # return exactly those.
    custom_cats = ["cat1", "cat2"]
    # Test-specific instance for custom categories
    registry = function_registry.FunctionRegistry(
        allowed_categories=custom_cats
    )
    self.assertCountEqual(registry.list_categories(), custom_cats)

  def test_empty_categories_instance(self):
    # [FUNCTION DOC]
    # Verifies an empty allowed_categories list falls back to the defaults
    # (asserts 4 categories are listed).
    # KNOWN ISSUE: the test name suggests an empty registry, but the expected
    # behavior is fallback to defaults (empty list treated as unset); the
    # count 4 is hard-coded rather than tied to DEFAULT_ALLOWED_CATEGORIES.
    # Test-specific instance for empty categories
    registry = function_registry.FunctionRegistry(allowed_categories=[])
    self.assertLen(registry.list_categories(), 4)

  @parameterized.named_parameters(
      dict(
          testcase_name="loss_fn_a",
          category="policy_loss_fn",
          name="func_a",
          func=dummy_func_a,
      ),
      dict(
          testcase_name="advantage_a",
          category="advantage_estimator",
          name="func_a",
          func=dummy_func_a,
      ),
  )
  def test_register_and_get_success_default(self, category, name, func):
    # [FUNCTION DOC]
    # Verifies register(category, name) returns a decorator that returns the
    # function unchanged, get retrieves the same object, and list_functions
    # for the category contains only that name.
    decorator = self.registry.register(category, name)
    registered_func = decorator(func)
    self.assertIs(registered_func, func)

    retrieved_func = self.registry.get(category, name)
    self.assertIs(retrieved_func, func)
    self.assertEqual(self.registry.list_functions(category), [name])

  def test_register_duplicate_name_logs_warning(self):
    # [FUNCTION DOC]
    # Verifies registering a second function under an existing name logs a
    # WARNING "Function 'my_func' is already registered in category
    # 'policy_loss_fn'" and the newer function replaces the old one.
    self.registry.register("policy_loss_fn", "my_func")(dummy_func_a)

    with self.assertLogs(level="WARNING") as cm:
      self.registry.register("policy_loss_fn", "my_func")(dummy_func_b)

    self.assertTrue(
        any(
            "Function 'my_func' is already registered in category"
            " 'policy_loss_fn'"
            in output
            for output in cm.output
        )
    )

    self.assertEqual(
        self.registry.get("policy_loss_fn", "my_func"), dummy_func_b
    )

  def test_custom_categories_behavior(self):
    # [FUNCTION DOC]
    # Verifies a registry with custom categories accepts registration in
    # them, while register on a non-allowed category raises ValueError
    # "Invalid category: '<cat>'" (eagerly, even before applying the
    # decorator) and list_functions on an unknown category raises
    # LookupError "No such category: '<cat>'".
    custom_cats = ["custom1", "custom2"]
    # Test-specific instance for custom categories
    registry = function_registry.FunctionRegistry(
        allowed_categories=custom_cats
    )

    # Successful registration and get in custom
    registry.register("custom1", "func_a")(dummy_func_a)
    self.assertIs(registry.get("custom1", "func_a"), dummy_func_a)
    self.assertEqual(registry.list_functions("custom1"), ["func_a"])

    # Default categories should fail
    with self.assertRaisesRegex(
        ValueError, "Invalid category: 'policy_loss_fn'"
    ):
      registry.register("policy_loss_fn", "some_func")(dummy_func_a)

    with self.assertRaisesRegex(ValueError, "Invalid category: 'loss_agg'"):
      registry.register("loss_agg", "some_func")

    with self.assertRaisesRegex(
        LookupError, "No such category: 'advantage_estimator'"
    ):
      registry.list_functions("advantage_estimator")


if __name__ == "__main__":
  absltest.main()
