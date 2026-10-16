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
# Unit tests for tunix.distillation.feature_extraction.sowed_module:
# wrap_model_with_sowed_modules replaces (in place, recursively, including
# inside list attributes) every submodule of the target types with a
# SowedModule wrapper that records its output under SowedModule._SOW_TAG,
# never double-wraps an existing SowedModule, and also recurses into wrapped
# targets; pop_sowed_intermediate_outputs collects and removes the recorded
# outputs as a state tree; unwrap_sowed_modules restores the originals.
"""Unit tests for sowed module wrapping."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.distillation.feature_extraction import sowed_module
from tunix.tests import test_common as tc


class SimpleLayer(nnx.Module):
  # [CLASS DOC]
  # HELPER: leaf nnx layer with a string tag and a 1-element Param of 1.0.
  """A basic NNX layer for testing."""

  def __init__(self, tag: str):
    # [FUNCTION DOC]
    # HELPER: stores tag and creates the [1.0] parameter.
    self.tag = tag  # Just to differentiate instances
    self.param = nnx.Param(jnp.array([1.0]))  # Dummy parameter

  def __call__(self, x: jax.Array) -> jax.Array:
    # [FUNCTION DOC]
    # HELPER: returns x plus the parameter value.
    # Simple operation: add param and return
    return x + self.param.value


class Block(nnx.Module, pytree=False):
  # [CLASS DOC]
  # HELPER: non-pytree nnx module holding layer1, layer2 and a Python list
  # other_layers of two SimpleLayers, applied sequentially.
  """A block containing nested layers."""

  def __init__(self, tag: str):
    # [FUNCTION DOC]
    # HELPER: creates the four SimpleLayers tagged from tag.
    self.tag = tag
    self.layer1 = SimpleLayer(f'{tag}_layer1')
    self.layer2 = SimpleLayer(f'{tag}_layer2')  # Another instance
    self.other_layers = [
        SimpleLayer(f'{tag}_layer3'),
        SimpleLayer(f'{tag}_layer4'),
    ]

  def __call__(self, x: jax.Array) -> jax.Array:
    # [FUNCTION DOC]
    # HELPER: applies layer1, layer2, then each of other_layers in order.
    x = self.layer1(x)
    x = self.layer2(x)
    for layer in self.other_layers:
      x = layer(x)
    return x


class OuterModel(nnx.Module):
  # [CLASS DOC]
  # HELPER: top-level model: block1 -> block2 -> final_layer.
  """A top-level model with nested blocks."""

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: creates two Blocks and a final SimpleLayer.
    self.block1 = Block('block1')
    self.block2 = Block('block2')
    self.final_layer = SimpleLayer('final')

  def __call__(self, x: jax.Array) -> jax.Array:
    # [FUNCTION DOC]
    # HELPER: runs block1, block2 and final_layer sequentially.
    x = self.block1(x)
    x = self.block2(x)
    x = self.final_layer(x)
    return x


class SowedModuleTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for wrapping, sow capture, and unwrapping.

  def test_wrap_model_with_sowed_modules_recursive(self):
    # [FUNCTION DOC]
    # Verifies all nine SimpleLayers (including list items) are replaced by
    # SowedModules wrapping the original objects, while Blocks are untouched.
    # KNOWN ISSUE: dummy_input is created but unused.
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Target SimpleLayer for wrapping
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert all SimpleLayers are wrapped
    self.assertIsInstance(
        model.block1.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.final_layer,
        sowed_module.SowedModule,
    )
    # Check intermediate modules are untouched
    self.assertIsInstance(model.block1, Block)
    self.assertIsInstance(model.block2, Block)
    # Check wrapped models are the originals
    self.assertIs(model.block1.layer1.wrapped_model, original_b1_l1)
    self.assertIs(model.block1.layer2.wrapped_model, original_b1_l2)
    self.assertIs(
        model.block1.other_layers[0].wrapped_model, original_b1_other_layers_0
    )
    self.assertIs(
        model.block1.other_layers[1].wrapped_model, original_b1_other_layers_1
    )
    self.assertIs(model.block2.layer1.wrapped_model, original_b2_l1)
    self.assertIs(model.block2.layer2.wrapped_model, original_b2_l2)
    self.assertIs(
        model.block2.other_layers[0].wrapped_model, original_b2_other_layers_0
    )
    self.assertIs(
        model.block2.other_layers[1].wrapped_model, original_b2_other_layers_1
    )
    self.assertIs(model.final_layer.wrapped_model, original_final)

  def test_wrap_model_with_multiple_target_types(self):
    # [FUNCTION DOC]
    # Verifies wrapping with [SimpleLayer, Block]: Blocks become SowedModules
    # whose wrapped Block's SimpleLayers are themselves wrapped, and all
    # wrapped_model references are the originals.
    # KNOWN ISSUE: dummy_input is created but unused.
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1 = model.block1
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2 = model.block2
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Target SimpleLayer for wrapping
    target_types = [SimpleLayer, Block]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert all SimpleLayers are wrapped
    self.assertIsInstance(
        model.block1.wrapped_model.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block1.wrapped_model.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.layer1,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.other_layers[0],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.block2.wrapped_model.other_layers[1],
        sowed_module.SowedModule,
    )
    self.assertIsInstance(
        model.final_layer,
        sowed_module.SowedModule,
    )
    self.assertIsInstance(model.block1, sowed_module.SowedModule)
    self.assertIsInstance(model.block2, sowed_module.SowedModule)
    # Check wrapped models are the originals
    self.assertIs(
        model.block1.wrapped_model.layer1.wrapped_model, original_b1_l1
    )
    self.assertIs(
        model.block1.wrapped_model.layer2.wrapped_model, original_b1_l2
    )
    self.assertIs(
        model.block1.wrapped_model.other_layers[0].wrapped_model,
        original_b1_other_layers_0,
    )
    self.assertIs(
        model.block1.wrapped_model.other_layers[1].wrapped_model,
        original_b1_other_layers_1,
    )
    self.assertIs(
        model.block2.wrapped_model.layer1.wrapped_model, original_b2_l1
    )
    self.assertIs(
        model.block2.wrapped_model.layer2.wrapped_model, original_b2_l2
    )
    self.assertIs(
        model.block2.wrapped_model.other_layers[0].wrapped_model,
        original_b2_other_layers_0,
    )
    self.assertIs(
        model.block2.wrapped_model.other_layers[1].wrapped_model,
        original_b2_other_layers_1,
    )
    self.assertIs(model.final_layer.wrapped_model, original_final)
    self.assertIs(model.block1.wrapped_model, original_b1)
    self.assertIs(model.block2.wrapped_model, original_b2)

  def test_wrap_model_with_sowed_modules_forward_pass(self):
    # [FUNCTION DOC]
    # Verifies a forward pass through the wrapped model gives the same output as
    # chaining the original layers, and the popped sowed state (pure dict) holds
    # a 1-tuple of each layer's output under the sow tag at the matching path.
    dummy_input = jnp.array([10.0])
    model = OuterModel()
    # Store references to all original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer
    # Wrap the model (all SimpleLayers)
    sowed_module.wrap_model_with_sowed_modules(model, [SimpleLayer])
    # Assertions about captured state after forward pass
    output = model(dummy_input)  # Run forward pass to populate intermediates
    state = sowed_module.pop_sowed_intermediate_outputs(model).to_pure_dict()

    val_b1_l1 = original_b1_l1(dummy_input)  # Input: dummy_input
    val_b1_l2 = original_b1_l2(val_b1_l1)  # Input: output of b1_l1
    val_b1_other_layers_0 = original_b1_other_layers_0(
        val_b1_l2
    )  # Input: output of b1_l2
    val_b1_other_layers_1 = original_b1_other_layers_1(
        val_b1_other_layers_0
    )  # Input: output of b1_other_layers_0
    val_b2_l1 = original_b2_l1(
        val_b1_other_layers_1
    )  # Input: output of b1_other_layers_1
    val_b2_l2 = original_b2_l2(val_b2_l1)  # Input: output of b2_l1
    val_b2_other_layers_0 = original_b2_other_layers_0(
        val_b2_l2
    )  # Input: output of b2_l2
    val_b2_other_layers_1 = original_b2_other_layers_1(
        val_b2_other_layers_0
    )  # Input: output of b2_other_layers_0
    val_final = original_final(
        val_b2_other_layers_1
    )  # Input: output of b2_other_layers_1
    # Final output should match val_final
    tc.assert_close('final_output', output, val_final)

    # Check state structure and values for ALL wrapped layers
    tag = sowed_module.SowedModule._SOW_TAG
    expected_state = {
        'block1': {
            'layer1': {tag: (val_b1_l1,)},
            'layer2': {tag: (val_b1_l2,)},
            'other_layers': {
                0: {tag: (val_b1_other_layers_0,)},
                1: {tag: (val_b1_other_layers_1,)},
            },
        },
        'block2': {
            'layer1': {tag: (val_b2_l1,)},
            'layer2': {tag: (val_b2_l2,)},
            'other_layers': {
                0: {tag: (val_b2_other_layers_0,)},
                1: {tag: (val_b2_other_layers_1,)},
            },
        },
        'final_layer': {tag: (val_final,)},
    }
    jax.tree.map_with_path(
        tc.assert_close,
        state,
        expected_state,
    )

  def test_capture_avoids_double_wrapping(self):
    # [FUNCTION DOC]
    # Verifies a manually wrapped layer is not re-wrapped while other target
    # layers are wrapped.
    model = Block('block')
    original_l1 = model.layer1
    original_l2 = model.layer2
    # Manually wrap one layer first
    manual_wrapper = sowed_module.SowedModule(original_l1)
    model.layer1 = manual_wrapper

    # Run the utility function
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Assert manually wrapped layer wasn't re-wrapped
    self.assertIs(model.layer1, manual_wrapper)
    self.assertIs(model.layer1.wrapped_model, original_l1)
    # Assert the other target layers was wrapped
    self.assertIsInstance(
        model.layer2,
        sowed_module.SowedModule,
    )
    self.assertIs(model.layer2.wrapped_model, original_l2)

  def test_unwrap_sowed_modules_recursive(self):
    # [FUNCTION DOC]
    # Verifies unwrap_sowed_modules restores every original SimpleLayer object,
    # including list entries.
    model = OuterModel()
    # Store references to ALL original SimpleLayer instances
    original_b1_l1 = model.block1.layer1
    original_b1_l2 = model.block1.layer2
    original_b1_other_layers_0 = model.block1.other_layers[0]
    original_b1_other_layers_1 = model.block1.other_layers[1]
    original_b2_l1 = model.block2.layer1
    original_b2_l2 = model.block2.layer2
    original_b2_other_layers_0 = model.block2.other_layers[0]
    original_b2_other_layers_1 = model.block2.other_layers[1]
    original_final = model.final_layer

    # Wrap the model (all SimpleLayers)
    target_types = [SimpleLayer]
    sowed_module.wrap_model_with_sowed_modules(model, target_types)

    # Unwrap the sowed modules
    sowed_module.unwrap_sowed_modules(model)

    # Check if the instances are the originals
    self.assertIs(model.block1.layer1, original_b1_l1)
    self.assertIs(model.block1.layer2, original_b1_l2)
    self.assertIs(model.block1.other_layers[0], original_b1_other_layers_0)
    self.assertIs(model.block1.other_layers[1], original_b1_other_layers_1)
    self.assertIs(model.block2.layer1, original_b2_l1)
    self.assertIs(model.block2.layer2, original_b2_l2)
    self.assertIs(model.block2.other_layers[0], original_b2_other_layers_0)
    self.assertIs(model.block2.other_layers[1], original_b2_other_layers_1)
    self.assertIs(model.final_layer, original_final)


if __name__ == '__main__':
  absltest.main()
