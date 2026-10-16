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
# Unit tests for tunix.models.gemma3.utils.get_attention_mask, which builds a boolean causal
# attention mask of shape [batch, seq, seq] from token ids. Padding tokens (utils._PADDING_ID) are
# never attended to; contiguous image placeholder tokens (id 219 here) attend bidirectionally
# within their block; an optional precomputed inputs mask further removes columns.
# Inputs: small hand-built token arrays. Outputs: absltest results. No side effects.

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma3 import utils


_TOKEN_PLACEHOLDER = 219


class UtilsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests attention mask construction for text-only, multimodal and precomputed-mask inputs.

  def test_get_attention_mask_not_multimodal(self):
    # [FUNCTION DOC]
    # Verifies text-only tokens with two trailing pads give a lower-triangular causal mask over the
    # three real tokens, where pad rows attend to all real tokens and no pad column is ever attended.
    tokens = jnp.array([[1, 2, 3, utils._PADDING_ID, utils._PADDING_ID]])
    attention_mask = utils.get_attention_mask(
        tokens, token_placeholder_id=_TOKEN_PLACEHOLDER
    )

    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_get_attention_mask_multimodal(self):
    # [FUNCTION DOC]
    # Verifies that two adjacent placeholder (image) tokens attend to each other bidirectionally
    # (both rows see both placeholder columns) while the rest stays causal and padding is masked.
    tokens = jnp.array([[
        1,
        2,
        _TOKEN_PLACEHOLDER,
        _TOKEN_PLACEHOLDER,
        3,
        utils._PADDING_ID,
    ]])
    attention_mask = utils.get_attention_mask(
        tokens, token_placeholder_id=_TOKEN_PLACEHOLDER
    )

    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)

  def test_get_attention_mask_precomputed_mask(self):
    # [FUNCTION DOC]
    # Verifies that when inputs_mask is given, columns with a zero in inputs_mask (position 1 and the
    # pads) are masked out of the causal mask for every row.
    tokens = jnp.array([[1, 2, 3, utils._PADDING_ID, utils._PADDING_ID]])
    inputs_mask = jnp.array([[1, 0, 1, 0, 0]])
    attention_mask = utils.get_attention_mask(
        tokens, inputs_mask=inputs_mask, token_placeholder_id=_TOKEN_PLACEHOLDER
    )

    expected_attention_mask = jnp.array(
        [[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ]],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(attention_mask, expected_attention_mask)


if __name__ == '__main__':
  absltest.main()
