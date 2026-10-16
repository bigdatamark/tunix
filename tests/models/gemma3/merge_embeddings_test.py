# Copyright 2026 Google LLC
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
# Unit tests for tunix.models.gemma3.merge_embeddings.merge_embeddings, which scatters flattened
# vision embeddings (shape [batch, images, tokens_per_image, dim]) into text embeddings
# ([batch, seq, dim]) at positions where mask is nonzero, in order, while keeping position 0 (BOS)
# from the text embeddings. Inputs: small synthetic arrays. Outputs: test results. No side effects.

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma3 import merge_embeddings as merge_embeddings_lib


class MergeEmbeddingsTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests vision/text embedding merging for single, batched, empty-mask and BOS-preservation cases.

  def test_merge_embeddings(self):
    # [FUNCTION DOC]
    # Verifies that for a single sequence, the four masked positions are replaced by vision values
    # (-1) and all other positions keep their text embeddings.
    text_embeddings = jnp.arange(20, dtype=jnp.float32).reshape(1, 10, 2)
    vision_embeddings = jnp.full((1, 2, 2, 2), -1.0, dtype=jnp.float32)
    mask = jnp.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 0]])
    merged = merge_embeddings_lib.merge_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings,
        mask=mask,
    )
    expected = jnp.array(
        [[
            [0, 1],
            [-1, -1],
            [4, 5],
            [-1, -1],
            [8, 9],
            [-1, -1],
            [12, 13],
            [-1, -1],
            [16, 17],
            [18, 19],
        ]],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(merged, expected)

  def test_merge_embeddings_batch(self):
    # [FUNCTION DOC]
    # Verifies merging operates per batch element, with differing mask patterns per row.
    text_embeddings = jnp.arange(40, dtype=jnp.float32).reshape(2, 10, 2)
    vision_embeddings = jnp.full((2, 2, 2, 2), -1.0, dtype=jnp.float32)
    mask = jnp.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    ])
    merged = merge_embeddings_lib.merge_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings,
        mask=mask,
    )
    expected = jnp.array(
        [
            [
                [0, 1],
                [-1, -1],
                [4, 5],
                [-1, -1],
                [8, 9],
                [-1, -1],
                [12, 13],
                [-1, -1],
                [16, 17],
                [18, 19],
            ],
            [
                [20, 21],
                [22, 23],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [32, 33],
                [34, 35],
                [36, 37],
                [38, 39],
            ],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(merged, expected)

  def test_merge_embeddings_without_vision_positions(self):
    # [FUNCTION DOC]
    # Verifies that an all-zero mask returns the text embeddings unchanged.
    text_embeddings = jnp.arange(20, dtype=jnp.float32).reshape(1, 10, 2)
    vision_embeddings = jnp.full((1, 2, 2, 2), -1.0, dtype=jnp.float32)
    mask = jnp.zeros((1, 10), dtype=jnp.int32)
    merged = merge_embeddings_lib.merge_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings,
        mask=mask,
    )
    np.testing.assert_array_equal(merged, text_embeddings)

  def test_merge_embeddings_bos_preserved(self):
    # [FUNCTION DOC]
    # Verifies that a mask set at position 0 does not overwrite the BOS embedding: only position 1 is
    # replaced by a vision value and position 0 keeps its text value.
    text_embeddings = jnp.arange(10, dtype=jnp.float32).reshape(1, 5, 2)
    vision_embeddings = jnp.full((1, 1, 2, 2), -1.0, dtype=jnp.float32)
    mask = jnp.array([[1, 1, 0, 0, 0]])
    merged = merge_embeddings_lib.merge_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings,
        mask=mask,
    )
    # BOS at pos 0 should be preserved.
    expected = jnp.array(
        [[
            [0, 1],
            [-1, -1],
            [4, 5],
            [6, 7],
            [8, 9],
        ]],
        dtype=jnp.float32,
    )
    np.testing.assert_array_equal(merged, expected)


if __name__ == "__main__":
  absltest.main()
