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
# HELPER: small utility module for Gemma 3 multimodal models that scatters
# vision (image soft-token) embeddings into the text embedding sequence at the
# positions flagged by a mask. Pure JAX functions, no side effects.
"""Merge text and vision embeddings."""

import einops
import jax
import jax.numpy as jnp
import jaxtyping


def merge_embeddings(
    *,
    text_embeddings: jaxtyping.ArrayLike,  # (B, L, D)
    vision_embeddings: jaxtyping.ArrayLike,  # (B, N, P, D)
    mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:  # (B, L, D)
  # [FUNCTION DOC]
  # Merges vision embeddings into text embeddings per batch element (vmap over
  # the batch axis of all three inputs of _merge_embeddings_inner).
  # Inputs (keyword-only): text_embeddings [B, L, D], vision_embeddings
  # [B, N_images, P_tokens_per_image, D], mask [B, L] (True at image
  # placeholder positions; expected count per row equals N*P).
  # Output: [B, L, D] with masked positions replaced by the flattened vision
  # embeddings in order.
  """Merge the text and vision embeddings."""
  return jax.vmap(_merge_embeddings_inner, in_axes=(0, 0, 0))(
      text_embeddings, vision_embeddings, mask
  )


def _merge_embeddings_inner(
    text_embeddings: jaxtyping.ArrayLike,  # (L, D)
    vision_embeddings: jaxtyping.ArrayLike,  # (N, P, D)
    mask: jaxtyping.ArrayLike,  # (L)
) -> jaxtyping.ArrayLike:  # (L, D)
  # [FUNCTION DOC]
  # HELPER: unbatched merge. Flattens vision embeddings to [N*P, D]; finds the
  # first N*P indices where mask is true (jnp.nonzero with fixed size, missing
  # entries filled with index 0); scatters the vision rows into those text
  # positions; then restores text position 0 to its original embedding (so the
  # fill index 0 used when there are fewer mask positions does not corrupt it).
  # Output: [L, D].
  # KNOWN ISSUE: if the mask has fewer than N*P true positions, the extra
  # vision rows are all written to index 0 and then discarded; and if a real
  # image token sits at position 0 its vision embedding is overwritten by the
  # original text embedding.
  """`merge_embeddings` without batch dimension."""

  vision_embeddings = einops.rearrange(
      vision_embeddings,
      'num_images num_toks_per_image d -> (num_images num_toks_per_image) d',
  )

  # len(vision_embeddings) == max_num_images * num_tokens_per_image
  target_pos = jnp.nonzero(mask, size=len(vision_embeddings))

  # Save and restore the first position overwritten if there's no MM tokens.
  first_pos = text_embeddings[0]

  merged = text_embeddings.at[target_pos, :].set(vision_embeddings)  # pytype: disable=attribute-error  # jax-arraylike

  merged = merged.at[0].set(first_pos)

  return merged
