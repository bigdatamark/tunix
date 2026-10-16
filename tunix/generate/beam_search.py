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
# Pure-functional beam search helpers used by the native JAX sampler's decode
# loop. All functions operate on JAX arrays and are jit-friendly (no Python
# side effects). Layout convention: the "flattened" batch dimension B equals
# original_batch_size times beam_size, where beams of the same example are
# contiguous (row index = batch_index * beam_size + beam_index).
# Flow: init_batched_beam_state expands the prefill outputs by beam_size; each
# decode step calls beam_search_step which re-ranks and reorders beams; at the
# end finalize_beam_search_state picks beam 0 (the best, since beams are kept
# sorted by score) of each example.

"""Beam search for LLM generation related utils."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import jaxtyping


@flax.struct.dataclass
class _BeamSearchSamplingState:
  # [CLASS DOC]
  # HELPER: immutable flax struct dataclass (a JAX pytree) carrying beam search
  # state inside the main sampling state.
  # Fields: scores, a float32 array of shape [batch_size, beam_size] holding the
  # accumulated log-probability of each beam (pytree leaf); initialized, a static
  # (non-pytree) boolean telling whether at least one beam step has run. Since
  # initialized is static, changing it alters the pytree structure/treedef.
  """The state used by beam search.

  This is intented to be included by the main sampling state.
  """

  # Accumulated scores (log sum probabilities) for each beam. [B, beam_size]
  scores: jnp.ndarray
  # flag to indicate whether the state is initialized or not
  initialized: bool = flax.struct.field(pytree_node=False)


def init_batched_beam_state(
    logits: jax.Array,
    input_token_buffer: jax.Array,
    initial_cache: dict[str, dict[str, jaxtyping.Array]],
    done: jax.Array,
    positions: jax.Array,
    logits_buffer: jax.Array | None,
    beam_size: int,
) -> tuple[_BeamSearchSamplingState, dict[str, Any]]:
  # [FUNCTION DOC]
  # Builds the initial beam state and beam-expanded copies of sampler buffers.
  # Inputs: logits [B, V] for the current (last prefill) step; input_token_buffer
  # [B, L]; initial_cache, a nested dict pytree of KV cache arrays with batch on
  # axis 0; done [B, ...] boolean; positions [B, L]; optional logits_buffer
  # [B, L, V]; beam_size integer.
  # Behavior: batch_size is taken from input_token_buffer axis 0. Every array
  # (logits, token buffer, every cache leaf, done, positions, logits_buffer when
  # not None) is repeated beam_size times along axis 0 with element repetition
  # (each row duplicated consecutively, so beams of an example are adjacent).
  # Output: a tuple of (a _BeamSearchSamplingState with zero float32 scores of
  # shape [batch_size, beam_size] and initialized False, and a dict with keys
  # logits, token_buffer, cache, done, positions, logits_buffer where
  # logits_buffer is None if the input was None).
  # No exceptions raised explicitly; no side effects.
  """Intializes the beam search sampling state.

  In order to support beam search, we need to expand the input other states to
  support beam search by repeating the batch size by beam size.

  Args:
    logits: The logits of the current step. [B, vocab_size]
    input_token_buffer: The input token buffer with max length and padding if
      needed. [B, L]
    initial_cache: The initial cache for original batch size.
    done: The done array for original batch size. [B, 1]
    positions: The original positions of the input tokens. [B, L]
    logits_buffer: The logits buffer for original batch size. [B, L, V]
    beam_size: The number of beams.

  Returns:
    The beam search sampling state.
  """
  batch_size = input_token_buffer.shape[0]

  caches = jax.tree.map(
      lambda x: jnp.repeat(x, beam_size, axis=0), initial_cache
  )

  return _BeamSearchSamplingState(
      scores=jnp.zeros((batch_size, beam_size), dtype=jnp.float32),
      initialized=False,
  ), {
      "logits": jnp.repeat(logits, beam_size, axis=0),
      "token_buffer": jnp.repeat(input_token_buffer, beam_size, axis=0),
      "cache": caches,
      "done": jnp.repeat(done, beam_size, axis=0),
      "positions": jnp.repeat(positions, beam_size, axis=0),
      "logits_buffer": (
          jnp.repeat(logits_buffer, beam_size, axis=0)
          if logits_buffer is not None
          else None
      ),
  }


def beam_search_step(
    logits: jax.Array,
    done: jax.Array,
    token_buffer: jax.Array,
    cache: dict[str, dict[str, jaxtyping.Array]],
    logits_buffer: jax.Array | None,
    state: _BeamSearchSamplingState,
    pad_token_id: jax.Array,
    decoding_step: int,
    logprobs_buffer: jax.Array | None = None,
) -> tuple[_BeamSearchSamplingState, dict[str, Any]]:
  # [FUNCTION DOC]
  # Performs one beam search decoding step over flattened beams.
  # Inputs: logits [batch_size*beam_size, V]; done flags for each flattened beam;
  # token_buffer [batch*beam, L]; cache pytree with flattened batch on axis 0;
  # optional logits_buffer; the previous beam state; pad_token_id; decoding_step
  # (index of the last written token; the new token is written at index
  # decoding_step + 1); optional logprobs_buffer [batch*beam, L].
  # Algorithm:
  # 1. Reshape logits to [batch, beam, V] and compute log_softmax over V.
  # 2. If state is not initialized, all beams are identical copies, so candidates
  #    are only beam 0's log-probs, shape [batch, V]. Otherwise candidate scores
  #    are previous beam scores plus log-probs, with candidates from already
  #    finished beams set to negative infinity, flattened to [batch, beam*V].
  # 3. Append, per example, the scores of finished beams (negative infinity for
  #    unfinished) so finished beams can survive unchanged; take top beam_size of
  #    the concatenation (jax top_k; result is sorted descending).
  # 4. For each pick, if its flat index is below the candidate width it is an
  #    extension: source beam = index divided by V, token = index modulo V.
  #    Otherwise it is a kept finished beam: source beam = index minus candidate
  #    width, token = pad_token_id.
  # 5. Reorder token_buffer, logits_buffer, logprobs_buffer, done and every cache
  #    leaf by gathering rows at batch_index*beam_size + source_beam (nested
  #    helper). Write the chosen token at column decoding_step+1 of the token
  #    buffer; write the chosen token's log-prob (looked up in log_probs by source
  #    beam and token) at the same column of the logprobs buffer if present.
  # Output: tuple of (new state with the top-k scores and initialized True, dict
  # with token_buffer, cache, done, logits_buffer, logprobs_buffer). done is only
  # reordered, not updated with end-of-sequence detection (caller must do that).
  # No side effects.
  # KNOWN ISSUE: in the uninitialized branch, done flags are ignored and
  # candidate width is V (not beam*V), so a kept finished beam index offset is
  # computed relative to V; any finished beam at init would map to a beam index
  # relative to that width, which is consistent but means finished-at-prefill
  # beams still receive extension candidates.
  # KNOWN ISSUE: the log-prob lookup for kept finished beams uses pad_token_id as
  # the token index, recording the pad token's log-prob instead of zero.
  # KNOWN ISSUE: the logits_buffer is reordered but no logits for the new step
  # are written into it here.
  """Beam search step.

  In beam search, at each step, we generate possible next tokens for each of
  previous candidates. This gives us beam_size_prev * vocab_size potential new
  sequences. Then we evaluate these new sequences scores and select the top
  beam_size sequences to continue with in the next step. The input B (batch
  size)
  is actually the original batch size * beam_size.

  Args:
    logits: The logits of the current step. [B, vocab_size]
    done: The done array of the previous step. [B]
    token_buffer: The token buffer of the previous step. [B, L]
    cache: The cache of the current step.
    logits_buffer: The logits buffer of the current step. [B, L, vocab_size]
    state: The previous beam search sampling state, which contains the last step
      scores and some constant parameters.
    pad_token_id: The pad token id used for end of sequence padding.
    decoding_step: The previous decoding step.

  Returns:
    The new beam search sampling state along with the updated token buffer,
    cache and done array because beam selection has changed the history with the
    new logits calcualtion. The returned results are sorted.
  """
  batch_size, beam_size = state.scores.shape
  vocab_size = logits.shape[-1]
  logits = logits.reshape((batch_size, beam_size, -1))
  # Calculate candidate scores
  log_probs = jax.nn.log_softmax(
      logits, axis=-1
  )  # (batch_size, beam_size, vocab_size)
  if not state.initialized:
    # the beam information is duplicated, just select the first one from every
    # batch to dedup.
    candidate_scores_per_item = log_probs[:, 0, :].reshape(
        batch_size, vocab_size
    )
  else:
    # Add current beam scores to the new log probabilities
    candidate_scores = (
        state.scores[:, :, None] + log_probs
    )  # (batch_size, beam_size, vocab_size)

    # For finished beams, set the candidate scores to -inf
    candidate_scores = jnp.where(
        done.reshape(batch_size, beam_size)[:, :, None],
        -jnp.inf,
        candidate_scores,
    )

    candidate_scores_per_item = candidate_scores.reshape(
        batch_size, beam_size * vocab_size
    )

  # don't forget finished beams
  scores_of_finished_beams = jnp.where(
      done.reshape(batch_size, beam_size), state.scores, -jnp.inf
  )
  # (batch_size, beam_size * vocab_size + beam_size)
  combined_scores = jnp.concatenate(
      [candidate_scores_per_item, scores_of_finished_beams], axis=1
  )

  # top_k_flat_indices in shape of (batch_size, beam_size)
  new_scores, top_k_flat_indices = jax.lax.top_k(combined_scores, k=beam_size)

  # Map flat indices (local to beam_size*vocab_size + finished_beam_size)
  # back to(source_beam_idx, next_token_idx). It's used to idenfity the picked
  # beam and new token. Given the current candidates also include the finished
  # beams, so additional check is needed here:
  # If indices < candidate_scores_per_item.shape[-1], candidates coming from
  # new extension candidates, otherwise, candidates comfing from
  # finished beams.
  is_extension_candidate = (
      top_k_flat_indices < candidate_scores_per_item.shape[-1]
  )
  ext_source_beam_indices = top_k_flat_indices // vocab_size
  ext_next_token_indices = top_k_flat_indices % vocab_size

  done_source_beam_indices = top_k_flat_indices - (
      candidate_scores_per_item.shape[-1]
  )
  done_next_token_indices = jnp.full(
      (batch_size, beam_size), pad_token_id, dtype=jnp.int32
  )  # Use PAD as placeholder

  source_beam_indices = jnp.where(
      is_extension_candidate, ext_source_beam_indices, done_source_beam_indices
  )

  next_token_indices = jnp.where(
      is_extension_candidate, ext_next_token_indices, done_next_token_indices
  )
  # Construct the new beam search state
  batch_indices = jnp.arange(batch_size)[:, None].repeat(
      beam_size, axis=1
  )  # (batch_size, beam_size)
  gather_indices = jnp.stack(
      [batch_indices, source_beam_indices], axis=-1
  )  # (batch_size, beam_size, 2)
  # [..., 0] is batch index, [..., 1] is beam index

  # Gather function for arrays with shape (batch_size, beam_size_prev, ...)
  def gather_beams_by_indices(arr, indices):
    # [FUNCTION DOC]
    # HELPER: gathers rows of a flattened [batch*beam, ...] array. Input indices
    # has shape [batch, beam, 2] with (batch index, source beam index) pairs.
    # Computes flat row index batch_index * beam_size + beam_index (beam_size from
    # the enclosing scope), flattens it in row-major order and returns the array
    # indexed by those rows, shape [batch*beam, ...].
    # expected arr shape (batch_size * beam_size, ...)
    # indices shape (batch_size, beam_size, 2)
    # output shape (batch_size *beam_size, ...) with selected beam index
    batch_indices = indices[:, :, 0]
    beam_indices = indices[:, :, 1]
    flat_indices = (batch_indices * beam_size + beam_indices).ravel()
    return arr[flat_indices]

  # Gather tokens, lengths, and finished status
  source_beams_tokens = gather_beams_by_indices(
      token_buffer, gather_indices
  )  # (batch_size * beam_size, max_total_len)

  source_logits_buffer = (
      gather_beams_by_indices(logits_buffer, gather_indices)
      if logits_buffer is not None
      else None
  )  # (batch_size * beam_size, max_total_len, vocab_size)

  source_logprobs_buffer = (
      gather_beams_by_indices(logprobs_buffer, gather_indices)
      if logprobs_buffer is not None
      else None
  )  # (batch_size * beam_size, max_total_len)

  source_beams_done = gather_beams_by_indices(done, gather_indices)

  new_cache = jax.tree.map(
      lambda x: gather_beams_by_indices(x, gather_indices), cache
  )

  source_beams_tokens = source_beams_tokens.at[:, decoding_step + 1].set(
      next_token_indices.reshape(batch_size * beam_size)
  )

  # Selected log probabilities for the chosen tokens
  selected_log_probs = log_probs[
      jnp.arange(batch_size)[:, None],
      source_beam_indices,
      next_token_indices,
  ]

  if source_logprobs_buffer is not None:
    source_logprobs_buffer = source_logprobs_buffer.at[
        :, decoding_step + 1
    ].set(selected_log_probs.reshape(batch_size * beam_size))

  return _BeamSearchSamplingState(
      scores=new_scores,
      initialized=True,
  ), {
      "token_buffer": source_beams_tokens,
      "cache": new_cache,
      "done": source_beams_done,
      "logits_buffer": source_logits_buffer,
      "logprobs_buffer": source_logprobs_buffer,
  }


def finalize_beam_search_state(
    beam_search_state: _BeamSearchSamplingState,
    token_buffer: jax.Array,
    logits_buffer: jax.Array | None,
    logprobs_buffer: jax.Array | None = None,
) -> dict[str, Any]:
  # [FUNCTION DOC]
  # Selects the final output after beam search finishes.
  # Inputs: the beam state (only its scores shape [batch, beam] is used);
  # token_buffer [batch*beam, L]; optional logits_buffer; optional
  # logprobs_buffer [batch*beam, L].
  # Behavior: reshapes each buffer to [batch, beam, -1] and keeps beam 0 of each
  # example (beams are sorted best-first by beam_search_step).
  # Output: dict with token_buffer [batch, L], logits_buffer (or None),
  # logprobs_buffer (or None). No side effects.
  # KNOWN ISSUE: logits_buffer of shape [batch*beam, L, V] is reshaped to
  # [batch, beam, L*V], so the returned logits are flattened to [batch, L*V]
  # rather than [batch, L, V].
  """Finalize the beam search sampling state.

  This function is called when decoding is done. Beams are sorted. So the
  final output is the first beam of each batch.

  Args:
    beam_search_state: The beam search sampling state. It contains the useful
      information for this function, for example, batch_size, beam_size, etc.
    token_buffer: The token buffer of the previous step. [B, L]
    logits_buffer: The logits buffer of the current step. [B, L, vocab_size]

  Returns:
    The updated token buffer and logits buffer by choosing the top beam result.
  """
  batch_size, beam_size = beam_search_state.scores.shape
  token_buffer = token_buffer.reshape((batch_size, beam_size, -1))
  logits_buffer = (
      logits_buffer.reshape((batch_size, beam_size, -1))
      if logits_buffer is not None
      else None
  )
  logprobs_buffer = (
      logprobs_buffer.reshape((batch_size, beam_size, -1))
      if logprobs_buffer is not None
      else None
  )
  return {
      "token_buffer": token_buffer[:, 0, ...],
      "logits_buffer": (
          logits_buffer[:, 0, ...] if logits_buffer is not None else None
      ),
      "logprobs_buffer": (
          logprobs_buffer[:, 0, ...] if logprobs_buffer is not None else None
      ),
  }

