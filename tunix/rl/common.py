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
# Shared RL helpers: a re-iterating mini-batch iterable over rollout batches,
# the TrainExample struct passed from rollout to the loss, KL estimators,
# log-softmax gathering, model forward helpers that compute per-token
# log-probabilities or per-token scores from prompt and completion tokens
# (including packed-sequence support), completion-mask builders, padding, and
# loss aggregation across tokens and sequences. Re-exports
# make_causal_attn_mask and build_positions_from_mask from tunix.sft.utils.
"""Common RL helper classes and functions."""

from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Iterable

import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jax.tree_util as jtu
import numpy as np
from tunix.sft import utils

make_causal_attn_mask = utils.make_causal_attn_mask
build_positions_from_mask = utils.build_positions_from_mask


class RepeatIterable(Iterable[Any]):
  # [CLASS DOC]
  # Iterable over a list of rollout batches (each a pytree of arrays sharing a
  # leading batch dimension). On each pass over the data (repeated "repeat"
  # times) every rollout batch is optionally shuffled and split into
  # consecutive mini-batches of mini_batch_size rows. State: data, repeat,
  # mini_batch_size (public, may be None meaning whole batch), shuffle flag,
  # PRNG key (default key 0), private effective mini-batch size.
  """An iterable that processes a list of rollout batches.

  For each rollout batch, it shuffles its contents, slices it into mini-batches,
  and yields them sequentially before moving to the next rollout batch. This
  entire process is repeated for a specified number of epochs.
  """

  def __init__(
      self,
      data: list[Any],
      repeat: int,
      mini_batch_size: int | None = None,
      shuffle: bool = False,
      key: jnp.ndarray | None = None,
  ):
    # [FUNCTION DOC]
    # Stores data, repeat, mini_batch_size, shuffle, and key (a fresh PRNGKey of
    # 0 when None). Copies mini_batch_size into a private effective size.
    self._data = data

    self.repeat = repeat
    self.mini_batch_size = mini_batch_size

    self.shuffle = shuffle
    self.key = key if key is not None else jax.random.PRNGKey(0)

    # Maintain a private, mutable `mini_batch_size`, for simpler code.
    self._mini_batch_size = mini_batch_size

  def _shuffle_and_slice_one_batch(self, rollout_batch: Any):
    # [FUNCTION DOC]
    # HELPER: Generator for one rollout batch. Batch size is the leading dim of
    # the first flattened leaf. If mini_batch_size is None the effective size
    # becomes the full batch size. Raises ValueError if the batch size is not
    # divisible by the effective size. If shuffling, splits the stored key,
    # keeps the first half as the new key (mutates state) and uses it to draw a
    # permutation; otherwise identity order. Yields, for each consecutive slice
    # of the index order, a pytree with every leaf gathered at those indices.
    """A generator that shuffles and slices a single rollout batch."""
    leaves, _ = jtu.tree_flatten(rollout_batch)
    rollout_batch_size = leaves[0].shape[0]

    if self.mini_batch_size is None:
      self._mini_batch_size = rollout_batch_size

    if rollout_batch_size % self._mini_batch_size != 0:
      raise ValueError(
          "Each rollout batch's size must be divisible by `mini_batch_size`."
      )
    num_mini_batches = rollout_batch_size // self._mini_batch_size

    # Shuffle indices.
    if self.shuffle:
      self.key, _ = jax.random.split(self.key)
      shuffled_indices = jax.random.permutation(self.key, rollout_batch_size)
    else:
      shuffled_indices = jnp.arange(rollout_batch_size)

    # Slice the rollout batch into mini-batches.
    for i in range(num_mini_batches):
      start = i * self._mini_batch_size
      end = start + self._mini_batch_size
      batch_indices = shuffled_indices[start:end]

      mini_batch = jtu.tree_map(
          lambda leaf, indices=batch_indices: leaf[indices], rollout_batch
      )
      yield mini_batch

  def __iter__(self):
    # [FUNCTION DOC]
    # Generator: for each of repeat passes, for each stored rollout batch,
    # yields all its mini-batches via _shuffle_and_slice_one_batch.
    """The main generator for the iterable."""
    for _ in range(self.repeat):
      for rollout_batch in self._data:
        yield from self._shuffle_and_slice_one_batch(rollout_batch)


@flax.struct.dataclass(frozen=True)
class TrainExample:
  # [CLASS DOC]
  # Frozen flax struct dataclass (a pytree) holding one training batch:
  # prompt_ids and prompt_mask (batch by prompt length, left padded),
  # completion_ids and completion_mask (batch by completion length),
  # advantages (per sequence or per token), ref_per_token_logps and
  # old_per_token_logps (optional, batch by completion length), optional
  # segment_ids and segment_positions for sequence packing, and optional
  # sampler_is_weights (detached per-token truncated importance-sampling
  # weights applied to the policy-gradient loss before aggregation).
  prompt_ids: jax.Array
  prompt_mask: jax.Array
  completion_ids: jax.Array
  completion_mask: jax.Array
  advantages: jax.Array
  ref_per_token_logps: jax.Array | None
  old_per_token_logps: jax.Array | None
  segment_ids: jax.Array | None = None
  segment_positions: jax.Array | None = None
  # Truncated importance-sampling correction weights for off-policy
  # correction between the rollout sampler and the trainer. Per-token,
  # detached, multiplied into the policy-gradient loss BEFORE aggregation
  # to dampen positions where the trainer's recomputed log-probability
  # diverges from the rollout sampler's. ``None`` disables the correction.
  sampler_is_weights: jax.Array | None = None


def compute_kl_divergence(
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    method: str = "low_var_kl",
    clamp_value: float | None = None,
) -> jax.Array:
  # [FUNCTION DOC]
  # Inputs: policy per-token log-probs, reference per-token log-probs, method,
  # optional clamp. Casts both to float32. Method "kl": logp minus ref_logp.
  # "mse_kl": half the squared difference. "low_var_kl": with diff = ref_logp
  # minus logp, exp(diff) minus diff minus 1. Any other method raises
  # ValueError. If clamp_value is set, clips the result to the symmetric range.
  # Returns the per-token KL array (same shape as inputs).
  """Compute per token KL divergence between trained and reference policy.

  Based on `method`, we compute one of three kinds of KL divergence:
  - "kl": Unbiased, high-variance estimator. Simple Forward KL:
    `logp - ref_logp`.
  - "mse_kl": Biased, low-variance estimator. Squared log-difference:
    `0.5 * (logp - ref_logp)^2`.
  - "low_var_kl": Unbiased, low-variance estimator. J. Schulman low-variance
    approx: `(r - 1) - log r`, where `r = q/p = exp(ref_logp - logp)`.

  Args:
    per_token_logps: Per token log probabilities from the trained policy.
    ref_per_token_logps: Per token log probabilities from the reference policy.
    method: KL penalty method. Defaults to "low_var_kl".
    clamp_value: Optional symmetric clamp applied to the returned KL,
      i.e. `clip(kl, -clamp_value, +clamp_value)`. `None`
      (default) disables the clamp and preserves prior behavior. Set to a
      positive float (e.g. `10000.0`) to bound rare outliers — useful when
      the trained policy briefly drifts far from the reference and the
      `low_var_kl` estimator's `exp(diff)` term can overflow fp32 / saturate
      bf16.

  Returns:
    KL divergence.
  """
  per_token_logps = per_token_logps.astype(jnp.float32)
  if ref_per_token_logps is not None:
    ref_per_token_logps = ref_per_token_logps.astype(jnp.float32)

  if method == "kl":
    kl = per_token_logps - ref_per_token_logps
  elif method == "mse_kl":
    kl = 0.5 * jnp.square(per_token_logps - ref_per_token_logps)
  elif method == "low_var_kl":
    diff = ref_per_token_logps - per_token_logps
    kl = jnp.exp(diff) - diff - 1
  else:
    raise ValueError(
        "`method` must be one of 'kl', 'mse_kl', 'low_var_kl'. Received:"
        f" {method}"
    )

  if clamp_value is not None:
    kl = jnp.clip(kl, -clamp_value, clamp_value)
  return kl


def selective_log_softmax(logits: jax.Array, input_ids: jax.Array) -> jax.Array:
  # [FUNCTION DOC]
  # Inputs: logits (..., vocab) and integer ids (...). Gathers the logit of
  # each id along the last axis (as float32) and subtracts the float32
  # log-sum-exp over the vocab. Returns log-probabilities of the given ids.
  """Compute the log probablity based on the input ids.

  Args:
    logits: Logits from the model.
    input_ids: Input ids to get logits.

  Returns:
    Selected log probabilities.
  """
  target_logits = (
      jnp.take_along_axis(logits, input_ids[..., None], axis=-1)
      .squeeze(-1)
      .astype(jnp.float32)
  )
  normalizer = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
  return target_logits - normalizer


# TODO(tsbao): remove this once old callsite is cleaned up.
@nnx.jit(static_argnames=("logits_to_keep",))
def get_per_token_logps(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    logits_to_keep: int,
    images: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  # [FUNCTION DOC]
  # Legacy, jitted with logits_to_keep static. Runs the model on input tokens
  # with given positions, attention mask, no cache and optional images. Keeps
  # logits for the positions that predict the last logits_to_keep tokens
  # (slice from -logits_to_keep-1 to -1) and returns selective_log_softmax of
  # them against the last logits_to_keep input tokens (batch by
  # logits_to_keep).
  # KNOWN ISSUE: the return annotation allows a tuple but only an array is
  # returned.
  """Computes the per-token log probabilities."""
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(
      input_tokens,
      positions=positions,
      attention_mask=attn_mask,
      cache=None,
      **kwargs,
  )
  logits = logits[:, -logits_to_keep - 1 : -1, :]
  input_tokens = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens)
  return per_token_logps


# TODO(abheesht): This is computed 4 times - twice in `compute_per_token_logps`
# and twice in `compute_score`. We can factor this out and compute it just once.
@partial(jax.jit, static_argnames=("pad_id", "eos_id"))
def process_ids(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
):
  # [FUNCTION DOC]
  # Jitted (pad_id and eos_id static). Concatenates prompt and completion tokens
  # along the sequence axis. Packed mode (segment_ids given): raises ValueError
  # if segment_positions is None; otherwise returns the concatenated ids,
  # segment_positions, no attention mask and no derived segment ids. Unpacked
  # mode: non-pad masks for prompt and completion are concatenated; positions
  # are built from the mask (tunix.sft.utils), a causal attention mask is built
  # from the mask, and an int32 copy of the mask serves as per-position segment
  # ids. Returns a tuple of (ids, positions, attention mask, segment ids).
  # KNOWN ISSUE: eos_id is unused; the docstring documents a completion_mask
  # argument that does not exist.
  """Processes prompt and completion ids.

  Args:
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    completion_mask: optional attention weights mapping completion sequences.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
  """
  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)

  if segment_ids is not None:
    # Positions are either provided or must be computed correctly (assumed
    # provided for packed).
    if segment_positions is None:
      raise ValueError(
          "segment_positions must be explicitly provided for packed sequences. "
      )
    attn_mask = None  # Relies on segment_ids inside the model
    # Packed callers supply their own segment_ids that already separate
    # distinct documents in the buffer; no need for an additional padding
    # mask here.
    return prompt_completion_ids, segment_positions, attn_mask, None

  prompt_mask = prompt_tokens != pad_id
  completion_mask = completion_tokens != pad_id

  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)

  # 1-D per-position non-pad mask for the full prompt+completion sequence.
  # Used as ``segment_ids`` by attention kernels that cannot consume the 2-D
  # ``attn_mask`` directly (e.g. pallas splash attention takes only a causal
  # mask kernel-side and respects per-position segment ids to suppress
  # cross-segment attention). With pad=0 and real=1, a real position never
  # attends to a pad position regardless of where padding lives in the
  # sequence (typically left-padded for prompt-side alignment).
  input_seg_ids = prompt_completion_mask.astype(jnp.int32)

  return prompt_completion_ids, positions, attn_mask, input_seg_ids


@partial(
    jax.jit,
    static_argnames=(
        "pad_id",
        "eos_id",
        "stop_gradient",
        "return_logits",
        "temperature",
    ),
)
def compute_per_token_logps(
    graphdef,
    state,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    images: jax.Array | None = None,
    stop_gradient: bool = True,
    return_logits: bool = False,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    temperature: float = 1.0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  # [FUNCTION DOC]
  # Jitted (pad_id, eos_id, stop_gradient, return_logits, temperature static).
  # Merges graphdef and state into a model; prepares ids/positions/masks via
  # process_ids. Builds model kwargs with positions, no cache and the attention
  # mask. Inspects the model's call signature: if it accepts segment_ids (or
  # arbitrary keyword args), passes the caller's segment_ids, else the derived
  # non-pad segment ids. Adds images if given. Runs the model and takes the
  # logits. Number of predicted positions kept: full length minus one in packed
  # mode, otherwise completion length; slices logits shifted by one so each
  # position predicts the next token. Divides logits by temperature unless it
  # is 0 or 1. Computes selective log-softmax against the kept tokens. In packed
  # mode left-pads the log-probs (and logits when returned) by one zero column
  # so outputs align with tokens. Optionally stops gradients on both. Returns
  # log-probs (batch by completion length, or batch by full length when
  # packed), or a tuple of log-probs and logits when return_logits is true.
  """Computes the per-token log probabilities.

  Args:
    graphdef: Flax NNX GraphDef.
    state: Flax NNX State.
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    images: optional images array.
    stop_gradient: whether to stop gradient.
    return_logits: whether to return logits.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
    temperature: temperature used for rollout.

  Returns:
    per_token_logps: jax.Array token-level logarithmic values.
      Without sequence packing, returns log probs for completion tokens only,
      with shape `[B, completion_len]`. With sequence packing, returns log
      probs for the full packed sequence padded out (since prompts and
      completions of multiple sequences are concatenated), with shape `[B,
      FullSeqLen]`.
    logits: optional output tensor associated directly when tracking
    derivatives.
  """
  model = nnx.merge(graphdef, state)
  input_tokens, calculated_positions, attn_mask, input_seg_ids = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
  )

  model_kwargs = {
      "positions": calculated_positions,
      "cache": None,
      "attention_mask": attn_mask,
  }
  # Pass through any segment ids so the model's attention kernel can respect
  # them only if the model signature accepts it: caller-provided packing ids take
  # precedence; otherwise we pass the per-position non-pad mask derived in
  # ``process_ids`` so flash-attention variants that lack a separate
  # padding-mask input still skip pad positions.
  import inspect  # pylint: disable=g-import-not-at-top
  try:
    sig = inspect.signature(model.__call__)
    has_segment_ids = ("segment_ids" in sig.parameters) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
  except Exception:
    has_segment_ids = False

  if has_segment_ids:
    if segment_ids is not None:
      model_kwargs["segment_ids"] = segment_ids
    elif input_seg_ids is not None:
      model_kwargs["segment_ids"] = input_seg_ids
  if images is not None:
    model_kwargs["images"] = images

  logits, _ = model(input_tokens, **model_kwargs)

  if segment_ids is not None:
    # Packed Mode: Evaluate the full sequence (mixed prompts + completions).
    # Since predicting token[i] requires logit[i-1], we skip the first token.
    # This shrinks the output shape to [Batch, FullSeqLen - 1]
    logits_to_keep = input_tokens.shape[1] - 1
  else:
    logits_to_keep = completion_tokens.shape[1]

  logits = logits[:, -logits_to_keep - 1 : -1, :]
  if temperature != 0.0 and temperature != 1.0:
    logits /= temperature

  input_tokens_to_keep = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens_to_keep)

  if segment_ids is not None:
    # Pad the front with 0.0 to make shape back to [Batch, FullSeqLen]. This
    # aligns indices (logp[i] matches token[i]) and avoids mask slicing downstream.
    per_token_logps = jnp.pad(
        per_token_logps, ((0, 0), (1, 0)), constant_values=0.0
    )
    if return_logits:
      logits = jnp.pad(logits, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)

  if stop_gradient:
    per_token_logps = jax.lax.stop_gradient(per_token_logps)
    logits = jax.lax.stop_gradient(logits)

  if return_logits:
    return per_token_logps, logits
  else:
    return per_token_logps


@nnx.jit(static_argnames=("pad_id", "eos_id", "stop_gradient"))
def compute_score(
    model,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
):
  # [FUNCTION DOC]
  # nnx-jitted (pad_id, eos_id, stop_gradient static). Runs a scalar-head model
  # (reward or critic) over concatenated prompt and completion tokens prepared
  # by process_ids. Packed mode passes the caller's segment_ids; otherwise
  # passes the causal attention mask and derived segment ids. Takes the first
  # element if the model returns a tuple, squeezes the trailing size-1 axis to
  # get batch by total length scores, optionally stops gradients, and returns
  # them.
  # KNOWN ISSUE: segment ids are always passed to the model without checking
  # whether its signature accepts them (unlike compute_per_token_logps).
  """Computes reward using the provided model."""
  (
      prompt_completion_ids,
      calculated_positions,
      attn_mask,
      input_seg_ids,
  ) = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
  )

  model_kwargs = {"positions": calculated_positions, "cache": None}
  if segment_ids is not None:
    model_kwargs["segment_ids"] = segment_ids
  else:
    model_kwargs["attention_mask"] = attn_mask
    if input_seg_ids is not None:
      model_kwargs["segment_ids"] = input_seg_ids

  out = model(prompt_completion_ids, **model_kwargs)
  per_token_scores = out[0] if isinstance(out, tuple) else out
  # The model returns a tensor of shape [B, T, 1]. We squeeze the last
  # dimension to get a tensor of shape [B, T].
  per_token_scores = jnp.squeeze(per_token_scores, axis=-1)

  if stop_gradient:
    per_token_scores = jax.lax.stop_gradient(per_token_scores)

  return per_token_scores


def np_make_completion_mask(
    completion_ids: np.ndarray, eos_tok: int = 0
) -> np.ndarray:
  # [FUNCTION DOC]
  # NumPy (CPU) completion mask: for each row finds the first position equal to
  # eos_tok (or the row length if none) and returns an int32 mask that is 1 up
  # to and including that position and 0 after. Input batch by length.
  """Numpy version of make_completion_mask which executes on CPU.

  Args:
    completion_ids: Completion ids with shape [B, T].
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  seq_len = is_eos.shape[1]

  first_eos_idx = np.argmax(is_eos, axis=1)
  any_eos = np.any(is_eos, axis=1)
  eos_idx = np.where(any_eos, first_eos_idx, seq_len)
  sequence_indices = np.arange(seq_len)

  return (sequence_indices < eos_idx[:, None] + 1).astype(np.int32)


def make_completion_mask(
    completion_ids: jax.Array, eos_tok: int = 0
) -> jax.Array:
  # [FUNCTION DOC]
  # JAX version of np_make_completion_mask: 1 at positions up to and including
  # the first eos_tok in each row (all ones if no EOS), 0 afterwards; int32.
  """Create completion mask based on the EOS token.

  Args:
    completion_ids: Completion ids with shape [B, T].
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  eos_idx = jnp.full((is_eos.shape[0],), is_eos.shape[1], dtype=jnp.int32)

  any_eos = jnp.any(is_eos, axis=1)
  eos_idx = jax.lax.select(any_eos, jnp.argmax(is_eos, axis=1), eos_idx)

  sequence_indices = jnp.arange(is_eos.shape[1])[None, :]
  sequence_indices = jnp.broadcast_to(
      sequence_indices, (is_eos.shape[0], is_eos.shape[1])
  )
  return (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)


def pad_to_length(
    x: jax.Array,
    target_length: int,
    pad_value: int = 0,
    left=False,
    axis: int = 0,
) -> jax.Array:
  # [FUNCTION DOC]
  # Inputs: array x, target_length, pad_value (default 0), left flag, axis
  # (default 0). If the array is already at least target_length along axis,
  # returns it unchanged (no truncation). Otherwise builds a pad block of
  # pad_value with x's dtype and concatenates it before (left) or after x along
  # axis. Returns a new array.
  """Pads a JAX array to a specified target length along a given axis.

  Args:
      x: The JAX array to pad.
      target_length: The desired length of the padded array.
      pad_value: The value to use for padding (default: 0).
      left: If True, add padding tokens to the left of the array.
      axis: The axis along which to pad (default: 0).

  Returns:
      A new JAX array that is padded to the target length along the specified
      axis. Return original array if it is already longer than the target
      length.
  """
  length = x.shape[axis]
  if length >= target_length:
    return x

  padding_shape = list(x.shape)
  padding_shape[axis] = target_length - length
  padding = jnp.full(padding_shape, pad_value, dtype=x.dtype)

  if left:
    return jnp.concatenate([padding, x], axis=axis)
  else:
    return jnp.concatenate([x, padding], axis=axis)


def aggregate_loss(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    **kwargs: Any,
) -> jax.Array:
  # [FUNCTION DOC]
  # Inputs: per_token_loss and completion_mask (batch by length), mode, kwargs
  # (optional "norm"). Casts loss to float32; counts rows with at least one
  # valid token (minimum 1). Modes:
  # - token-mean: masked loss sum divided by total valid tokens (minimum 1).
  # - sequence-mean-token-mean: per-row masked mean (row token count minimum
  # 1), summed and divided by the non-empty row count.
  # - sequence-mean-token-scale: per-row masked sum divided by norm (default
  # the padded sequence length, minimum 1e-6), summed and divided by the
  # non-empty row count.
  # - seq-mean-token-sum: per-row masked sums averaged over non-empty rows.
  # - sequence-mean-token-sum-norm: total masked sum divided by norm (default
  # the non-empty row count, minimum 1e-6).
  # Unknown modes raise ValueError; invalid norm raises ValueError via
  # _check_get_norm. Returns a scalar loss.
  """Aggregate loss based on the loss aggregation mode.

  Args:
      per_token_loss: Per token loss.[batch_size, sequence_len]
      completion_mask: Completion mask.[batch_size, sequence_len]
      loss_agg_mode: Loss aggregation mode.

  Returns:
      Aggregated loss.
  """

  per_token_loss = per_token_loss.astype(jnp.float32)
  seq_mask = completion_mask.sum(axis=-1)
  non_zero_rows = jnp.clip((seq_mask > 0).sum(), min=1)

  if loss_agg_mode == "token-mean":
    # sum all the token loss, and average by total number of completion tokens
    # in the batch
    loss = (per_token_loss * completion_mask).sum() / (
        jnp.clip(completion_mask.sum(), min=1)
    )
  elif loss_agg_mode == "sequence-mean-token-mean":
    seq_mask = completion_mask.sum(axis=-1)  # per-sequence token count
    seq_loss = ((per_token_loss * completion_mask).sum(axis=-1)) / jnp.clip(
        seq_mask, min=1
    )
    loss = seq_loss.sum() / non_zero_rows
  elif loss_agg_mode == "sequence-mean-token-scale":
    # Look up custom normalization factor, default to max response length.
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])

    # Scale by maximum response length instead of actual response length.
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1) / jnp.clip(
        norm, min=1e-6
    )
    loss = seq_loss.sum() / non_zero_rows
  elif loss_agg_mode == "seq-mean-token-sum":
    # 1) sum token losses within each sequence
    # 2) average only across sequences that have at least one valid token
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1)
    seq_mask = (completion_mask.sum(axis=-1) > 0).astype(jnp.float32)
    loss = (seq_loss * seq_mask).sum() / jnp.clip(seq_mask.sum(), min=1e-6)
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    # Get custom normalization factor from kwargs, default to number of
    # non-empty rows.
    norm = _check_get_norm(kwargs, non_zero_rows)

    # Sum the per-sequence sums and normalize
    # TODO(sizhi): Experiment with loss in precision if loss is fp16.
    loss = (per_token_loss * completion_mask).sum() / jnp.clip(norm, min=1e-6)
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )
  return loss


def _check_get_norm(
    arguments: dict[str, Any], default: float | int | jax.Array
) -> float | jax.Array:
  # [FUNCTION DOC]
  # HELPER: Reads "norm" from the dict, falling back to default. Accepts int,
  # float, jax.Array or numpy array; Python numbers must be positive, else
  # ValueError. Any other type raises ValueError. Returns the norm unchanged.
  """Get custom normalization factor from kwargs with a default value.

  Args:
      arguments: The arguments dictionary.
      default: The default value to use if no 'norm' key is found.

  Returns:
      The normalization factor.

  Raises:
      ValueError: If the 'norm' key is present but has an invalid value or type.
  """
  norm = arguments.get("norm", default)
  if isinstance(norm, (int, float, jax.Array, np.ndarray)):
    if isinstance(norm, (int, float)):
      if norm <= 0:
        raise ValueError(
            f"Invalid 'norm' value: {norm}. Must be a positive number."
        )
    return norm

  raise ValueError(
      f"Invalid 'norm' value: {norm}. Must be a positive number (int, float,"
      " or jax.Array)."
  )
