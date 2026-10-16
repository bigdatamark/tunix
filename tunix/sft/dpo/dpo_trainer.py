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
# DPO (Direct Preference Optimization) and ORPO (Odds Ratio Preference
#   Optimization) trainer built on PeftTrainer.
# Data flow: a batch arrives as a dict, DataInput (raw strings, tokenized
#   here) or TrainingInput (pre-tokenized: left-padded prompts,
#   right-padded chosen/rejected responses). _prepare_inputs stacks the
#   chosen and rejected examples along the batch axis (first half chosen,
#   second half rejected, prompts duplicated), builds attention mask and
#   positions, optionally computes frozen reference-model log-probs, and
#   returns a TrainExample. dpo_loss_fn computes the summed completion
#   log-probs of the policy and the DPO or ORPO loss plus reward metrics,
#   which the trainer buffers and logs (mean over steps).
# Also defines RawImageType (accepted raw image forms), compute_logps,
#   tokenization helpers, process_dpo_record (usable as a Grain map fn), and
#   aliases DpoTrainingConfig/ORPOTrainingConfig/OrpoTrainingConfig ->
#   DPOTrainingConfig and DpoTrainer/ORPOTrainer/OrpoTrainer -> DPOTrainer.
# No side effects at import beyond jitting compute_logps lazily.
"""DPO trainer."""

from __future__ import annotations

import dataclasses
from typing import Any
import warnings

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
# TODO(abheesht): We should move TokenizerAdapter outside `generate`.
from tunix.generate import tokenizer_adapter
from tunix.rl import common
from tunix.sft import peft_trainer
from typing_extensions import override


RawImageType = (
    str
    | np.ndarray
    | list[str | np.ndarray | list[str | np.ndarray] | None]
)


@flax.struct.dataclass(frozen=True)
class DataInput:
  # [CLASS DOC]
  # Frozen flax struct of raw (untokenized) DPO data: prompts, chosen_responses,
  # rejected_responses (lists of strings, equal length) and optional images
  # (RawImageType). Tokenized by DPOTrainer._prepare_inputs via
  # process_dpo_record.
  """Training data input for DPO.

  This can be used when inputs are raw strings. Tokenization, padding and
  preprocessing is taken care of by `DPOTrainer`.

  Attributes:
    prompts: A list of prompts.
    images: List of images (or list of list of images, if the model supports
      multiple images).
    chosen_responses: A list of chosen responses.
    rejected_responses: A list of rejected responses.
  """

  prompts: list[str]
  chosen_responses: list[str]
  rejected_responses: list[str]
  images: RawImageType | None = None


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # [CLASS DOC]
  # Frozen flax struct of tokenized DPO data (arrays shaped [B, L]):
  # prompt_ids/prompt_mask (left-padded, length P), chosen_ids/chosen_mask
  # and rejected_ids/rejected_mask (right-padded, same length R for both so
  # they can be concatenated), optional images (batch-leading array).
  """Tokenized training input for DPO.

  This can be used when inputs are already tokenized, padded and preprocessed.

  Attributes:
    prompt_ids: Prompt IDs. Should be left-padded.
    prompt_mask: Prompt mask. Should be left-padded.
    images: Optional images.
    chosen_ids: Chosen response IDs. Should be right-padded.
    chosen_mask: Chosen response mask. Should be right-padded.
    rejected_ids: Rejected response IDs. Should be right-padded.
    rejected_mask: Rejected response mask. Should be right-padded.
  """

  # Prompt IDs should be left padded.
  prompt_ids: jax.Array | np.ndarray
  prompt_mask: jax.Array | np.ndarray
  # Chosen IDs should be right padded.
  chosen_ids: jax.Array | np.ndarray
  chosen_mask: jax.Array | np.ndarray
  # Rejected IDs should be right padded.
  rejected_ids: jax.Array | np.ndarray
  rejected_mask: jax.Array | np.ndarray
  # Optional images.
  images: np.ndarray | jax.Array | None = None


@flax.struct.dataclass(frozen=True)
class TrainExample:
  # [CLASS DOC]
  # Frozen flax struct passed to the loss. For a batch of B preference pairs
  # all arrays have leading dim 2B (chosen first, then rejected):
  # input_ids [2B, P+R] (prompt then completion), positions [2B, P+R],
  # attention_mask (as produced by the model or causal helper),
  # ref_chosen_logps / ref_rejected_logps ([B] summed reference log-probs or
  # None), completion_mask [2B, R], logits_to_keep (static int R; not a
  # pytree leaf), images (optional, duplicated to 2B).
  input_ids: jax.Array  # Concatenated [prompt_ids, completion_ids]
  positions: jax.Array
  attention_mask: jax.Array
  ref_chosen_logps: jax.Array | None
  ref_rejected_logps: jax.Array | None
  completion_mask: jax.Array
  logits_to_keep: int = flax.struct.field(pytree_node=False)
  images: jax.Array | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class DPOTrainingConfig(peft_trainer.TrainingConfig):
  # [CLASS DOC]
  # Keyword-only slotted dataclass extending peft_trainer.TrainingConfig.
  # Fields: algorithm ("dpo" or "orpo"; anything else behaves as DPO),
  # beta (0.1, DPO temperature), lambda_orpo (0.1, ORPO odds-ratio weight),
  # label_smoothing (0.0, used by both), max_prompt_length and
  # max_response_length (required only for raw-string inputs).
  """DPO/ORPO Training Config."""

  algorithm: str = "dpo"  # "dpo" or "orpo"
  beta: float = (
      0.1  # 𝛽 for KL penalty (DPO only) https://arxiv.org/pdf/2305.18290
  )
  lambda_orpo: float = 0.1  # Weight for preference loss (ORPO only)
  label_smoothing: float = 0.0

  # Should be specified only if your input has strings instead of tokenized IDs.
  max_prompt_length: int | None = None
  max_response_length: int | None = None


@nnx.jit(static_argnums=(4,))
def compute_logps(
    model,
    input_ids,
    positions,
    attention_mask,
    logits_to_keep,
    completion_mask,
    images=None,
):
  # [FUNCTION DOC]
  # nnx.jit-compiled (logits_to_keep static, argnum 4). Inputs: model,
  #   input_ids, positions, attention_mask, logits_to_keep (int R),
  #   completion_mask [2B, R], images (optional).
  # Steps: per-token log-probs of the last R target tokens via
  #   common.get_per_token_logps; multiply by completion_mask and sum over
  #   the sequence axis; split the batch in half.
  # Output: (chosen_logps [B], rejected_logps [B]) summed log-probs.
  # Exceptions: from the model. Side effects: none.
  """Computes the log probabilities for chosen and rejected tokens."""
  token_logps = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
      images=images,
  )
  token_logps = (token_logps * completion_mask).sum(axis=-1)

  batch_size = token_logps.shape[0]
  chosen_logps = token_logps[: batch_size // 2]
  rejected_logps = token_logps[batch_size // 2 :]
  return chosen_logps, rejected_logps


class DPOTrainer(peft_trainer.PeftTrainer):
  """Direct Preference Optimization (DPO) and ORPO trainer.
  # [CLASS DOC]
  # Preference trainer supporting DPO and ORPO selected by
  # training_config.algorithm. State beyond PeftTrainer: ref_model (frozen
  # reference or None), dpo_config, algorithm, tokenizer (TokenizerAdapter
  # or None), image_processor, _ref_model_exists, _aux_metrics_to_log
  # (metric name -> aggregation fn, np.mean for all: rewards/chosen,
  # rewards/rejected, rewards/margin, rewards/accuracy, log_probs/chosen,
  # log_probs/rejected, plus odds_ratio for ORPO). Loss is dpo_loss_fn with
  # aux output; the model-input fn passes the TrainExample and algorithm
  # hyperparameters as keyword arguments to the loss.

  DPO is a preference tuning method for aligning large language models with
  human or AI preferences. It is a more efficient, performant alternative
  to RLHF.

  DPO is simpler because it eliminates the need for text generation in the
  training loop. Moreover, DPO bypasses the reward modeling step entirely, i.e.,
  we do not need to train a separate reward model. It uses a dataset of
  preferences (pairs of "chosen" and "rejected responses) to directly optimize
  the policy model by using a classification-style loss.

  ORPO (Odds Ratio Preference Optimization) is a memory-efficient variant that
  combines supervised fine-tuning with preference alignment without requiring
  a separate reference model, making it approximately 50% more memory-efficient.

  References:
  - DPO: https://arxiv.org/abs/2305.18290
  - ORPO: https://arxiv.org/abs/2403.07691
  """

  def __init__(
      self,
      model: nnx.Module,
      ref_model: nnx.Module | None,
      optimizer: optax.GradientTransformation,
      training_config: DPOTrainingConfig,
      tokenizer: Any | None = None,
      image_processor: Any | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs: model (policy), ref_model (optional), optimizer,
    #   training_config (DPOTrainingConfig), tokenizer (optional; wrapped in
    #   TokenizerAdapter), image_processor (optional callable).
    # Steps: store model/ref_model/config/algorithm; init PeftTrainer; wrap
    #   tokenizer; set loss fn dpo_loss_fn with has_aux; set the model
    #   input fn to map a TrainExample x to {"train_example": x,
    #   "algorithm": "orpo", "lambda_orpo", "label_smoothing"} for ORPO, or
    #   {"train_example": x, "algorithm": "dpo", "beta", "label_smoothing"}
    #   otherwise (set both via with_gen_model_input_fn and by direct
    #   assignment); set _has_aux; record whether ref_model exists; set up
    #   aux metric aggregation (adding odds_ratio for ORPO).
    # Exceptions: from parent init. Side effects: none beyond state.
    # Note: for ORPO a provided ref_model is still run every step though
    #   the ORPO loss ignores its outputs.
    """Initializes the DPO/ORPO trainer.

    Args:
      model: The policy model to be trained.
      ref_model: The reference/anchor model which is kept fixed/frozen during
        training (DPO only). It is used to prevent the policy model from
        drifting too far from its original capabilities. For ORPO, this should
        be None. If `ref_model` is None for DPO, we don't use it in the loss
        term.
      optimizer: The optimizer used for training the policy model.
      training_config: A `DPOTrainingConfig` object containing DPO/ORPO-specific
        hyperparameters like `beta`, `lambda_orpo`, and `label_smoothing`.
      tokenizer: An optional tokenizer. If provided, the trainer can accept
        string inputs and tokenize them internally.
      image_processor: An optional image processor. If provided, the trainer can
        accept raw images and process (resize, normalize, etc.) them internally.
    """
    self.model = model
    self.ref_model = ref_model
    self.dpo_config = training_config
    self.algorithm = training_config.algorithm
    super().__init__(model, optimizer, training_config)

    self.tokenizer = (
        None
        if tokenizer is None
        else tokenizer_adapter.TokenizerAdapter(tokenizer)
    )
    self.image_processor = image_processor

    self.with_loss_fn(dpo_loss_fn, has_aux=True)

    if self.algorithm == "orpo":
      self.with_gen_model_input_fn(
          lambda x: {
              "train_example": x,
              "algorithm": "orpo",
              "lambda_orpo": self.dpo_config.lambda_orpo,
              "label_smoothing": self.dpo_config.label_smoothing,
          }
      )
      self.gen_model_input_fn = lambda x: {
          "train_example": x,
          "algorithm": "orpo",
          "lambda_orpo": self.dpo_config.lambda_orpo,
          "label_smoothing": self.dpo_config.label_smoothing,
      }
    else:
      self.with_gen_model_input_fn(
          lambda x: {
              "train_example": x,
              "algorithm": "dpo",
              "beta": self.dpo_config.beta,
              "label_smoothing": self.dpo_config.label_smoothing,
          }
      )
      self.gen_model_input_fn = lambda x: {
          "train_example": x,
          "algorithm": "dpo",
          "beta": self.dpo_config.beta,
          "label_smoothing": self.dpo_config.label_smoothing,
      }

    self._has_aux = True

    # If reference model is not provided, we don't use it in the loss term.
    self._ref_model_exists = ref_model is not None

    self._aux_metrics_to_log = {
        "rewards/chosen": np.mean,
        "rewards/rejected": np.mean,
        "rewards/margin": np.mean,
        "rewards/accuracy": np.mean,
        "log_probs/chosen": np.mean,
        "log_probs/rejected": np.mean,
    }

    if self.algorithm == "orpo":
      self._aux_metrics_to_log["odds_ratio"] = np.mean

  @override
  def _prepare_inputs(
      self,
      training_input: dict[str, Any] | DataInput | TrainingInput,
  ) -> Any:
    # [FUNCTION DOC]
    # Converts a batch into a TrainExample.
    # Input: training_input (dict, DataInput or TrainingInput).
    # Steps: dict -> _preprocess_dict. DataInput -> require tokenizer
    #   (else ValueError) and both max lengths (else ValueError), then
    #   process_dpo_record. Then: prompt ids/mask duplicated along batch;
    #   completion ids/mask = chosen then rejected along batch;
    #   input_ids = [prompt | completion] along the sequence axis and the
    #   same for the mask; images duplicated if present. Attention mask:
    #   model.get_attention_mask(input_ids, inputs_mask=mask) if the model
    #   has that method, else common.make_causal_attn_mask(mask).
    #   logits_to_keep = R; positions = common.build_positions_from_mask.
    #   If a reference model exists, compute ref chosen/rejected summed
    #   log-probs via compute_logps (also in eval).
    # Output: TrainExample. Exceptions: ValueError as above; shape errors
    #   if chosen and rejected lengths differ. Side effects: ref forward.
    if isinstance(training_input, dict):
      training_input = _preprocess_dict(training_input)

    # If the inputs are list of strings, let's tokenise them and pad them.
    if isinstance(training_input, DataInput):
      if self.tokenizer is None:
        raise ValueError(
            "Tokenizer must be provided if training input is not tokenized."
        )

      max_prompt_length = self.dpo_config.max_prompt_length
      max_response_length = self.dpo_config.max_response_length
      if (
          self.dpo_config.max_prompt_length is None
          or self.dpo_config.max_response_length is None
      ):
        raise ValueError(
            "max_prompt_length and max_response_length must be provided if "
            "training input is not tokenized. Received: "
            f"max_prompt_length={max_prompt_length}, "
            f"max_response_length={max_response_length}."
        )

      training_input = process_dpo_record(
          record={
              "prompts": training_input.prompts,
              "images": training_input.images,
              "chosen_responses": training_input.chosen_responses,
              "rejected_responses": training_input.rejected_responses,
          },
          tokenizer=self.tokenizer,
          max_prompt_length=self.dpo_config.max_prompt_length,
          max_response_length=self.dpo_config.max_response_length,
          image_processor=self.image_processor,
      )

    # Concatenate chosen and rejected IDs so we can do a forward pass together.
    prompt_ids = jnp.concatenate(
        [training_input.prompt_ids, training_input.prompt_ids], axis=0
    )
    prompt_mask = jnp.concatenate(
        [training_input.prompt_mask, training_input.prompt_mask], axis=0
    )
    completion_ids = jnp.concatenate(
        [training_input.chosen_ids, training_input.rejected_ids], axis=0
    )
    completion_mask = jnp.concatenate(
        [training_input.chosen_mask, training_input.rejected_mask], axis=0
    )
    input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)

    # Compute positions, attention mask, etc., to be fed to the model.
    mask = jnp.concat([prompt_mask, completion_mask], axis=1)

    # Duplicate images as well (for multimodal inputs only).
    images = training_input.images
    if images is not None:
        images = jnp.concatenate([images, images], axis=0)

    if hasattr(self.model, "get_attention_mask"):
      attention_mask = self.model.get_attention_mask(
        input_ids, inputs_mask=mask
      )
    else:
      attention_mask = common.make_causal_attn_mask(mask)

    logits_to_keep = completion_ids.shape[1]
    positions = common.build_positions_from_mask(mask)

    # Compute the log probabilities for the chosen and rejected tokens.
    ref_chosen_logps = None
    ref_rejected_logps = None
    if self._ref_model_exists:
      ref_chosen_logps, ref_rejected_logps = compute_logps(
          self.ref_model,
          input_ids,
          positions,
          attention_mask,
          logits_to_keep,
          completion_mask,
          images=images,
      )
    return TrainExample(
        input_ids=input_ids,
        positions=positions,
        attention_mask=attention_mask,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        completion_mask=completion_mask,
        logits_to_keep=logits_to_keep,
        images=images,
    )

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Input: aux (metrics dict from dpo_loss_fn for a train step). For each
    #   metric in _aux_metrics_to_log, append aux[name] to the buffered
    #   train metrics' additional_metrics[name] list, creating the entry
    #   (values list, aggregation fn) if absent.
    # Exceptions: AssertionError if no train metrics buffer; KeyError if aux
    #   lacks a metric. Side effects: mutates the buffer.
    # Note: ORPO's sft_loss and or_loss aux values are never logged.
    assert self._buffered_train_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Same as _post_process_train_step but appends into the buffered eval
    # metrics. AssertionError if no eval metrics buffer exists.
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self._aux_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )


def dpo_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    algorithm: str = "dpo",
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  # [FUNCTION DOC]
  # DPO/ORPO loss. Inputs: model, train_example (TrainExample), algorithm,
  #   beta, lambda_orpo, label_smoothing (ls).
  # First compute policy summed log-probs chosen (c) and rejected (r) via
  #   compute_logps.
  # ORPO branch (algorithm == "orpo"): B = half of the batch; chosen
  #   lengths = sum of chosen completion mask, floored at 1; sft = -c /
  #   length; log_odds = (c - r) - (log1p(-exp(c)) - log1p(-exp(r)));
  #   or_loss = -(log_sigmoid(log_odds)*(1-ls) + log_sigmoid(-log_odds)*ls);
  #   per-example total = sft + lambda_orpo*or_loss; rewards =
  #   lambda_orpo * c and lambda_orpo * r; odds_ratio = exp(log_odds).
  #   Returns (mean total, aux) with aux keys rewards/chosen,
  #   rewards/rejected, rewards/margin, rewards/accuracy (fraction chosen
  #   reward > rejected), log_probs/chosen, log_probs/rejected, odds_ratio,
  #   sft_loss, or_loss (all batch means).
  # DPO branch (otherwise): chosen ratio = c minus ref chosen if present;
  #   rejected ratio likewise; delta = difference; losses =
  #   -(log_sigmoid(beta*delta)*(1-ls) + log_sigmoid(-beta*delta)*ls);
  #   rewards = beta * ratios. Returns (mean loss, aux with the first six
  #   keys above).
  # Exceptions: none explicit. Side effects: none.
  # KNOWN ISSUE: ORPO uses summed (not length-averaged) log-probs in
  #   log1p(-exp(.)), which is NaN/inf when a summed log-prob is >= 0 or
  #   numerically 0 (e.g. empty completions); see the inline note.
  """DPO/ORPO loss function.

  Args:
    model: The model to compute loss for.
    train_example: Training example containing input_ids, masks, etc.
    algorithm: "dpo" or "orpo".
    beta: Weight for KL penalty (DPO only).
    lambda_orpo: Weight for preference loss (ORPO only).
    label_smoothing: Label smoothing factor.

  Returns:
    A tuple of (loss, auxiliary_metrics_dict).
  """
  chosen_logps, rejected_logps = compute_logps(
      model,
      train_example.input_ids,
      train_example.positions,
      train_example.attention_mask,
      train_example.logits_to_keep,
      train_example.completion_mask,
      images=train_example.images,
  )

  if algorithm == "orpo":
    # ORPO loss = L_SFT + λ * L_OR
    # Paper: https://arxiv.org/abs/2403.07691

    # L_SFT: Supervised fine-tuning loss on chosen responses
    # Normalize by sequence length as per Equation 2 in paper
    batch_size = train_example.completion_mask.shape[0] // 2
    chosen_mask = train_example.completion_mask[:batch_size]
    chosen_lengths = chosen_mask.sum(axis=-1)
    chosen_lengths = jnp.maximum(chosen_lengths, 1.0)  # Avoid division by zero

    # L_SFT = -(1/|y_w|) * Σ log P (Paper Equation 2)
    sft_loss = -chosen_logps / chosen_lengths

    # L_OR: Odds ratio preference loss
    # Following HuggingFace TRL implementation exactly (Eqs. 4 and 7 from paper)
    # Note: log1p(-exp(x)) requires x < 0 to avoid NaN. This works when log probs
    # are averaged per token, but may produce NaN for summed log probs if sequences
    # are long. TRL uses summed log probs and relies on them being negative.
    log_odds = (chosen_logps - rejected_logps) - (
        jnp.log1p(-jnp.exp(chosen_logps)) - jnp.log1p(-jnp.exp(rejected_logps))
    )

    # Apply label smoothing to odds ratio loss
    or_loss = -(
        jax.nn.log_sigmoid(log_odds) * (1 - label_smoothing)
        + jax.nn.log_sigmoid(-log_odds) * label_smoothing
    )

    # Combined ORPO loss: L_ORPO = L_SFT + λ * L_OR
    total_loss = sft_loss + lambda_orpo * or_loss

    # Compute rewards for logging (matching HuggingFace TRL implementation)
    chosen_rewards = lambda_orpo * chosen_logps
    rejected_rewards = lambda_orpo * rejected_logps

    # Compute odds ratio for logging
    odds_ratio = jnp.exp(log_odds)

    aux = {
        "rewards/chosen": chosen_rewards.mean(),
        "rewards/rejected": rejected_rewards.mean(),
        "rewards/margin": (chosen_rewards - rejected_rewards).mean(),
        "rewards/accuracy": (chosen_rewards > rejected_rewards).mean(),
        "log_probs/chosen": chosen_logps.mean(),
        "log_probs/rejected": rejected_logps.mean(),
        "odds_ratio": odds_ratio.mean(),
        "sft_loss": sft_loss.mean(),
        "or_loss": or_loss.mean(),
    }

    return total_loss.mean(), aux
  else:
    # DPO loss
    chosen_log_ratio = chosen_logps
    if train_example.ref_chosen_logps is not None:
      chosen_log_ratio = chosen_log_ratio - train_example.ref_chosen_logps
    rejected_log_ratio = rejected_logps
    if train_example.ref_rejected_logps is not None:
      rejected_log_ratio = rejected_log_ratio - train_example.ref_rejected_logps
    delta = chosen_log_ratio - rejected_log_ratio
    losses = -(
        jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
        + jax.nn.log_sigmoid(-beta * delta) * label_smoothing
    )

    # Compute rewards.
    chosen_rewards = beta * chosen_log_ratio
    rejected_rewards = beta * rejected_log_ratio

    aux = {
        "rewards/chosen": chosen_rewards.mean(),
        "rewards/rejected": rejected_rewards.mean(),
        "rewards/margin": (chosen_rewards - rejected_rewards).mean(),
        "rewards/accuracy": (chosen_rewards > rejected_rewards).mean(),
        "log_probs/chosen": chosen_logps.mean(),
        "log_probs/rejected": rejected_logps.mean(),
    }

    return losses.mean(), aux


def _generate_ids_and_masks(
    input_strings: list[str],
    tokenizer: Any,
    max_length: int,
    left_pad: bool = True,
) -> tuple[jax.Array, jax.Array]:
  # [FUNCTION DOC]
  # HELPER: tokenizes and pads a list of strings. Inputs: input_strings,
  #   tokenizer (TokenizerAdapter-like), max_length, left_pad (default True).
  # Each string -> _tokenize, truncated to its first max_length tokens,
  #   padded with tokenizer.pad_id() to max_length (left or right) via
  #   common.pad_to_length. Output: (ids [N, max_length], mask int32 where
  #   ids != pad id).
  # KNOWN ISSUE: the mask is derived by comparing to the pad id, so real
  #   tokens equal to the pad id (e.g. when pad == eos) are masked out;
  #   truncation keeps the head, so left-padded prompts lose their end.
  """Generates ids and masks for a list of strings."""
  tokens = [_tokenize(x, tokenizer) for x in input_strings]
  all_input_ids = jnp.array([
      common.pad_to_length(
          x[:max_length],
          target_length=max_length,
          pad_value=tokenizer.pad_id(),
          left=left_pad,
          axis=-1,
      )
      for x in tokens
  ])
  # generate masks
  all_input_mask = (all_input_ids != tokenizer.pad_id()).astype("int32")
  return all_input_ids, all_input_mask


def _tokenize(input_string: str, tokenizer: Any) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: Inputs: input_string, tokenizer. Encodes the string, prepends
  #   [bos_id] if bos_id() is truthy, removes duplicate leading BOS via
  #   tokenizer.dedup_bos_ids, returns an int32 array.
  # KNOWN ISSUE: a BOS id of 0 is treated as "no BOS".
  """Tokenizes the input string."""
  input_ids = tokenizer.encode(input_string)
  bos_tok = [tokenizer.bos_id()] if tokenizer.bos_id() else []
  input_ids = jnp.array(
    tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=jnp.int32
  )
  return input_ids


def _preprocess_dict(
    training_input: dict[str, Any],
) -> DataInput | TrainingInput:
  """Wraps input dict with either DataInput or TrainingInput."""
  # [FUNCTION DOC]
  # HELPER: Input: training_input dict. If it contains every TrainingInput
  #   field except images, returns TrainingInput built from those fields
  #   (missing images -> None). Else if it contains every DataInput field
  #   except images, returns DataInput similarly. Otherwise raises
  #   ValueError listing both field sets and the received keys.

  data_input_fields = [field.name for field in dataclasses.fields(DataInput)]
  tokenized_input_fields = [
      field.name for field in dataclasses.fields(TrainingInput)
  ]

  # If the dict contains tokenized fields, we should wrap it with TrainingInput.
  if all(
      field in training_input
      for field in tokenized_input_fields
      if field != "images"
  ):
    return TrainingInput(**{
        field: training_input.get(field, None)
        for field in tokenized_input_fields
    })
  elif all(
    field in training_input for field in data_input_fields if field != "images"
  ):
    return DataInput(
        **{field: training_input.get(field, None) for field in data_input_fields}
    )
  else:
    raise ValueError(
        "Training input must contain either tokenized fields "
        f"({tokenized_input_fields}) or raw string fields "
        f"({data_input_fields}). Received: {training_input.keys()}."
    )


def process_dpo_record(
    record: dict[str, str | list[str] | RawImageType],
    tokenizer: Any,
    max_prompt_length: int,
    max_response_length: int,
    *,
    image_processor: Any = None,
) -> TrainingInput:
  # [FUNCTION DOC]
  # Tokenizes one record (or a batch) for DPO.
  # Inputs: record dict with "prompts", "chosen_responses",
  #   "rejected_responses" (string or list of strings) and optional
  #   "images"; tokenizer; max_prompt_length; max_response_length;
  #   image_processor (keyword-only, optional callable).
  # Steps: unbatched if prompts is a str or dict; wrap single strings in
  #   lists. Prompts -> left-padded ids/masks of max_prompt_length; chosen
  #   and rejected -> right-padded of max_response_length. If images given:
  #   warn if no image_processor (images assumed preprocessed), else images
  #   = array(image_processor(images)). If unbatched, squeeze the leading
  #   axis of every array (images too).
  # Output: TrainingInput. Exceptions: KeyError for missing keys; squeeze
  #   errors if unbatched images lack a leading size-1 axis. Side effects:
  #   warnings.
  # KNOWN ISSUE: a dict prompt is treated as unbatched but then passed to
  #   the string tokenizer as is.
  """Processes and tokenizes a single record for DPO training.

  This function takes a dictionary containing a prompt, a chosen response,
  and a rejected response. It tokenizes each text field and creates the
  corresponding attention masks.

  Note: We use a dictionary here, to make it easier to use on any Grain dataset
  with `.map`.

  Args:
      record: A dictionary, containing "prompts", "images", "chosen_responses",
        "rejected_responses" as keys. For text fields, the values can be a
        single string, or a list of strings. For `"images"`, the fields can be
        a path (str), a NumPy array, list of paths, list of arrays, list of
        lists of paths/arrays, or just None.
      tokenizer: The tokenizer or processor to use for converting text into
        token IDs.
      max_prompt_length: The maximum length for the tokenized prompts. Any
        sequence longer than this will be truncated.
      max_response_length: The maximum length for the tokenized responses. Any
        sequence longer than this will be truncated.

  Returns:
      A `TrainingInput` object.
  """

  prompts = record["prompts"]
  images = record.get("images", None)
  chosen_responses = record["chosen_responses"]
  rejected_responses = record["rejected_responses"]

  unbatched = isinstance(prompts, (str, dict))

  if unbatched:
    prompts = [prompts]
  if isinstance(chosen_responses, str):
    chosen_responses = [chosen_responses]
  if isinstance(rejected_responses, str):
    rejected_responses = [rejected_responses]

  # Only prompt is left padded, others are right padded.
  prompt_ids, prompt_mask = _generate_ids_and_masks(
      prompts,
      tokenizer,
      max_prompt_length,
      left_pad=True,
  )
  chosen_ids, chosen_mask = _generate_ids_and_masks(
      chosen_responses, tokenizer, max_response_length, left_pad=False
  )
  rejected_ids, rejected_mask = _generate_ids_and_masks(
      rejected_responses, tokenizer, max_response_length, left_pad=False
  )
  if images is not None:
    if image_processor is None:
      warnings.warn(
          "`image_processor` was not provided. Images are expected to be "
          "preprocessed (resized, normalized, etc.)."
      )
    else:
      images = jnp.array(image_processor(images))

  if unbatched:
    prompt_ids = jnp.squeeze(prompt_ids, axis=0)
    chosen_ids = jnp.squeeze(chosen_ids, axis=0)
    rejected_ids = jnp.squeeze(rejected_ids, axis=0)
    prompt_mask = jnp.squeeze(prompt_mask, axis=0)
    chosen_mask = jnp.squeeze(chosen_mask, axis=0)
    rejected_mask = jnp.squeeze(rejected_mask, axis=0)
    if images is not None:
      images = jnp.squeeze(images, axis=0)

  return TrainingInput(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      chosen_ids=chosen_ids,
      chosen_mask=chosen_mask,
      rejected_ids=rejected_ids,
      rejected_mask=rejected_mask,
      images=images,
  )


DpoTrainingConfig = DPOTrainingConfig
DpoTrainer = DPOTrainer

# ORPO aliases
ORPOTrainingConfig = DPOTrainingConfig
ORPOTrainer = DPOTrainer
OrpoTrainingConfig = DPOTrainingConfig
OrpoTrainer = DPOTrainer
