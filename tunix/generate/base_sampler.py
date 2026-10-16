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
# Defines the abstract sampler interface shared by all text generation backends
# (native JAX sampler, vLLM, SGLang-JAX) and the common output container.
# Contents: the SamplerOutput dataclass (result of one sampling call) and the
# BaseSampler abstract base class (properties transformer and transformer_state,
# plus abstract call and tokenize methods). Two module aliases are defined: ABC
# and abstractmethod, pointing to the abc module equivalents.
# No side effects at import time beyond defining these names.
"""Base class for samplers."""

import abc
import dataclasses
from typing import List, Optional

from flax import nnx
from flax.nnx import statelib
import jax
import numpy as np

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass
class SamplerOutput:
  # [CLASS DOC]
  # Plain dataclass holding the result of a sampling call. Fields:
  # - text: list of decoded output strings, one per generated sample.
  # - logits: optional per-step logits; either a list of per-sample arrays or one
  #   stacked array; None when logits were not requested.
  # - tokens: generated token ids per sample, as numpy arrays (host memory) or one
  #   stacked numpy array.
  # - padded_prompt_tokens: numpy array of left-padded prompt token ids.
  # - logprobs: optional log-probabilities of the generated tokens (typed as a list
  #   of floats; in practice backends may store per-sample lists).
  # No validation, no methods; constructing it has no side effects.
  """Output of the sampler."""

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: Optional[list[jax.Array] | jax.Array]

  # Tokens corresponding to the generated samples.
  # Since tokens need to be transfered to RAM for decoding, we use numpy array
  # here.
  tokens: list[np.ndarray] | np.ndarray

  # Left padded prompt tokens.
  padded_prompt_tokens: np.ndarray

  logprobs: Optional[list[float]]


class BaseSampler(ABC):
  # [CLASS DOC]
  # Abstract base class every sampler implementation must subclass. Subclasses
  # must provide: a transformer property (the flax nnx model module), a
  # transformer_state property (the nnx State of that model), a callable
  # interface generating text from prompt strings, and a tokenize method.
  # Instantiating a subclass that does not implement all abstract members raises
  # TypeError (standard abc behavior).
  """Base class for samplers."""

  @property
  @abstractmethod
  def transformer(self) -> nnx.Module:
    # [FUNCTION DOC]
    # Abstract read-only property. Implementations return the nnx.Module used for
    # generation. No inputs; no body in the base class (returns None if called via
    # super). No side effects.
    """Returns the transformer module used by the sampler."""

  @property
  @abstractmethod
  def transformer_state(self) -> statelib.State:
    # [FUNCTION DOC]
    # Abstract read-only property. Implementations return the flax nnx State
    # (parameters) of the transformer used by the sampler. Base body is empty.
    """Returns the transformer state used by the sampler."""

  @abstractmethod
  def __call__(
      self,
      input_strings: str | List[str],
      max_generation_steps,
      max_prompt_length=None,
      temperature=0.0,
      top_p=None,
      top_k=None,
      beam_size=None,
      seed=None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
  ):
    # [FUNCTION DOC]
    # Abstract generation entry point. Inputs: one prompt string or a list of
    # prompt strings; max_generation_steps (maximum number of new tokens);
    # optional max_prompt_length (prompts are padded/truncated to this);
    # temperature (0 means greedy); optional top_p (nucleus) and top_k filters;
    # optional beam_size for beam search; optional seed for the random key;
    # multi_sampling (number of samples per prompt); return_logits (whether to keep
    # per-step logits); echo (whether to include the prompt in the output);
    # pad_output (whether outputs are kept padded to a fixed length).
    # Output: implementations return a SamplerOutput. Base body is empty.
    """Returns a list of generated samples for the input strings."""

  @abstractmethod
  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    # [FUNCTION DOC]
    # Abstract method. Input: a single string. Output: its token ids as a numpy
    # array or list of ints. Base body is empty.
    """Returns the tokenized the input string."""
