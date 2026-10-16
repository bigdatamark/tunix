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
# HELPER: shared test utilities for Tunix tests. Provides pytree assertion helpers (equal,
# not-equal, close) that convert JAX arrays to numpy, toy nnx models (a small decoder block, a
# toy transformer with optional fake vision embeddings, and a wrapper adding a scalar score
# head), frozen dataclass configs for those models, LoRA application for the toy model, a mock
# SentencePiece vocabulary with a fixed word-to-id mapping, retrying HuggingFace download helpers,
# chat-template batching, LLM output keyword validation, directory deletion, JAX array cleanup and
# Colab detection.
# Side effects at import: calls the env utility that sets up the sharding environment (e.g.
# forcing multiple virtual CPU devices) so tests can build meshes.

"""Common test utilities."""

from collections.abc import Iterable
import dataclasses
import gc
import os
import shutil
import sys
from typing import Any, List, Tuple

from flax import nnx
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import tenacity
from tunix.models.gemma3 import merge_embeddings as merge_embeddings_lib
from tunix.models.gemma3 import utils as gemma_utils
from tunix.rl import reshard
from tunix.utils import env_utils

import sentencepiece as spm

env_utils.setup_sharding_environment()


def _convert_leaf_to_nparray(leaf):
  # [FUNCTION DOC]
  # HELPER: if the leaf is a JAX array, blocks until it is computed and returns a numpy copy;
  # otherwise returns the leaf unchanged.
  if isinstance(leaf, jax.Array):
    jax.block_until_ready(leaf)
    return np.asarray(leaf)
  return leaf


def _convert_to_nparray(tree):
  # [FUNCTION DOC]
  # HELPER: maps the leaf converter over a pytree, returning a tree of numpy arrays/other leaves.
  return jax.tree.map(_convert_leaf_to_nparray, tree)


def assert_equal(path, x, y):
  # [FUNCTION DOC]
  # HELPER: asserts two pytrees are exactly equal (numpy array-equal after conversion); the error
  # message names the given path. Raises AssertionError on mismatch.
  np.testing.assert_array_equal(
      _convert_to_nparray(x),
      _convert_to_nparray(y),
      err_msg=f'Mismatch at path: {path}',
  )


def assert_not_equal(path, x, y):
  # [FUNCTION DOC]
  # HELPER: asserts that at least one element differs between the two converted values
  # (elementwise not-equal, any); AssertionError naming the path otherwise. Shapes must broadcast.
  np.testing.assert_(
      np.any(np.not_equal(_convert_to_nparray(x), _convert_to_nparray(y))),
      msg=f'Unexpected match at path: {path}',
  )


def assert_close(path, x, y, atol=1e-5, rtol=1e-5):
  # [FUNCTION DOC]
  # HELPER: asserts the converted values are close with given absolute and relative tolerances
  # (default 1e-5 each); AssertionError naming the path otherwise.
  # KNOWN ISSUE: atol and rtol are passed positionally, and numpy's allclose assertion takes rtol
  # before atol, so the tolerances are swapped (harmless with equal defaults).
  np.testing.assert_allclose(
      _convert_to_nparray(x),
      _convert_to_nparray(y),
      atol,
      rtol,
      err_msg=f'Mismatch at path: {path}',
  )


class Decoder(nnx.Module):
  # [CLASS DOC]
  # HELPER: toy decoder block: 4-head self-attention over 16 features (no bias, no decode cache)
  # with residual, then a 16-to-32 linear with ReLU and a 32-to-16 linear with residual. Linear
  # kernels/biases carry sharding annotations over "fsdp" and "tp" mesh axes.
  """Toy decoder for testing."""

  def __init__(self, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # HELPER: builds the attention layer and the two linear layers (LeCun-normal kernels with
    # partitioning ("fsdp","tp") and ("tp","fsdp"); zero biases partitioned "tp" and "fsdp") using rngs.
    self.attn = nnx.MultiHeadAttention(
        num_heads=4,
        in_features=16,
        qkv_features=16,
        use_bias=False,
        decode=False,
        rngs=rngs,
    )
    kernel_init_fn = nnx.initializers.lecun_normal()
    self.w1 = nnx.Linear(
        in_features=16,
        out_features=32,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('fsdp', 'tp')),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('tp',)),
    )
    self.w2 = nnx.Linear(
        in_features=32,
        out_features=16,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('tp', 'fsdp')),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), ('fsdp',)
        ),
    )

  def __call__(self, x):
    # [FUNCTION DOC]
    # HELPER: forward pass. Input: array of shape (..., 16). Output: attention residual, then MLP
    # output plus the attention residual, same shape.
    x = self.attn(x) + x
    h = nnx.relu(self.w1(x))
    h = self.w2(h) + x
    return h


@dataclasses.dataclass(kw_only=True, frozen=True)
class VisionConfig:
  # [CLASS DOC]
  # HELPER: frozen keyword-only dataclass with toy vision token settings: number of multimodal
  # tokens per image (4) and token ids for the soft placeholder (22), start of image (23), end of
  # image (24) and double newline (25).
  """Vision config for testing."""

  num_mm_tokens_per_image: int = 4
  soft_token_placeholder: int = 22
  start_of_image_token: int = 23
  end_of_image_token: int = 24
  double_new_line_token: int = 25


@dataclasses.dataclass(kw_only=True, frozen=True)
class ModelConfig:
  # [CLASS DOC]
  # HELPER: frozen keyword-only dataclass for the toy transformer: num_layers 4, num_kv_heads 4,
  # head_dim 16, vocab_size 256, optional vision_config (default None).
  """Model config for testing."""

  num_layers: int = 4
  num_kv_heads: int = 4
  head_dim: int = 16
  vocab_size: int = 256
  vision_config: VisionConfig | None = None


class ToyTransformer(nnx.Module):
  # [CLASS DOC]
  # HELPER: toy decoder-only transformer: embedding (vocab x 16), a list of toy decoder blocks,
  # and a 16-to-vocab LM head; head_dim attribute fixed to 16. Mimics the Tunix model interface
  # (call signature, num_embed, attention-mask builder, example model input).
  """Toy transformer for testing."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # HELPER: stores config and builds embedding, num_layers decoder blocks and the LM head with rngs.
    self.config = config
    self.emb = nnx.Embed(config.vocab_size, 16, rngs=rngs)
    self.layers = nnx.List(
        [Decoder(rngs=rngs) for _ in range(config.num_layers)]
    )
    self.lm_head = nnx.Linear(
        in_features=16, out_features=config.vocab_size, rngs=rngs
    )

    self.head_dim = 16

  def __call__(
      self,
      x,
      positions,
      cache=None,
      attention_mask=None,
      output_hidden_states=False,
      images=None,
      segment_ids: jax.Array | None = None,
  ):
    # [FUNCTION DOC]
    # HELPER: forward pass. Inputs: token ids, positions (ignored), cache (passed through), attention
    # mask (ignored), output_hidden_states flag, optional images (batch, num_images, ...), segment ids
    # (ignored). Embeds tokens; if images are given, generates deterministic random vision embeddings
    # (fixed key 2) of shape (batch, num_images, tokens per image, 16) and merges them into the
    # positions where the token equals the soft placeholder id. Runs all decoder blocks; if requested,
    # sows the final hidden state as an intermediate named "all_hidden_states". Output: tuple
    # (logits, cache). Raises AttributeError if images are given without a vision config.
    tokens = x
    x = self.emb(tokens)
    if images is not None:
      num_images = images.shape[1]
      vision_embs = jax.random.normal(
          key=jax.random.key(2),
          shape=(
              x.shape[0],
              num_images,
              self.config.vision_config.num_mm_tokens_per_image,  # pytype: disable=attribute-error
              x.shape[-1],
          ),
          dtype=x.dtype,
      )
      # Merge the soft tokens back with the text embeddings.
      x = merge_embeddings_lib.merge_embeddings(
          text_embeddings=x,
          vision_embeddings=vision_embs,
          mask=tokens == self.config.vision_config.soft_token_placeholder,  # pytype: disable=attribute-error
      )

    for layer in self.layers:
      x = layer(x)
    if output_hidden_states:
      self.sow(
          nnx.Intermediate,
          'all_hidden_states',
          x,
      )
    return self.lm_head(x), cache

  @property
  def num_embed(self) -> int:
    # [FUNCTION DOC]
    # HELPER: property returning the embedding table size (vocab size).
    return self.emb.num_embeddings

  def get_attention_mask(self, tokens, inputs_mask=None):
    # [FUNCTION DOC]
    # HELPER: builds the attention mask via the Gemma3 utility from tokens and an optional input mask,
    # passing the soft-token placeholder id when a vision config exists (else None).
    token_placeholder_id = (
        None
        if self.config.vision_config is None
        else self.config.vision_config.soft_token_placeholder
    )
    return gemma_utils.get_attention_mask(
        tokens,
        inputs_mask=inputs_mask,
        token_placeholder_id=token_placeholder_id,
    )

  def get_model_input(self):
    # [FUNCTION DOC]
    # HELPER: returns the dummy example inputs used for LoRA tracing (see the module-level function).
    return get_dummy_inputs_for_lora_toy_transformer_tests()


def get_dummy_inputs_for_lora_toy_transformer_tests():
  # [FUNCTION DOC]
  # HELPER: returns example inputs for LoRA application: x and positions as int32 ones of shape
  # (1, 1), cache None, and a boolean all-true attention mask of shape (1, 1, 1).
  return {
      'x': jnp.ones((1, 1), dtype=jnp.int32),
      'positions': jnp.ones((1, 1), dtype=jnp.int32),
      'cache': None,
      'attention_mask': jnp.ones((1, 1, 1), dtype=jnp.bool),
  }


def get_lora_model(
    model: nnx.Module,
    module_path: str = '.*w1|.*w2',
    mesh: jax.sharding.Mesh | None = None,
    rank: int = 4,
    alpha: float = 2.0,
) -> nnx.Module:
  # [FUNCTION DOC]
  # HELPER: applies qwix LoRA to a model. Inputs: model, module_path regex (default matches w1 and
  # w2 layers), optional mesh, rank (4), alpha (2.0). Traces with the dummy inputs; reshards onto the
  # mesh if provided. Output: the LoRA model.
  """Apply LoRA to ToyTransformer."""
  lora_provider = qwix.LoraProvider(
      module_path=module_path,
      rank=rank,
      alpha=alpha,
  )
  dummy_model_input = get_dummy_inputs_for_lora_toy_transformer_tests()
  lora_model = qwix.apply_lora_to_model(
      model, lora_provider, **dummy_model_input
  )
  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


class MockVocab(spm.SentencePieceProcessor):
  # [CLASS DOC]
  # HELPER: mock SentencePiece processor with a whitespace-tokenized fixed vocabulary. Class
  # attribute: default word-to-id mapping (pad 0, BOS 1, EOS 2, then 19 words with ids 3 to 21).
  # pad/bos/eos ids are 0/1/2.
  """Mock vocabulary for testing."""

  DEFAULT_MAPPING = {
      '<pad>': 0,
      '<s>': 1,
      '</s>': 2,
      'input': 3,
      'string': 4,
      'hello': 5,
      'world': 6,
      'Hello': 7,
      'there': 8,
      '!': 9,
      'My': 10,
      'name': 11,
      'is': 12,
      'Morgane': 13,
      'Tunix': 14,
      'Parallax': 15,
      'PT': 16,
      'library': 17,
      'distributed': 18,
      'training': 19,
      'optimizer': 20,
      'quantization': 21,
  }

  def __init__(
      self, mapping_text_to_id: dict[str, int] | None = None,
      is_multimodal=False
  ):
    # [FUNCTION DOC]
    # HELPER: initializes the base processor; sets a start id of 3; when multimodal, adds image
    # tokens (ids 22-25) to the default mapping; uses the given mapping if truthy else the default,
    # and stores the vocab size as the mapping length.
    # KNOWN ISSUE: the multimodal update mutates the shared class-level default mapping in place, so
    # image tokens leak into every later instance (and into vocab sizes) in the same process.
    super().__init__()
    self._start_id = 3
    if is_multimodal:
      self.DEFAULT_MAPPING.update({
          '<img>': 22,
          '<soi>': 23,
          '<eoi>': 24,
          '<doubleline>': 25,
      })
    self._mapping_text_to_id = mapping_text_to_id or self.DEFAULT_MAPPING
    self._vocab_size = len(self._mapping_text_to_id)

  def pad_id(self) -> int:
    # [FUNCTION DOC]
    # HELPER: returns pad id 0.
    return 0

  def bos_id(self) -> int:
    # [FUNCTION DOC]
    # HELPER: returns BOS id 1.
    return 1

  def eos_id(self) -> int:
    # [FUNCTION DOC]
    # HELPER: returns EOS id 2.
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: returns the vocabulary size (mapping length).
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: decodes ids by reverse lookup in the mapping and joins the words with single spaces.
    # Raises KeyError for unknown ids.
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return ' '.join(reverse_mapping[e] for e in ids)

  def EncodeAsIds(self, text: str, **kwargs) -> list[int]:  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: encodes text by splitting on single spaces and mapping known words to ids; unknown
    # words are silently dropped; extra kwargs ignored. Output: list of ids.
    words = text.split(' ')
    res = [
        self._mapping_text_to_id[word]
        for word in words
        if word in self._mapping_text_to_id
    ]
    return res


class ToyTransformerWithScoreHead(nnx.Module):
  # [CLASS DOC]
  # HELPER: wraps a transformer backbone with a bias-free linear score head (head_dim to 1) applied
  # to the backbone's last hidden states, mimicking a reward/value model.
  """Toy transformer with a score head."""

  def __init__(self, transformer: nnx.Module, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # HELPER: stores the backbone and creates the score head with input size equal to the backbone's
    # head_dim.
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """

    self.transformer = transformer
    self.score = nnx.Linear(
        in_features=transformer.head_dim,
        out_features=1,
        use_bias=False,
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: runs the backbone with all given args plus output_hidden_states True (TypeError if the
    # caller also passes that keyword), pops the sown intermediates, takes the last element of the
    # "all_hidden_states" entry and applies the score head. Output: scores with trailing dim 1.
    # Side effects: removes intermediates from the backbone state.
    self.transformer(*args, **kwargs, output_hidden_states=True)
    hidden_states = nnx.pop(self.transformer, nnx.Intermediate)[
        'all_hidden_states'
    ].value[-1]
    score = self.score(hidden_states)
    return score


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def safe_list_files(repo_id):
  # [FUNCTION DOC]
  # HELPER: lists files of a HuggingFace repo, retried up to 3 attempts with exponential backoff
  # (4 to 10 seconds), re-raising the last error.
  return huggingface_hub.list_repo_files(repo_id)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def safe_download(repo_id, filename, local_dir):
  # [FUNCTION DOC]
  # HELPER: downloads one file from a HuggingFace repo into local_dir with the same retry policy;
  # returns the local file path.
  return huggingface_hub.hf_hub_download(
      repo_id=repo_id, filename=filename, local_dir=local_dir
  )


def download_from_huggingface(repo_id: str, model_path: str):
  # [FUNCTION DOC]
  # HELPER: prints a login reminder, lists the repo files, skips those under "original/",
  # downloads each into model_path via the retrying helper, and prints the list. Output: none.
  # Side effects: network and disk writes, stdout.
  """Download checkpoint files from huggingface."""
  print('Make sure you logged in to the huggingface cli.')

  all_files = safe_list_files(repo_id)
  filtered_files = [f for f in all_files if not f.startswith('original/')]

  for filename in filtered_files:

    safe_download(repo_id=repo_id, filename=filename, local_dir=model_path)
  print(f'Downloaded {filtered_files} to: {model_path}')


def batch_templatize(prompts: List[str], tokenizer: Any):
  # [FUNCTION DOC]
  # HELPER: asserts the tokenizer has apply_chat_template, then returns, for each prompt, the
  # untokenized chat-templated string of a single user message with a generation prompt appended.
  """Use tokenizer to batch templatize the prompts."""
  assert hasattr(tokenizer, 'apply_chat_template')
  out = []
  for p in prompts:
    out.append(
        tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
  return out


def validate_llm_outputs(
    expected_output_pattern: List[Tuple[str, List[str]]],
    serving_outputs: List[str],
):
  # [FUNCTION DOC]
  # HELPER: for each (prompt, keywords) pair zipped with generated outputs, asserts every keyword
  # (case-insensitive) appears in the stripped lowercased output; AssertionError naming the prompt
  # and keyword otherwise. Extra items in either list are ignored.
  for (prompt, expectations), generated in zip(
      expected_output_pattern, serving_outputs
  ):
    normalized = generated.strip().lower()
    for keyword in expectations:
      assert keyword.lower() in normalized, (
          f"Response '{generated}' for prompt '{prompt}' does not contain "
          f"expected keyword '{keyword}'."
      )


def delete_directory(path: str):
  # [FUNCTION DOC]
  # HELPER: deletes a directory tree if the path exists and is a directory, printing what happened;
  # prints a message (and does nothing) if the path is not a directory or does not exist.
  """Safely delete directory from filesystem."""
  if os.path.exists(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
      print(f'Deleted directory: {path}')
    else:
      print(f'Path exists but is not a directory: {path}')
  else:
    print(f'Directory does not exist: {path}')


def clear_jax_arrays():
  # [FUNCTION DOC]
  # HELPER: deletes every module-global variable of this module bound to a JAX array, then runs
  # garbage collection. Side effects: mutates module globals.
  # KNOWN ISSUE: it only inspects this test-utility module's own globals, not the caller's, so it
  # typically frees nothing.
  """Clear all the Jax arrays from hbm."""
  for name, obj in list(globals().items()):
    if isinstance(obj, jnp.ndarray):
      del globals()[name]
  gc.collect()


def is_running_in_colab() -> bool:
  # [FUNCTION DOC]
  # HELPER: returns True when an IPython module is loaded and its current shell has a "kernel"
  # attribute (Colab/Jupyter); False on missing module, missing function or attribute errors.
  """Checks if the code is running within a Colab IPython kernel."""
  try:
    # get_ipython() is defined in IPython. Check for 'kernel' attribute
    # which is characteristic of a Colab/Jupyter kernel.
    return hasattr(sys.modules['IPython'].get_ipython(), 'kernel')
  except (NameError, KeyError, AttributeError):
    return False
