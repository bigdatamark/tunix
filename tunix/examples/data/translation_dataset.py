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
# English-to-French translation dataset pipeline for SFT/PEFT examples. Loads either the TFDS
# "mtnt/en-fr" dataset or the HuggingFace "Helsinki-NLP/opus-100" dataset (en-fr config), and
# builds grain DataLoaders that tokenize source/target text using a prompt template, concatenate
# them into a fixed-length TrainingInput with a loss mask covering only target tokens, filter
# overlength examples, and batch with drop_remainder.
# Module constants: a plain prompt template (prefix asks to translate into French followed by a
# newline; suffix newline) and an instruction-tuned Gemma chat-turn template (user turn prefix
# and an end-of-turn plus model-turn suffix).

"""Data loading and preprocessing."""

from collections.abc import Iterable
from typing import Any

import datasets
from grain import python as grain
import numpy as np
import tensorflow_datasets as tfds
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft.peft_trainer import TrainingInput  # pylint: disable=g-importing-member


INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}

INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}


def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
    instruct_tuned: bool = False,
    tfds_download: bool = True,
    input_template: dict[str, str] | None = None,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  # [FUNCTION DOC]
  # Creates train and eval data loaders.
  # Inputs: dataset_name ("mtnt/en-fr" or "Helsinki-NLP/opus-100"); global_batch_size;
  # max_target_length; num_train_epochs (None means repeat indefinitely); tokenizer;
  # instruct_tuned (choose the chat template); tfds_download; input_template (ignored, see issue).
  # Logic: for MTNT imports the TFDS MTNT builder module (registration side effect) and opens the
  # "train" and "valid" splits as TFDS data sources; for OPUS-100 loads "train" and "validation"
  # with data_dir "en-fr"; otherwise ValueError. Chooses the template, then builds a training loader
  # with num_train_epochs and an eval loader with one epoch, both with the same batch size, length
  # and tokenizer.
  # Output: tuple (train_loader, eval_loader). Exceptions: ValueError for unsupported datasets;
  # download errors. Side effects: dataset download/reads.
  # KNOWN ISSUE: the input_template argument is always overwritten by the instruct_tuned choice,
  # so a caller-supplied template is ignored.
  """Creates train and eval data iterator.

  Args:
    dataset_name: The name of the dataset to use.
    global_batch_size: The global batch size to use for both train and eval.
    max_target_length: The maximum length of the target sequence.
    num_train_epochs: The number of epochs to use for training. If None, the
      dataset will be repeated indefinitely.
    tokenizer: The tokenizer to use for tokenizing the dataset.
    instruct_tuned: Whether the dataset should be instruct tuned.
    tfds_download: the download flag when using TFDS datasets.
    input_template: The input template to use for the dataset.

  Returns:
    A tuple of train and eval data iterators.
  """
  if dataset_name == "mtnt/en-fr":
    import tensorflow_datasets.translate.mtnt

    train_ds, eval_ds = tfds.data_source(
        dataset_name, split=("train", "valid"), download=tfds_download
    )
  elif dataset_name == "Helsinki-NLP/opus-100":  # Hugging Face dataloader
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, data_dir="en-fr", split=("train", "validation")
    )
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

  input_template = INPUT_TEMPLATE_IT if instruct_tuned else INPUT_TEMPLATE

  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  eval_loader = _build_data_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  return train_loader, eval_loader


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
    input_template: dict[str, str],
) -> grain.DataLoader:
  # [FUNCTION DOC]
  # HELPER: builds a grain DataLoader over a random-access source. Uses an index sampler over all
  # records with the given epoch count and no sharding (not shuffled), and operations: tokenize,
  # build fixed-length training input padded with the tokenizer's pad id, filter overlength,
  # then batch with drop_remainder True. Keyword-only inputs. Output: the DataLoader.
  """Builds a data loader for the given data source."""
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Tokenize(tokenizer, input_template),
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )


class _Tokenize(grain.MapTransform):
  # [CLASS DOC]
  # HELPER: grain map transform tokenizing a raw translation record into (source tokens, target
  # tokens). Holds the tokenizer and the prompt template.
  """Tokenize the input."""

  def __init__(
      self, tokenizer: tokenizer_lib.Tokenizer, input_template: dict[str, str]
  ):
    # [FUNCTION DOC]
    # HELPER: stores the tokenizer and template dict (keys "prefix" and "suffix").
    self._tokenizer = tokenizer
    self._input_template = input_template

  def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: tokenizes one record. MTNT records (have key "src") hold bytes in "src" and "dst",
    # which are UTF-8 decoded; OPUS-100 records hold a "translation" dict with "en" and "fr".
    # Source is tokenized with the template prefix/suffix and no EOS; target is tokenized with EOS.
    # Output: tuple of two token arrays. KeyError on unexpected schemas.
    """Tokenize the input."""
    if "src" in element.keys():  ## MTNT dataset
      src_tokens = self._tokenizer.tokenize(
          element["src"].decode(),
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["dst"].decode(), add_eos=True
      )
    else:  ## OPUS-100 dataset
      src_tokens = self._tokenizer.tokenize(
          element["translation"]["en"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["translation"]["fr"], add_eos=True
      )
    return src_tokens, dst_tokens


class _BuildTrainInput(grain.MapTransform):
  # [CLASS DOC]
  # HELPER: grain map transform turning (source, target) token arrays into a TrainingInput with
  # input_tokens and input_mask, right-padded to the max sequence length.
  """Build a TrainingInput from a tuple of source and destination tokens."""

  def __init__(self, max_seq_len: int, pad_value: int | bool):
    # [FUNCTION DOC]
    # HELPER: stores the max sequence length and the token pad value.
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value

  def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
    # [FUNCTION DOC]
    # HELPER: concatenates source and target tokens; builds a boolean mask that is False for source
    # positions and True for target positions; right-pads tokens with the pad value and the mask with
    # 0 (False) up to max length (no truncation if longer). Output: TrainingInput(input_tokens,
    # input_mask).
    src_tokens, dst_tokens = tokens

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = np.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)

    # If the input tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, 0)

    return TrainingInput(input_tokens=tokens, input_mask=mask)

  def _pad_up_to_max_len(
      self, input_tensor: np.ndarray, pad_value: int
  ) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: right-pads a 1-D array with a constant up to the max sequence length; arrays already
    # at or above the length are returned unpadded (pad amount clamped at 0).
    """Pad the given tensor up to sequence length of a batch."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )


class _FilterOverlength(grain.FilterTransform):
  # [CLASS DOC]
  # HELPER: grain filter transform dropping examples longer than the max sequence length.
  """Filter out overlength examples."""

  def __init__(self, max_seq_len: int):
    # [FUNCTION DOC]
    # HELPER: stores the max sequence length.
    self._max_seq_len = max_seq_len

  def filter(self, element: TrainingInput) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True when the example's input_tokens length is at most the max length.
    return element.input_tokens.shape[0] <= self._max_seq_len
