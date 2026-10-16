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
# GSM8K-style math dataset loading for RL (GRPO) examples. Loads data from TFDS (gsm8k),
# HuggingFace Datasets, or a local parquet file into grain datasets and optionally formats each
# record into a prompt containing a system prompt that asks for reasoning between reasoning tags
# and a single numeric answer between answer tags. Module constants: the four tag strings, the
# system prompt text, and a Gemma chat-turn template with system_prompt and question placeholders.
# Side effects at import: imports the TFDS gsm8k builder module to register it.

import logging
import os

import datasets as hf_datasets
import grain
import tensorflow_datasets as tfds
from tunix.utils import token_sanitization

# For OSS usage
import tensorflow_datasets.text.gsm8k

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def extract_hash_answer(text: str) -> str | None:
  # [FUNCTION DOC]
  # HELPER: extracts a GSM8K answer: returns None if the text has no "####"; otherwise the text
  # between the first and second "####" occurrence (i.e. after the first one), stripped, with commas removed.
  if "####" not in text:
    return None
  return text.split("####")[1].strip().replace(",", "")


# TODO(noghabi): Move these common dataset functions to a separate module.
def apply_template(
    dataset,
    tokenizer,
    tokenize=False,
    add_generation_prompt=True,
):
  # [FUNCTION DOC]
  # Formats each record of a dataset into prompt/question/answer.
  # Inputs: dataset (anything with map), tokenizer (optional), tokenize flag, add_generation_prompt flag.
  # Logic: if the tokenizer exists and has apply_chat_template, formatting uses it with a system
  # message (system prompt) and a user message (question) and the given flags; otherwise uses the
  # module template filled with the system prompt and question. Logs which path is taken. Each
  # record is copied to a dict, bytes values decoded as UTF-8, the question sanitized of control
  # tokens, and mapped to "prompts" (formatted), "question" (sanitized) and "answer" (hash answer
  # extraction of the record's answer).
  # Output: the mapped dataset. Exceptions: KeyError if question/answer missing. Side effects: logging.
  """Applies chat template with tokenizer to dataset."""

  _apply_template = None
  if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
    logging.info("Applying chat template with tokenizer to dataset")
    _apply_template = lambda question: tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )
  else:
    logging.info("Applying custom template to dataset")
    _apply_template = lambda question: TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT, question=question
    )

  def _process_element(x):
    # [FUNCTION DOC]
    # HELPER: per-record mapping described in the enclosing function's doc.
    item = dict(x)
    for key, value in item.items():
      if isinstance(value, bytes):
        item[key] = value.decode("utf-8")

    question = token_sanitization.sanitize_control_tokens(item["question"])
    return {
        "prompts": _apply_template(question),
        "question": question,
        "answer": extract_hash_answer(item["answer"]),
    }

  return dataset.map(_process_element)


def get_tfds_dataset(
    dataset_name: str,
    data_dir: str | None,
    download: bool,
    split: str,
    shuffle_seed: int = 42,
) -> grain.MapDataset:
  # [FUNCTION DOC]
  # Loads a TFDS dataset split as a shuffled grain MapDataset. Creates data_dir if given and
  # missing; opens the split as an ArrayRecord TFDS data source with the download flag; wraps it in
  # a MapDataset source and shuffles with shuffle_seed. Output: MapDataset. Side effects: directory
  # creation, downloads.
  """Get dataset from tfds.

  Args:
    dataset_name: The name of the dataset in tfds.
    data_dir: The directory to store the downloaded dataset.
    download: the download flag when using TFDS datasets.
    split: The dataset split to use (e.g., "train", "validation").
    shuffle_seed: The seed to use for shuffling the tfds dataset.

  Returns:
    A grain.MapDataset containing the processed dataset.
  """
  if data_dir and not os.path.exists(data_dir):
    os.makedirs(data_dir)

  data = tfds.data_source(
      dataset_name,
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=download,
  )

  dataset = grain.MapDataset.source(data).shuffle(seed=shuffle_seed)
  return dataset


def _parse_huggingface_dataset_name(
    dataset_name: str,
) -> tuple[str, str | None]:
  # [FUNCTION DOC]
  # HELPER: parses a HF dataset spec. "name:config" returns (name, config or None if empty);
  # otherwise a name containing "/" returns (name, "default"); otherwise (name, None).
  """Parses a Hugging Face dataset name into dataset/config components."""
  if ":" in dataset_name:
    name, config_name = dataset_name.split(":", maxsplit=1)
    return name, config_name or None

  if "/" in dataset_name:
    return dataset_name, "default"

  return dataset_name, None


def get_huggingface_dataset(
    dataset_name: str,
    split: str,
    shuffle_seed: int = 42,
) -> grain.MapDataset:
  # [FUNCTION DOC]
  # Loads a HuggingFace dataset split. Parses name/config, loads the split, shuffles it with the
  # seed, and wraps it as a grain MapDataset source. Output: MapDataset. Side effects: downloads.
  """Gets a dataset from Hugging Face Datasets."""
  hf_dataset_name, config_name = _parse_huggingface_dataset_name(dataset_name)
  data = hf_datasets.load_dataset(
      hf_dataset_name,
      config_name,
      split=split,
  )
  data = data.shuffle(seed=shuffle_seed)
  return grain.MapDataset.source(data)


def create_dataset(
    data_source: str,
    dataset: str,
    tokenizer=None,
    tfds_download: bool = True,
    split: str = "train",
    apply_chat_template_to_dataset: bool = False,
):
  # [FUNCTION DOC]
  # Creates a math dataset from a source.
  # Inputs: data_source ("local", "huggingface", "tfds"); dataset (parquet path, HF name, or "gsm8k");
  # tokenizer; tfds_download; split (default "train"); apply_chat_template_to_dataset.
  # Logic: "local" with a ".parquet" path builds a grain ParquetIterDataset; "huggingface" loads via
  # the HF loader; "tfds" with "gsm8k" loads via TFDS using data dir "./data/<split>" when
  # downloading, else None; anything else raises ValueError. If templating is requested, applies
  # the prompt template (untokenized).
  # Output: grain MapDataset or ParquetIterDataset (unbatched). Exceptions: ValueError.
  # Side effects: downloads, directory creation, logging.
  # KNOWN ISSUE: the docstring says "batched" and mentions directories for local data, but nothing
  # is batched and only paths ending in ".parquet" are accepted for "local".
  """Creates a dataset based on the given name.

  Args:
    data_source: The source of dataset. The currently supported options are
      'tfds' (load from tensorflow_datasets), 'huggingface' (load from
      Hugging Face Datasets), and 'local' (load local from parquet file).
    dataset: The name of the dataset to create. For 'tfds' data_source, the
      supported options are ['gsm8k']. For 'huggingface', this can be a dataset
      repo id like 'openai/gsm8k' or 'repo_id:config_name'. For 'local'
      data_source, this is the path to a parquet file or directory.
    tokenizer: The tokenizer to use for processing prompts. If no tokenizer is
      provided, the fixed template is used.
    tfds_download: the download flag when using TFDS datasets. If false, the
      data_dir used will be set to `None` and chosen by default by tfds.
    split: The dataset split to use (e.g., "train", "test").
    apply_chat_template_to_dataset: Whether to apply chat template to dataset.
      If true, the dataset is expected to have a "question" field and an
      "answer" field, and the output will have a "prompts" field with the
      applied template. If false, the dataset is returned as is without
      applying the template.

  Returns:
    A batched grain.MapDataset or grain.experimental.ParquetIterDataset.

  Raises:
    ValueError: If the dataset is not supported.
  """
  # parquet dataset
  if data_source == "local" and dataset.endswith(".parquet"):
    ds = grain.experimental.ParquetIterDataset(dataset)
  elif data_source == "huggingface":
    ds = get_huggingface_dataset(dataset_name=dataset, split=split)
  # tfds dataset
  elif data_source == "tfds" and dataset in ["gsm8k"]:
    data_dir = os.path.join("./data", split) if tfds_download else None
    ds = get_tfds_dataset(
        dataset_name=dataset,
        data_dir=data_dir,
        download=tfds_download,
        split=split,
    )
  else:
    raise ValueError(
        f"Unsupported combination of dataset='{dataset}' and"
        f" data_source='{data_source}'"
    )

  # Apply template
  if apply_chat_template_to_dataset:
    ds = apply_template(ds, tokenizer, tokenize=False)

  return ds
