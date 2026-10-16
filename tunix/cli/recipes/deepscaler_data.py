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
# Dataset loader for the DeepScaleR agentic GRPO recipe used by the CLI.
# Exposes a single function that reads the DeepScaleR training JSON (local path or any
# fsspec-supported URL such as GCS) and turns it into a grain map dataset whose elements are
# dictionaries with keys "prompts" (the question followed by a fixed step-by-step / boxed-answer
# instruction), "question" and "answer". Depends on the HuggingFace datasets library, fsspec,
# grain and pandas. No side effects at import time.

"""Dataset loading for the DeepScaleR agentic GRPO recipe.

Returns a raw grain.MapDataset ready for post_init_dataset().
Each element has keys: "prompts", "question", "answer".
"""

import datasets as datasets_lib
import fsspec
import grain
import pandas as pd


def create_dataset(
    train_data_path: str,
    eval_data_path: str,
    shuffle: bool = False,
    seed: int = 42,
) -> grain.MapDataset:
  # [FUNCTION DOC]
  # Builds the DeepScaleR training dataset for the agentic GRPO recipe.
  # Inputs: train_data_path - path/URL of a JSON file (records with "problem" and "answer" fields),
  # opened via fsspec and parsed with pandas JSON reader; eval_data_path - path/URL of a parquet
  # file opened in binary mode via fsspec and parsed with pandas; shuffle - whether to shuffle the
  # training rows; seed - shuffle seed.
  # Logic: defines an instruction string asking to think step by step and put the final answer
  # inside a LaTeX boxed command. Loads both files into DataFrames. Converts the training
  # DataFrame into a HuggingFace Dataset and maps each row (with index, ignored) to a dict with
  # "question" (from "problem"), "ground_truth" (from "answer") and "data_source" set to "math";
  # the original columns are kept by the HF map. Optionally shuffles with the seed. Wraps the result
  # as a grain MapDataset source and lazily maps each item to a dict with "prompts" equal to the
  # question, a space, then the instruction; "question"; and "answer" (prefers "question" over
  # "problem", and "answer" over "ground_truth", defaulting to empty strings).
  # Output: grain MapDataset of the training set only.
  # Exceptions: file-not-found / IO / parse errors from fsspec and pandas propagate.
  # Side effects: reads both files (network access for remote URLs).
  # KNOWN ISSUE: the eval parquet file is read into memory but never used or returned, despite
  # the docstring saying the function loads the AIME eval set.
  """Load DeepScaleR train set and AIME eval set.

  Args:
    train_data_path: Path to deepscaler.json (supports GCS via fsspec).
    eval_data_path: Path to AIME parquet (supports GCS via fsspec).
    shuffle: Whether to shuffle the training set.
    seed: Random seed for shuffling.

  Returns:
    grain.MapDataset where each element has "prompts", "question", "answer".
  """
  instruction = (
      "Let's think step by step, and put your final answer within \\boxed{}."
  )

  def _preprocess(example, index):
    # [FUNCTION DOC]
    # HELPER: per-row HF map callback. Input: a row dict and its index (discarded). Output: a new
    # dict with "question" taken from the row's "problem", "ground_truth" taken from the row's
    # "answer", and "data_source" fixed to "math". Raises KeyError if those fields are missing.
    # No side effects.
    del index
    return {
        "question": example["problem"],
        "ground_truth": example["answer"],
        "data_source": "math",
    }

  def _to_prompt(item):
    # [FUNCTION DOC]
    # HELPER: per-element grain map callback turning a record into the prompt format. Input: a
    # dict-like item. Reads the question (key "question", falling back to "problem", then empty
    # string) and the answer (key "answer", falling back to "ground_truth", then empty string).
    # Output: dict with "prompts" equal to question + space + the enclosing instruction string,
    # plus "question" and "answer". No exceptions or side effects.
    question = item.get("question", item.get("problem", ""))
    answer = item.get("answer", item.get("ground_truth", ""))
    return {
        "prompts": f"{question} {instruction}",
        "question": question,
        "answer": answer,
    }

  with fsspec.open(train_data_path) as f:
    train_df = pd.read_json(f)
  with fsspec.open(eval_data_path, "rb") as f:
    eval_df = pd.read_parquet(f)

  train_ds = datasets_lib.Dataset.from_pandas(train_df).map(
      _preprocess, with_indices=True
  )
  if shuffle:
    train_ds = train_ds.shuffle(seed)

  return grain.MapDataset.source(train_ds).map(_to_prompt)
