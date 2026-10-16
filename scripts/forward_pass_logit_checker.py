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
# CLI script that checks a Tunix model's logits against the Hugging Face reference implementation.
# Run as a module with --model_name (required, key of SUPPORTED_MODELS), --max_kl_div (float, default
# 0.015) and --max_target_length (int, default 1024); unknown args are ignored. Main flow: builds a
# 1 x N mesh ("fsdp","tp"), loads HF model (bfloat16) and tokenizer plus the Tunix model/tokenizer,
# then for 3 fixed prompts runs one HF forward pass and one Tunix Sampler call (1 generation step,
# echo, return_logits), prints top-10 token tables and overlap metrics for the last position, and
# asserts the per-token max KL divergence is below max_kl_div.
# SUPPORTED_MODELS maps model name to (HF repo id, Tunix loader taking a mesh and returning
# (config, model), Tunix tokenizer factory). Only "gemma3-27b-it" is registered.
# Side effects: sets jax_default_prng_impl to "unsafe_rbg", sets TF_CPP_MIN_LOG_LEVEL to "0" (the
# inline comment says suppress, but 0 shows all logs), downloads models, prints to stdout.
# KNOWN ISSUE: get_data is unused and references config attributes (global_batch_size_to_train_on,
# max_target_length) that the Tunix ModelConfig does not obviously provide.
"""Check if a Tunix supported model's logits match the HF logits."""
# python -m scripts.forward_pass_logit_checker --model_name=gemma3-27b-it

import argparse
import os

import jax
from jax.experimental import multihost_utils
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as gemma3_params_lib


def load_gemma3_model(mesh):
  # [FUNCTION DOC]
  # HELPER: builds the Gemma3 27B-IT ModelConfig and creates the model from the GEMMA3_27B_IT checkpoint
  # sharded on the given mesh. Input: mesh (jax Mesh). Output: tuple (config, model). Raises: checkpoint
  # loading errors propagate. Side effects: reads checkpoint, allocates device memory.
  config = (
      gemma3_model_lib.ModelConfig.gemma3_27b_it()
  )  # pick corresponding config based on model version
  return config, gemma3_params_lib.create_model_from_checkpoint(
      gemma3_params_lib.GEMMA3_27B_IT, config, mesh
  )


SUPPORTED_MODELS = {
    "gemma3-27b-it": (
        "google/gemma-3-27b-it",
        load_gemma3_model,
        gemma3_params_lib.create_tokenizer,
    )
}


def get_top_k_tokens_scores(
    logits_tensor, tokenizer_instance, k=10, description=""
):
  # [FUNCTION DOC]
  # Takes torch.topk of k entries from a 1-D logits tensor, decodes each id with the tokenizer, and
  # returns a list of dicts with keys id (int), token (decoded text stripped), score (float), in descending
  # score order. Inputs: logits_tensor (1-D torch tensor), tokenizer_instance (has decode), k (default 10),
  # description (label for printed header). Output: the list. Side effect: prints a header and a
  # markdown-style table of id/token/score. Raises: torch errors if k exceeds the vocab size.
  # KNOWN ISSUE: builds two identical lists (collected_tokens and tokens); one is redundant.
  """Get the top-k tokens and their scores from a given logits tensor."""
  print(f"\n--- {description} top {k} tokens ---")
  collected_tokens = []
  tokens = []
  topk_results = torch.topk(logits_tensor, k=k)
  for i in range(k):
    tok_id = topk_results.indices[i].item()
    score = topk_results.values[i].item()
    tok = tokenizer_instance.decode(tok_id)
    collected_tokens.append(
        {"id": int(tok_id), "token": tok.strip(), "score": float(score)}
    )
    tokens.append(
        {"id": int(tok_id), "token": tok.strip(), "score": float(score)}
    )

  # Prepare data for logging
  table_str = f"| {'Token ID':<10} | {'Token':<20} | {'Score':<10} |\n"
  table_str += f"|{'-'*12}|{'-'*22}|{'-'*12}|\n"
  for d in collected_tokens:
    table_str += f"| {d['id']:<10} | {d['token']:<20} | {d['score']:<10.4f} |\n"
  print(table_str)
  return tokens


def compare_top_tokens(converted_tokens, golden_tokens):
  # [FUNCTION DOC]
  # Compares two top-token lists (dicts with an "id" key). Computes: overlap count of id sets,
  # Jaccard similarity (intersection size divided by union size, 0.0 if union empty), and rank agreement
  # (percentage of positions i, up to the shorter length, where the ids match; 0.0 if that length is 0).
  # Prints a table of metrics, with overlap shown as "overlap/min_len". Output: none (returns None).
  # Raises: KeyError if an entry lacks "id". Side effects: stdout only.
  # Extract the sets of token IDs for comparison
  """Compares two lists of top tokens and calculates similarity metrics.

  Args:
      converted_tokens: top tokens from the converted model.
      golden_tokens:  top tokens from the golden model.
  """
  converted_ids = {token["id"] for token in converted_tokens}
  golden_ids = {token["id"] for token in golden_tokens}

  # --- Metric 1: Overlap Count & Jaccard Similarity ---
  intersection = converted_ids.intersection(golden_ids)
  union = converted_ids.union(golden_ids)

  overlap_count = len(intersection)
  jaccard_similarity = overlap_count / len(union) if union else 0.0

  # --- Metric 2: Rank Agreement ---
  rank_matches = 0
  min_len = min(len(converted_tokens), len(golden_tokens))
  for i in range(min_len):
    if converted_tokens[i]["id"] == golden_tokens[i]["id"]:
      rank_matches += 1

  rank_agreement = (rank_matches / min_len) * 100 if min_len > 0 else 0.0

  metrics = {
      "overlap_count": f"{overlap_count}/{min_len}",
      "jaccard_similarity": jaccard_similarity,
      "rank_agreement_percentage": rank_agreement,
  }

  print("\n--- Similarity Metrics of Top Tokens ---")
  table_str = f"| {'Metric':<30} | {'Value':<20} |\n"
  table_str += f"|{'-'*32}|{'-'*22}|\n"
  for key, value in metrics.items():
    table_str += f"| {key:<30} | {str(value):<20} |\n"
  print(table_str)


def check_kl_divergence(model_logits, golden_logits, atol=0.02):
  # [FUNCTION DOC]
  # Computes KL(P_golden || Q_model) per token between two 2-D logit tensors (SeqLen, Vocab). Slices both
  # to the vocabulary width given by min of their dim 1 sizes, reshapes to (s, v), softmaxes golden and
  # log-softmaxes model, prints the batchmean KL (average per row) and the max per-row KL (sum over vocab
  # with reduction none). Asserts the max is below atol (default 0.02), raising AssertionError otherwise.
  # Output: none. Side effects: prints.
  # KNOWN ISSUE: docstring says inputs are (Batch, SeqLen, VocabSize) but the code unpacks exactly two
  # dimensions and slices with shape[1], so 3-D inputs would fail; it only works for 2-D (seq, vocab).
  # Also the token_size slice is on the last axis but computed from shape[1]; correct only for 2-D.
  # 1. Select the relevant vocabulary slice from the logits.
  """Calculates KL divergence D_KL(P_golden || Q_model) over a batch of sequences.

  Args:
      model_logits: Logits from the converted model (Batch, SeqLen, VocabSize).
      golden_logits: Logits from the golden model (Batch, SeqLen, VocabSize).
      atol: Maximum allowed KL divergence per token.
  """
  token_size = min(model_logits.shape[1], golden_logits.shape[1])
  model_logits_sliced = model_logits[..., :token_size]
  golden_logits_sliced = golden_logits[..., :token_size]

  # 2. Reshape
  s, v = model_logits_sliced.shape
  model_logits_reshaped = model_logits_sliced.view(s, v)
  golden_logits_reshaped = golden_logits_sliced.view(s, v)

  # 3. Get the probability distributions.
  golden_probabilities = F.softmax(golden_logits_reshaped, dim=-1)
  model_log_probabilities = F.log_softmax(model_logits_reshaped, dim=-1)

  # 4. Calculate avg KL divergence for all token distributions.
  # use 'batchmean'; the sum of the KL divergences for each token in the batch
  # and then divides by the number of tokens (s)
  kl_div_value = F.kl_div(
      input=model_log_probabilities,
      target=golden_probabilities,
      reduction="batchmean",  # Use 'batchmean' for the average KL per token.
      log_target=False,
  )

  print(
      "\nAverage KL divergence per token (D_KL(P_golden || Q_model)): "
      f"{kl_div_value.item():.6f}"
  )

  # To find the max KL divergence for any single token in the set
  # use reduction='none'.
  kl_divs_per_token = F.kl_div(
      input=model_log_probabilities,
      target=golden_probabilities,
      reduction="none",
      log_target=False,
  ).sum(
      dim=-1
  )  # Sum over the vocab dim to get a single KL value per token

  max_kl_div = kl_divs_per_token.max()
  print(
      "\nMax KL divergence for a single token in the set: "
      f"{max_kl_div.item():.6f}"
  )

  assert (
      max_kl_div < atol
  ), f"KL divergence values {max_kl_div.item():.6f} exceed the threshold {atol}"


def get_data(golden_data, golden_data_index, config):
  # [FUNCTION DOC]
  # Prepares one golden-data example for a forward pass. golden_data is indexable; entry golden_data_index
  # must have "tokens", "logits", "prompt". Converts tokens to int32, raises ValueError if their length
  # exceeds config.max_target_length, zero-pads to max_target_length, tiles to
  # config.global_batch_size_to_train_on rows, builds segment ids (1 for real tokens, 0 for padding) and
  # positions (0..max_target_length-1 per row), and converts logits to float32.
  # Output: tuple (ids, decoder_segment_ids, decoder_positions, logits, seq_len). Side effects: prints.
  # KNOWN ISSUE: unused by this script.
  """Get the golden data for the test indexed at golden_data_index."""

  print(f"Comparing forward pass for golden data index = {golden_data_index}")
  print(
      "config.global_batch_size_to_train_on="
      f"{config.global_batch_size_to_train_on}"
  )

  original_ids = np.asarray(
      golden_data[golden_data_index]["tokens"], dtype=np.int32
  )
  seq_len = len(original_ids)

  if seq_len > config.max_target_length:
    raise ValueError(
        f"Golden data sequence length ({seq_len}) is greater than "
        f"max_target_length ({config.max_target_length})"
    )

  s = (config.global_batch_size_to_train_on, config.max_target_length)

  # Pad ids to max_target_length. Tunix expects 0 for padding.
  padded_ids = np.pad(
      original_ids,
      (0, config.max_target_length - seq_len),
      "constant",
      constant_values=0,
  )
  ids = np.stack(
      [padded_ids for _ in range(config.global_batch_size_to_train_on)]
  )

  logits = np.asarray(
      golden_data[golden_data_index]["logits"], dtype=np.float32
  )
  print(
      f" prompt=\"{golden_data[golden_data_index]['prompt']}\" raw "
      f"ids={original_ids}, logits.shape = {logits.shape}"
  )

  decoder_segment_ids = np.zeros(s, dtype=np.int32)
  decoder_segment_ids[:, :seq_len] = 1
  decoder_positions = np.stack([
      np.arange(config.max_target_length, dtype=np.int32)
      for _ in range(config.global_batch_size_to_train_on)
  ])

  print(
      f"ids={ids}, decoder_segment_ids = {decoder_segment_ids}, "
      f"decoder_positions= {decoder_positions}"
  )

  return ids, decoder_segment_ids, decoder_positions, logits, seq_len


def convert_jax_weight_to_torch(
    weight: "jax.Array", dtype: str | None = None
) -> torch.Tensor:
  # [FUNCTION DOC]
  # HELPER: converts a (possibly multi-host sharded) JAX array to a torch tensor. Records target dtype
  # (the array's dtype string unless dtype is given) and shape, all-gathers across processes, converts to a
  # float32 numpy array, then to torch with the dtype looked up by name on the torch module and reshaped to
  # the original shape. Output: torch.Tensor. Raises AttributeError if the dtype name is not a torch attr.
  expected_dtype = str(weight.dtype) if dtype is None else dtype
  expected_shape = weight.shape
  weight = multihost_utils.process_allgather(weight)
  weight = np.array(weight, dtype="float32")
  torch_dtype = getattr(torch, expected_dtype)
  torch_array = torch.from_numpy(weight).to(torch_dtype).reshape(expected_shape)
  return torch_array


def main(test_args):
  # [FUNCTION DOC]
  # Script entry point; see module doc for the full flow. Input: parsed args with model_name,
  # max_target_length, max_kl_div. Per prompt: HF tokenizes (truncation to max_target_length) and takes
  # logits of batch item 0; Tunix sampler cache size is max_target_length+5; Tunix logits for item 0 are
  # sliced to the last actual_seq_len positions (HF sequence length) and converted to torch; top-10 tables
  # for the final position are printed and compared; KL check runs on the full sequence.
  # Output: none. Raises: KeyError for unknown model, AssertionError on KL failure.
  # KNOWN ISSUE: assumes HF and Tunix tokenization produce the same length sequence; also mesh is assigned
  # twice in a chained assignment (harmless).
  devices = jax.devices()
  print(f"Running on devices: {devices}")
  mesh = mesh = jax.make_mesh(
      (1, len(devices)),
      ("fsdp", "tp"),
      axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
  )
  hf_name, tunix_model_loader, tunix_tokenizer_loader = SUPPORTED_MODELS[
      test_args.model_name
  ]

  hf_model = AutoModelForCausalLM.from_pretrained(
      hf_name, torch_dtype=torch.bfloat16
  )
  tokenizer = AutoTokenizer.from_pretrained(hf_name)

  config, tunix_model = tunix_model_loader(mesh)
  tunix_tokenizer = tunix_tokenizer_loader()
  sampler = sampler_lib.Sampler(
      tunix_model,
      tunix_tokenizer,
      sampler_lib.CacheConfig(
          cache_size=test_args.max_target_length + 5,
          num_layers=config.num_layers,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
      ),
  )
  print(
      f"Tokenizer vocab size: {tokenizer.vocab_size} vs "
      f"{tunix_tokenizer.vocab_size()}"
  )
  prompts = ["I love to", "Today is a", "What is the"]
  for input_text in prompts:
    print(f"\n--- Prompt: {input_text} ---")

    # Tokenize for HF
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        max_length=test_args.max_target_length,
        truncation=True,
    )
    actual_seq_len = inputs["input_ids"].shape[1]

    # --- HF Forward Pass ---
    with torch.no_grad():
      hf_logits_torch = hf_model(**inputs).logits[0]

    # --- Tunix Forward Pass ---
    output = sampler(
        [input_text],
        max_generation_steps=1,
        max_prompt_length=test_args.max_target_length,
        echo=True,
        return_logits=True,
    )
    tunix_logits_jax = output.logits[0]
    tunix_logits_jax_sliced = tunix_logits_jax[-actual_seq_len:, :]
    tunix_logits_torch = convert_jax_weight_to_torch(tunix_logits_jax_sliced)

    # --- Compare logits for the last token prediction ---
    hf_last_token_logits = hf_logits_torch[-1, :]
    tunix_last_token_logits = tunix_logits_torch[
        -1, :
    ]  # Tunix output already sliced to actual_seq_len

    tokens_tunix = get_top_k_tokens_scores(
        tunix_last_token_logits, tokenizer, k=10, description="Tunix model"
    )
    tokens_hf = get_top_k_tokens_scores(
        hf_last_token_logits, tokenizer, k=10, description="HF model"
    )
    compare_top_tokens(converted_tokens=tokens_tunix, golden_tokens=tokens_hf)

    # --- Compare all logits in the sequence (for the first batch item) ---
    # Unsqueeze to add batch dimension for check_kl_divergence: [1, seq, vocab]
    check_kl_divergence(
        tunix_logits_torch, hf_logits_torch, atol=test_args.max_kl_div
    )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_name", type=str, required=True, help="Name of the model to test."
  )
  parser.add_argument("--max_kl_div", type=float, required=False, default=0.015)
  parser.add_argument(
      "--max_target_length", type=int, required=False, default=1024
  )
  args, _ = parser.parse_known_args()

  assert (
      args.model_name in SUPPORTED_MODELS.keys()
  ), f"Model {args.model_name} is not supported."

  main(args)
