# [MODULE DOC]
# Notebook-style ("# %%" cells) math evaluation for Qwen2-family models (Qwen2.5-1.5B-Instruct,
# DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview) on MATH-500 and AIME-2024. Detects an
# internal "g3" environment (gfile + ecolab adhoc reload, file_open = gfile.Open) or "git" (null
# context, file_open = fsspec.open, data/models under gs://tunix). Defines answer extraction and
# grading helpers and Qwen25MathEvaluator, then (top-level) evaluates DeepSeek-R1-Distill-Qwen-1.5B on
# MATH-500 (prompt 1024, generation 1024, batch 8) and DeepScaleR-1.5B-Preview on AIME-2024 (prompt
# 2048, generation 32768, batch 1) on a (1,2) ("fsdp","tp") mesh, with temperature 0.6, top_p 0.95,
# printing accuracy. NUM_BATCHES env var (positive int) limits the number of batches.
# Side effects: GCS/HF reads, device memory, heavy stdout logging.
# %%
from pprint import pprint
import datasets as datasets_lib
import grain
import pandas as pd
import os
import fsspec

import transformers
from tunix.generate import mappings

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )

  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"

  import contextlib
  cm = contextlib.nullcontext()

  file_open = fsspec.open

with cm:
  from tunix.models.qwen2 import model as qwen2_lib
  from tunix.models.qwen2 import params as qwen2_params_lib
  from tunix.generate import sampler as sampler_lib
  from tunix.utils import math_utils
# %%
from typing import Any, Dict, Optional
import jax
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from tqdm.auto import tqdm
import re


def has_safetensors(path):
  # [FUNCTION DOC]
  # HELPER: returns True if any file under path (recursive walk; gfile in g3, else the fsspec filesystem
  # for the path) ends with ".safetensors", else False.
  if NOTEBOOK_ENV == "g3":
    for _, _, filenames in gfile.Walk(path):
      for filename in filenames:
        if filename.endswith(".safetensors"):
          return True
    return False
  else:
    # fsspec for gs:// paths
    fs, _, _ = fsspec.core.get_fs_token_paths(path)
    for _, _, filenames in fs.walk(path):
      for filename in filenames:
        if filename.endswith(".safetensors"):
          return True
    return False


# Only used for Math500
def extract_answer_robust(passage: str) -> str:
  # [FUNCTION DOC]
  # Extracts a final answer from a model response (used for MATH-500). Empty input gives "".
  # 1) Finds the first "\boxed", skips whitespace, and if followed by an opening brace returns the
  # stripped content up to the matching closing brace (brace counting handles nesting).
  # 2) Otherwise tries, in order, lenient regexes (case-insensitive, dotall): boxed with backslash,
  # boxed without backslash, boxed content up to a sentence end, and "final answer is"/"answer is"
  # followed by boxed; uses the last match, strips trailing punctuation and closing parens, and if it
  # contains a fraction command appends closing braces to balance unmatched opening braces.
  # 3) Otherwise matches "boxed" followed by up to 200 non-newline chars and cuts at the first of
  # ".", ")", newline, "The ", "Thus", "Therefore" (checked in that order), then strips punctuation.
  # Returns "" if nothing matches. Output: str (never None).
  # KNOWN ISSUE: the local variable stack is unused; step 1 uses only the first boxed occurrence, not the
  # last.
  if not passage:
    return ""

  # Pattern 1: Look for \boxed{...} with proper matching braces
  # This handles nested braces like \boxed{\frac{1}{2}}
  stack = []
  i = passage.find("\\boxed")
  if i != -1:
    i += 6  # Skip '\boxed'
    # Skip whitespace
    while i < len(passage) and passage[i].isspace():
      i += 1
    if i < len(passage) and passage[i] == "{":
      i += 1
      start = i
      brace_count = 1
      while i < len(passage) and brace_count > 0:
        if passage[i] == "{":
          brace_count += 1
        elif passage[i] == "}":
          brace_count -= 1
        i += 1
      if brace_count == 0:
        answer = passage[start : i - 1]
        return answer.strip()

  # Pattern 2: Lenient matching - extract up to common terminators
  patterns = [
      r"\\boxed\{([^}]+)\}",  # Standard
      r"boxed\{([^}]+)\}",  # Missing backslash
      r"\\boxed\s*\{(.+?)(?:\.\s|\)\.|\.$)",  # Ends with period
      r"final answer is.*?\\boxed\{([^}]+)",  # "final answer is"
      r"answer is.*?\\boxed\{([^}]+)",
  ]

  for pattern in patterns:
    matches = re.findall(pattern, passage, re.IGNORECASE | re.DOTALL)
    if matches:
      answer = matches[-1].strip()
      # Clean up
      answer = answer.rstrip(".,;:)")
      # Try to fix common LaTeX issues
      if "\\frac" in answer:
        # Count braces - each \frac needs 2 pairs
        open_braces = answer.count("{")
        close_braces = answer.count("}")
        if open_braces > close_braces:
          answer += "}" * (open_braces - close_braces)
      return answer

  # Pattern 3: Super lenient - just find anything after boxed{
  super_lenient = r"boxed\s*\{([^\n]{1,200})"
  matches = re.findall(super_lenient, passage, re.IGNORECASE)
  if matches:
    answer = matches[-1]
    # Find the first reasonable endpoint
    for char in [".", ")", "\n", "The ", "Thus", "Therefore"]:
      if char in answer:
        answer = answer[: answer.index(char)]
        break
    return answer.strip().rstrip(".,;:)")

  return ""
# %%

# only used for AIME-2024
THOUGHT_DELIMITER_END = "</think>"
def evaluate_correctness(response: Any, ground_truths: Any) -> bool:
  # [FUNCTION DOC]
  # Grades a response for AIME-2024. None or empty response gives False. If "</think>" occurs, the
  # solution is the text after the first delimiter (the second split segment), else the whole response.
  # Extracts the answer via math_utils.extract_answer (None gives False). ground_truths None gives
  # False; a scalar str/float/int is wrapped in a list; each truth is stringified and, if it contains
  # boxed, replaced by its extracted answer (dropped if extraction fails). Returns True if any processed
  # truth matches by grade_answer_mathd, grade_answer_sympy or grade_answer_special_handling, else
  # False. Side effect: prints diagnostics.
  # KNOWN ISSUE: with multiple "</think>" delimiters only the segment between the first and second is
  # graded.
  """Evaluate the correctness of a response."""
  if response is None or response == "":
    print(f"{response=} {ground_truths=} IS NOT CORRECT")
    return False
  if THOUGHT_DELIMITER_END in response:
    response = response.split(THOUGHT_DELIMITER_END)
    model_solution = response[1]
    print(f"{model_solution=} after THOUGHT_DELIMITER_END in evaluate_correctness")
  else:
    print(f"{response=} in evaluate_correctness")
    model_solution = response
  model_answer = math_utils.extract_answer(model_solution)
  if model_answer is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  if ground_truths is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  # Convert single answer to list for uniform processing
  if isinstance(ground_truths, str | float | int):
    ground_truths = [ground_truths]
  # Process each ground truth
  processed_ground_truths = []
  for truth in ground_truths:
    truth = str(truth)
    if "\\boxed" in truth:
      processed_truth = math_utils.extract_answer(truth)
      if processed_truth is not None:
        processed_ground_truths.append(processed_truth)
    else:
      processed_ground_truths.append(truth)
  print(f"{processed_ground_truths=} in evaluate_correctness")
  if not processed_ground_truths:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  # Check against all possible correct answers
  for ground_truth in processed_ground_truths:
    is_correct = (
        math_utils.grade_answer_mathd(model_answer, ground_truth)
        or math_utils.grade_answer_sympy(model_answer, ground_truth)
        or math_utils.grade_answer_special_handling(model_answer, ground_truth)
    )
    if is_correct:
      print(f" {model_answer=} {ground_truth=} IS CORRECT")
      return True
  print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
  return False
# %%

class Qwen25MathEvaluator:
  # [CLASS DOC]
  # Evaluator holding config (model_config, model_version (HF id), model_path, dataset path, prompt and
  # generation lengths, sampler_type vanilla|sglang_jax|vllm), a JAX mesh, tokenizer, model and the
  # chosen sampler (sampler_vanilla, sampler_sglang or sampler_vllm attribute).

  def __init__(
      self,
      model_config,
      model_version: str,
      model_path: str,
      dataset: str,
      mesh_config=None,
      max_prompt_length: int = 1024,  # Increased from 512
      max_generation_steps: int = 1024,  # Increased from 512
      sampler_type: str = "vanilla",  # vanilla, vllm, or sglang-jax
  ):
    # [FUNCTION DOC]
    # Stores arguments, builds the mesh from mesh_config (default [[1,4],["fsdp","tp"]]) with Auto axis
    # types, sets tokenizer/model/sampler to None and prints setup info. Raises if the mesh needs more
    # devices than are available.
    self.model_config = model_config
    self.model_version = model_version
    self.model_path = model_path
    self.dataset = dataset
    self.max_prompt_length = max_prompt_length
    self.max_generation_steps = max_generation_steps
    self.sampler_type = sampler_type

    if mesh_config is None:
      # Default: 4-way tensor parallelism
      mesh_config = [[1, 4], ["fsdp", "tp"]]
    self.mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))
    self.tokenizer = None
    self.model = None
    self.sampler = None

    print(f"Initializing {self.model_version} evaluator")
    print(f"Model path: {model_path}")
    print(f"Mesh config: {mesh_config}")
    print(f"Available devices: {jax.devices()}")

  def model_from_safe_tensors(self):
    # [FUNCTION DOC]
    # HELPER: inside the mesh context, loads the Qwen2 model from safetensors in model_path into self.model.
    print("Loading model from safe tensors...")
    with self.mesh:
      self.model = qwen2_params_lib.create_model_from_safe_tensors(
          file_dir=self.model_path, config=self.model_config, mesh=self.mesh
      )

  def model_from_orbax_ckpt(self):
    # [FUNCTION DOC]
    # HELPER: builds an abstract Qwen2 via eval_shape, maps its state to bfloat16 ShapeDtypeStructs with
    # named shardings, opens an Orbax CheckpointManager at model_path with model_params and
    # optimizer_state PyTree handlers, restores only model_params at step 160, and merges the restored
    # state into self.model.
    # KNOWN ISSUE: the checkpoint step 160 is hard-coded.
    print(f"Loading model from orbax checkpoint {self.model_path}...")
    with self.mesh:
      abs_model: nnx.Module = nnx.eval_shape(
          lambda: qwen2_lib.Qwen2(self.model_config, rngs=nnx.Rngs(params=0))
      )
      abs_state = nnx.state(abs_model)
      abs_state = jax.tree.map(
          lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
          abs_state,
          nnx.get_named_sharding(abs_state, self.mesh),
      )
      item_handlers = {
          "model_params": ocp.PyTreeCheckpointHandler(),
          "optimizer_state": ocp.PyTreeCheckpointHandler(),
      }
      checkpointer = ocp.CheckpointManager(
          self.model_path,
          item_handlers=item_handlers,
      )
      model_cp_args = ocp.args.PyTreeRestore(
          item=abs_state,
          restore_args=ocp.checkpoint_utils.construct_restore_args(
              target=abs_state
          ),
      )
      ckpt = checkpointer.restore(
          160,
          args=ocp.args.Composite(
              model_params=model_cp_args,
          ),
      )
      graphdef, _ = nnx.split(abs_model)
      new_state = nnx.State(ckpt.model_params)
      self.model = nnx.merge(graphdef, new_state)

  def load_model(self):
    # [FUNCTION DOC]
    # Loads the HF tokenizer (from model_version in git env, model_path in g3; trust_remote_code), loads
    # the model from safetensors if any exist under model_path else from the Orbax checkpoint, and
    # creates the sampler for sampler_type with context prompt + generation + 100: vanilla Sampler with a
    # KV cache config; sglang_jax (via internal stub module, mem fraction 0.4, radix cache off) or vllm
    # (internal stub, HBM 0.8, max 30 seqs) with weights synced from the model. Raises ValueError for an
    # unknown type. Output: dict with model, tokenizer, sampler and config.
    # KNOWN ISSUE: the returned "sampler" entry is always None because the sampler is stored under
    # type-specific attribute names; the sglang/vllm branches import internal-only stub modules.
    print("Loading model components...")

    print("Loading tokenizer...")

    # Huggingface API doesn't work with gcs, OSS loads from model directly
    tokenizer_source = (
        self.model_version if NOTEBOOK_ENV != "g3" else self.model_path
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )

    print("Setting up model config...")

    if has_safetensors(self.model_path):
      self.model_from_safe_tensors()
    else:
      self.model_from_orbax_ckpt()
    print("Model loaded successfully!")
    print("Creating sampler...")
    cache_config = sampler_lib.CacheConfig(
        cache_size=self.max_prompt_length + self.max_generation_steps + 100,
        num_layers=self.model_config.num_layers,
        num_kv_heads=self.model_config.num_kv_heads,
        head_dim=self.model_config.head_dim,
    )

    if self.sampler_type == "vanilla":
      self.sampler_vanilla = sampler_lib.Sampler(
          transformer=self.model,
          tokenizer=self.tokenizer,
          cache_config=cache_config,
      )
    elif self.sampler_type == "sglang_jax":
      from tunix.google.stubs import sglang_jax_sampler_stub as sglang_jax_sampler  # pylint: disable=g-import-not-at-top

      mapping_config = mappings.MappingConfig.build(
          mapping_obj=None,
          model=self.model,
          backend="sglang_jax",
      )
      self.sampler_sglang = sglang_jax_sampler.SglangJaxSampler(
          tokenizer=self.tokenizer,
          config=sglang_jax_sampler.SglangJaxConfig(
              mesh=self.mesh,
              context_length=self.max_prompt_length
              + self.max_generation_steps
              + 100,
              model_version=self.model_version,
              mem_fraction_static=0.4,
              init_with_random_weights=False,
              disable_radix_cache=True,
              enable_deterministic_sampling=False,
              mapping_config=mapping_config,
          ),
      )
      # sync weights from self.model to the sampler's internal model
      print("Syncing model weights to SGLang JAX sampler...")
      self.sampler_sglang.update_params(nnx.state(self.model))
    elif self.sampler_type == "vllm":
      from tunix.google.stubs import vllm_sampler_stub as vllm_sampler  # pylint: disable=g-import-not-at-top

      mapping_config = mappings.MappingConfig.build(
          mapping_obj=None,
          model=self.model,
          backend="vllm_jax",
      )
      self.sampler_vllm = vllm_sampler.VllmSampler(
          tokenizer=self.tokenizer,
          config=vllm_sampler.VllmConfig(
              mesh=self.mesh,
              hbm_utilization=0.8,
              init_with_random_weights=False,
              mapping_config=mapping_config,
              engine_kwargs={
                  "model": self.model_version,
                  "max_model_len": (
                      self.max_prompt_length + self.max_generation_steps + 100
                  ),
                  "max_num_seqs": 30,
                  "max_num_batched_tokens": 30 * 10 * 1024 // 8,
              },
          ),
      )
      # sync weights from self.model to the sampler's internal model
      print("Syncing model weights to VLLM sampler...")
      self.sampler_vllm.update_params(nnx.state(self.model))
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")

    print("Sampler created successfully!")

    return {
        "model": self.model,
        "tokenizer": self.tokenizer,
        "sampler": self.sampler,
        "config": self.model_config,
    }

  def load_dataset(self, split: str = "test") -> grain.MapDataset:
    # [FUNCTION DOC]
    # Reads the dataset file (JSONL, JSON or otherwise Parquet) via file_open into pandas, converts to a
    # HF Dataset mapped to {question: problem, answer: answer, data_source: "math"}, prints the first
    # example, and wraps it in a grain MapDataset mapped by process_item, which builds a chat-templated
    # prompt string (generation prompt, untokenized): for AIME the question followed by a "think step by
    # step, boxed answer" instruction, otherwise the instruction followed by the question. Prints the
    # first formatted prompt. The split argument is only printed. Output: grain MapDataset of dicts
    # with prompt, question, answer. Raises KeyError if problem/answer columns are missing.
    print(f"Loading {self.dataset} dataset (split: {split})...")

    def preprocess_fn(example, idx):
      # [FUNCTION DOC]
      # HELPER: maps a raw row to {question, answer, data_source "math"}; idx unused.
      return {
          "question": example["problem"],
          "answer": example["answer"],
          "data_source": "math",
          }

    with file_open(self.dataset, "rb") as test_f:
      if self.dataset.endswith("jsonl"):
        test_df = pd.read_json(test_f, lines=True)
      elif self.dataset.endswith("json"):
        test_df = pd.read_json(test_f)
      else:
        test_df = pd.read_parquet(test_f)

    test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)

    print(f"Loaded {len(test_ds)} examples")
    print("Example data:")
    pprint(test_ds[0])

    def process_item(item):
      # [FUNCTION DOC]
      # HELPER: builds the chat-templated prompt for one item as described in load_dataset and returns
      # {prompt, question, answer}.
      question = item["question"]
      answer = item["answer"]

      if "aime_2024" in self.dataset:
        instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        prompt = f"{question} {instruction}"
      else:
        instruction = "Please reason step by step. Your final answer must appear inside \\boxed{...} and nothing else."
        prompt = f"{instruction} {question}"
      prompt = self.tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False, add_generation_prompt=True)

      return {
          "prompt": prompt,
          "question": question,
          "answer": answer,
      }

    dataset = grain.MapDataset.source(test_ds).map(process_item)
    print("\n" + "=" * 60)
    print("DEBUG: First formatted prompt:")
    first_item = dataset[0]
    print(first_item["prompt"])
    print("=" * 60 + "\n")

    return dataset

  def generate(
      self,
      prompts: list[str],
      temperature: float = 0.6,
      top_k: int = 50,
      top_p: float = 0.95,
      seed: int | None = None,
  ) -> list[str]:
    # [FUNCTION DOC]
    # Generates completions for a list of prompt strings. Raises RuntimeError if the tokenizer is not
    # loaded. Computes the longest prompt token length and a safe generation length = min(max generation
    # steps, cache size - longest prompt - 100) with cache size = prompt + generation + 100, warning if
    # below 256. Uses the first token id of "<|im_end|>" as EOS for vanilla. Calls the selected sampler
    # (vanilla with a PRNGKey from seed; sglang_jax with integer seed and padded output; vllm with no seed),
    # without echo. Output: list of generated texts. Raises ValueError for unknown sampler types.
    # KNOWN ISSUE: a long prompt can make the safe length zero or negative, which is passed through.
    if self.tokenizer is None:
      raise RuntimeError(
          "Model components not loaded. Call load_model() first."
      )
    max_length = max(len(self.tokenizer.encode(p)) for p in prompts)
    cache_size = self.max_prompt_length + self.max_generation_steps + 100
    safe_gen_length = min(
        self.max_generation_steps,
        cache_size - max_length - 100,  # 100 token buffer
    )
    if safe_gen_length < 256:
      print(
          f"WARNING: Short generation length ({safe_gen_length} tokens) due to"
          f" long prompt ({max_length} tokens)"
      )

    stop_token_id = self.tokenizer.encode("<|im_end|>")[0]

    # Generate
    if self.sampler_type == "vanilla":
      out_data = self.sampler_vanilla(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          echo=False,
          eos_tokens=[stop_token_id],
          seed=jax.random.PRNGKey(seed) if seed is not None else None,
      )
    elif self.sampler_type == "sglang_jax":
      out_data = self.sampler_sglang(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          max_prompt_length=self.max_prompt_length,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          seed=seed,
          echo=False,
          pad_output=True,
      )
    elif self.sampler_type == "vllm":
      out_data = self.sampler_vllm(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          max_prompt_length=self.max_prompt_length,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          seed=None,
          echo=False,
          pad_output=True,
      )
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")
    return out_data.text

  def evaluate(
      self,
      batch_size: int = 8,
      num_batches: int | None = None,
      temperature: float = 0.6,
      top_k: Optional[int] = 50,
      top_p: Optional[float] = 0.95,
      num_passes: int = 1,
      debug_first_n: int = 3,  # NEW: Debug first N examples
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Runs the evaluation: prints config, loads the dataset, batches it (optionally truncated to
    # num_batches), and for each batch generates num_passes responses per prompt (seed = pass index except
    # for vllm). For each question, iterates responses until one is correct: AIME datasets use
    # evaluate_correctness; others use extract_answer_robust then grade_answer_mathd or grade_answer_sympy.
    # A question counts as correct if the last graded response is correct. Prints detailed debug output
    # for the first debug_first_n questions and progress every 10. Output: dict with correct, total,
    # accuracy (percent, 0 if empty), sampling settings, num_passes and detailed_results (question,
    # answer, responses, extracted answers, correct).
    # KNOWN ISSUE: for AIME the extracted_answers list stays empty, so per-response debug output is skipped;
    # the "first 300 chars" debug label prints the full prompt; the None check on extract_answer_robust's
    # result never triggers because it returns "" instead.
    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {num_batches or 'all'}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Passes per question: {num_passes}")
    print(f"  Debug first N examples: {debug_first_n}")
    print("=" * 60)

    # Load dataset
    dataset = self.load_dataset()

    # Create batched dataset
    if num_batches is not None:
      dataset = dataset.batch(batch_size)[:num_batches]
    else:
      dataset = dataset.batch(batch_size)

    correct = 0
    total = 0
    results = []
    debug_count = 0

    # Evaluate batch by batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
      prompts = batch["prompt"]

      questions = batch["question"]
      answers = batch["answer"]

      responses_collection = [[] for _ in range(len(prompts))]
      for pass_idx in range(num_passes):
        batch_response = self.generate(
            prompts=prompts,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=pass_idx
            if self.sampler_type != "vllm"
            else None,  # vllm handles seeding differently
        )
        for i, r in enumerate(batch_response):
          responses_collection[i].append(r)

      for prompt, question, answer, responses in zip(
          prompts, questions, answers, responses_collection
      ):
        is_correct = False
        extracted_answers = []
        answer_correct = []
        for response in responses:
          # Grade answer using both methods from utils.py
          if "aime_2024" in self.dataset:
            is_correct = evaluate_correctness(response, answer)
          else:
            model_answer = extract_answer_robust(response)
            extracted_answers.append(model_answer)

            if model_answer is None:
              continue
            # Grade answer using both methods from utils.py
            is_correct = math_utils.grade_answer_mathd(
                model_answer, answer
            ) or math_utils.grade_answer_sympy(model_answer, answer)

          answer_correct.append(is_correct)

          if is_correct:
            break

        if is_correct:
          correct += 1

        should_debug = debug_count < debug_first_n

        if should_debug:
          print(f"\n{'='*60}")
          print(f"DEBUG Example {debug_count + 1}/{debug_first_n}")
          print(f"Question: {question[:]}")
          print("=" * 60 + "\n")
          print(f"Ground truth: {answer}")
          print("=" * 60 + "\n")
          print(f"Prompt (first 300 chars): {prompt[:]}")
          if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            print(f"Prompt length: {len(self.tokenizer.encode(prompt))} tokens")
          print("=" * 60 + "\n")
          for i, (response, ans, cor) in enumerate(
              zip(responses, extracted_answers, answer_correct)
          ):
            print(f"Response {i}: {response}")
            print("=" * 120 + "\n")
            print(f"\nExtracted answer{i}: {ans}")
            print(f"Is correct: {cor}")
          print(f"Final result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
          print(
              f"Running accuracy: {correct}/{total+1} ="
              f" {(correct/(total+1)*100):.2f}%"
          )
          debug_count += 1

        total += 1

        # Store result
        results.append({
            "question": question,
            "answer": answer,
            "responses": responses,
            "extracted_answers": extracted_answers,
            "correct": is_correct,
        })

        # Print progress
        if total % 10 == 0:
          current_acc = (correct / total * 100) if total > 0 else 0
          print(f"\nProgress: {correct}/{total} = {current_acc:.2f}%")

    # Calculate final metrics
    accuracy = (correct / total * 100) if total > 0 else 0

    eval_results = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_passes": num_passes,
        "detailed_results": results,
    }

    return eval_results
# %%

if NOTEBOOK_ENV == "g3":
    DATA_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
    MODEL_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
else:
    DATA_PATH_PREFIX = "gs://tunix/data"
    MODEL_PATH_PREFIX = "gs://tunix/models"

MATH_500_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "MATH-500/test.jsonl")
AIME_2024_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "HuggingFaceH4/aime_2024/train-00000-of-00001.parquet")

MODEL_MAPPING = {
    "Qwen/Qwen2.5-1.5B-Instruct": (
        qwen2_lib.ModelConfig.qwen2p5_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "qwen2_5/torch/1.5b-it"),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B"),
    ),
    "agentica-org/DeepScaleR-1.5B-Preview": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepScaleR-1.5B-Preview"),
    ),
}

mesh_config = [[1, 2], ["fsdp", "tp"]]  # 2-way tensor parallelism
# %%
# MATH-500
num_batches_env = os.environ.get("NUM_BATCHES")
num_batches = int(num_batches_env) if num_batches_env and int(num_batches_env) > 0 else None

# model_version = "Qwen/Qwen2.5-1.5B-Instruct"
model_version = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset = MATH_500_DATA_PATH
model_config, model_path = MODEL_MAPPING[model_version]

evaluator = Qwen25MathEvaluator(
    model_config=model_config,
    model_version=model_version,
    model_path=model_path,
    dataset=dataset,
    mesh_config=mesh_config,
    max_prompt_length=1024,  # Increased
    max_generation_steps=1024,  # Increased
)

evaluator.load_model()

print("\nStarting evaluation...")
results = evaluator.evaluate(
    batch_size=8,
    num_batches=num_batches,
    temperature=0.6,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    debug_first_n=5,
)

# Print results
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Dataset: {dataset}")
print(f"Correct: {results['correct']}/{results['total']}")
print(f"Accuracy: {results['accuracy']:.2f}%")
print("=" * 60)
# %%
# AIME-2024
model_version = "agentica-org/DeepScaleR-1.5B-Preview"
# model_version = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset = AIME_2024_DATA_PATH
model_config, model_path = MODEL_MAPPING[model_version]


evaluator = Qwen25MathEvaluator(
    model_config=model_config,
    model_version=model_version,
    model_path=model_path,
    dataset=dataset,
    mesh_config=mesh_config,
    max_prompt_length=2048,  # Increased
    max_generation_steps=32768,  # Increased
)

evaluator.load_model()

print("\nStarting evaluation...")

results = evaluator.evaluate(
    batch_size=1,
    num_batches=num_batches,
    temperature=0.6,
    top_k=None,
    top_p=0.95,
    num_passes=1,
    debug_first_n=5,
)

# Print results
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Dataset: {dataset}")
print(f"Correct: {results['correct']}/{results['total']}")
print(f"Accuracy: {results['accuracy']:.2f}%")
print("=" * 60)
# %%
