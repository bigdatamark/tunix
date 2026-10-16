# [MODULE DOC]
# HELPER: dataset utilities for the Tunix CLI. Provides a structural protocol for tokenizers that
# support chat templating, chat-template application to records, a safe parser for literal
# call-argument strings, dynamic loading of a dataset factory from a module path or .py file
# (with optional call arguments in the specifier), and post-processing (prompt-key normalization,
# length filtering, truncation, train/validation split, epoch repetition and batching) of grain
# datasets. Module aliases: Tokenizer and TokenizerAdapter from the tokenizer adapter module.
# No side effects at import.

"""Utilities for handling and loading datasets in tunix CLI."""

import ast
import functools
import importlib
import os
from typing import Any, Callable, Optional, Protocol, Union

from tunix.generate import tokenizer_adapter
from tunix.utils import token_sanitization

Tokenizer = tokenizer_adapter.Tokenizer
TokenizerAdapter = tokenizer_adapter.TokenizerAdapter


class ChatTemplateTokenizer(Protocol):
  # [CLASS DOC]
  # HELPER: typing Protocol describing any tokenizer with an apply_chat_template method taking a
  # list of role/content message dicts, a tokenize flag (default False) and an
  # add_generation_prompt flag (default True), returning a string or list of token ids.
  # Used only for static typing; it has no runtime behavior.

  def apply_chat_template(
      self,
      messages: list[dict[str, str]],
      tokenize: bool = False,
      add_generation_prompt: bool = True,
  ) -> str | list[int]:
    # [FUNCTION DOC]
    # HELPER: protocol method stub (body is an ellipsis); declares the chat-template signature only.
    ...


def apply_chat_template(
    x, tokenizer: ChatTemplateTokenizer
) -> dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: applies the tokenizer's chat template to a record. Input: record x containing a
  # "prompt" field (list of chat messages) and a tokenizer. Output: new dict with "prompts" set to
  # the untokenized templated string with a generation prompt appended, plus every other key of x
  # except "prompt" (the "prompt" key is dropped). Raises KeyError if "prompt" is missing.
  # No side effects.
  return {
      "prompts": tokenizer.apply_chat_template(
          x["prompt"], tokenize=False, add_generation_prompt=True
      ),
      **{k: v for k, v in x.items() if k != "prompt"},
  }


def parse_call_string(arg_string: str) -> tuple[list[Any], dict[str, Any]]:
  # [FUNCTION DOC]
  # Parses a literal argument-list string. Input: e.g. text of comma-separated positional literals
  # and name=literal keyword pairs. Logic: blank/whitespace input returns an empty list and dict.
  # Otherwise wraps the text as a call to a dummy function, parses it as Python source into a syntax
  # tree (SyntaxError becomes ValueError), checks the first statement is an expression whose value
  # is a call (else ValueError "Internal error"), and literal-evaluates each positional argument and
  # each keyword value in order. Output: tuple of (positional list, keyword dict).
  # Exceptions: ValueError for bad syntax or non-literal values (literal evaluation raises
  # ValueError); no code is executed. No side effects.
  # KNOWN ISSUE: a starred-kwargs entry yields a keyword with a None name, which ends up as a None
  # key in the dict rather than being rejected.
  """Parses a string containing function call arguments and keyword arguments.

  Args:
    arg_string: A string representing the arguments of a function call,
      e.g., "'arg1', 123, kwarg1='value', kwarg2=456".

  Returns:
    A tuple containing two elements:
      - A list of positional arguments.
      - A dictionary of keyword arguments.

  Raises:
    ValueError: If the arg_string is not a valid argument syntax.
  """
  if not arg_string.strip():
    return [], {}

  fake_expression = f"dummy_func({arg_string})"
  try:
    tree = ast.parse(fake_expression)
  except SyntaxError as exc:
    raise ValueError(f"Invalid argument syntax: {arg_string}") from exc

  if not tree.body or not isinstance(tree.body[0], ast.Expr):
    raise ValueError(
        f"Internal error: Expected an expression node for '{arg_string}'"
    )

  call_node = tree.body[0].value
  if not isinstance(call_node, ast.Call):
    raise ValueError(f"Internal error: Expected a Call node for '{arg_string}'")

  parsed_args = []
  for node in call_node.args:
    parsed_args.append(ast.literal_eval(node))

  parsed_kwargs = {}
  for keyword in call_node.keywords:
    parsed_kwargs[keyword.arg] = ast.literal_eval(keyword.value)

  return parsed_args, parsed_kwargs


def get_dataset_from_module(
    specifier: str,
    tokenizer: ChatTemplateTokenizer | None = None,
    apply_chat_template_to_dataset: bool = False,
    **dataset_kwargs,
):
  # [FUNCTION DOC]
  # Loads a dataset by calling a factory function located by a specifier string.
  # Inputs: specifier "module.path[:func[(args)]]" or "/path/file.py[:func[(args)]]"; optional
  # tokenizer; apply_chat_template_to_dataset flag; extra keyword args forwarded to the factory.
  # Logic: if the specifier contains both "(" and ":", splits at the last "(" into specifier and
  # argument text. Then, if a ":" remains, splits at the last ":" into module part and function
  # name. If the module part is an existing path ending in ".py", loads it as a module from file
  # (module name = file base name) and executes it, wrapping execution failures in ImportError;
  # otherwise imports it by dotted name, wrapping any failure in ImportError. If a function name was
  # given, looks it up on the module (AttributeError if missing) and, when argument text exists,
  # strips trailing ")" characters and parses it as literal args/kwargs; with no function name uses
  # the module's create_dataset. Calls the function with the positional args and the merge of the
  # forwarded kwargs and parsed kwargs (parsed take precedence). If chat templating is not
  # requested returns the dataset; otherwise requires a tokenizer (ValueError if None) and returns
  # the dataset mapped with the chat-template helper.
  # Output: the dataset object returned by the factory (possibly mapped).
  # Exceptions: ImportError, ValueError, AttributeError, factory errors.
  # Side effects: imports/executes arbitrary module code.
  # KNOWN ISSUE: the None checks for spec/module happen after module_from_spec is called with the
  # spec, so a None spec would fail earlier with a different error; argument parsing ignores
  # arguments when no ":" function name is present (they would not be split off since "(" splitting
  # requires ":"); stripping all trailing ")" can remove parentheses belonging to the last argument.
  """Get dataset from module.

  Examples of specifier:
    - "data.coding" # create_dataset is the default function
    - "data.coding:create_dataset"
    - "data.coding:get_my_dataset"
    - "data.coding:create_dataset(name='coding_v0')"
    - "data.coding:create_dataset('coding_v0', split='train')"
    - "/home/user/project/data/coding.py:get_dataset"

  Args:
    specifier: The specifier of the module.
    tokenizer: The tokenizer to apply to the dataset when the module returns
      raw chat messages under ``prompt``. It must expose
      ``apply_chat_template``.
    apply_chat_template_to_dataset: Whether to apply chat templating when the
      module dataset exposes a raw ``prompt`` field.
    **dataset_kwargs: Keyword arguments forwarded to ``create_dataset``.

  Returns:
    The dataset.
  Raises:
    ImportError: If the module cannot be imported or loaded.
  """
  if "(" in specifier and ":" in specifier:
    specifier, args_part = specifier.rsplit("(", 1)
  else:
    args_part = ""
  if ":" in specifier:
    specifier, func_spec = specifier.rsplit(":", 1)
  else:
    func_spec = ""
  if os.path.exists(specifier) and specifier.endswith(".py"):
    module_name = os.path.splitext(os.path.basename(specifier))[0]
    spec = importlib.util.spec_from_file_location(module_name, specifier)
    module = importlib.util.module_from_spec(spec)

    if spec is None:
      raise ImportError(f"Failed to create spec for {specifier}")
    if spec.loader is None:
      raise ImportError(f"Failed to get loader for spec {specifier}")
    if module is None:
      raise ImportError(f"Failed to create module for {specifier}")

    try:
      spec.loader.exec_module(module)
    except Exception as e:
      raise ImportError(
          f"Failed to execute module {module_name} from {specifier}: {e}"
      ) from e
  else:
    try:
      module = importlib.import_module(specifier)
    except Exception as e:
      raise ImportError(f"Failed to import module {specifier}: {e}") from e
  args = []
  kwargs = {}
  if func_spec:
    func = getattr(module, func_spec)
    if args_part:
      args_part = args_part.rstrip(")")
      args, kwargs = parse_call_string(args_part)

  else:
    func = module.create_dataset
  dataset = func(*args, **{**dataset_kwargs, **kwargs})
  if not apply_chat_template_to_dataset:
    return dataset

  if tokenizer is None:
    raise ValueError(
        "tokenizer must be provided when apply_chat_template_to_dataset is"
        " True."
    )

  return dataset.map(
      functools.partial(apply_chat_template, tokenizer=tokenizer)
  )


def post_init_dataset(
    dataset,
    tokenizer: TokenizerAdapter,
    batch_size: int,
    num_batches: Optional[int],
    max_prompt_length: Optional[int],
    fraction: float = 1.0,
    num_epochs: int = 1,
    prompt_key: str = "prompts",
    custom_batch_fn: Optional[Callable] = None,
):
  # [FUNCTION DOC]
  # Post-processes a grain map dataset into batched train and optional validation iter datasets.
  # Inputs: dataset (grain MapDataset supporting len, map, filter, slicing, repeat); tokenizer with
  # encode; batch_size; num_batches (optional cap on number of examples as num_batches * batch_size);
  # max_prompt_length (optional); fraction (train split fraction, default 1.0); num_epochs (default
  # 1); prompt_key (default "prompts"); custom_batch_fn (optional batching function).
  # Logic: records original length. If prompt_key differs from "prompts", maps each record to add a
  # "prompts" field equal to the control-token-sanitized value of the source key and thereafter uses
  # "prompts". If max_prompt_length is positive, filters out records whose encoded prompt has more
  # tokens than the limit. Raises ValueError (with diagnostics) if nothing remains. If num_batches is
  # given, truncates to min(num_batches * batch_size, length). If 0 < fraction < 1, splits at
  # floor(length * fraction) into first and second segments; else the whole set is the first segment
  # and there is no second. Raises ValueError if the first segment is empty. Each segment is repeated
  # num_epochs times, converted to an iter dataset and batched with batch_size and custom_batch_fn.
  # Output: tuple (train_batches, validation_batches or None).
  # Exceptions: ValueError as above; tokenizer errors. Side effects: none beyond dataset pipeline creation.
  # KNOWN ISSUE: the length after filtering is computed with len on a filtered map dataset, which
  # in grain still reports the pre-filter length, so the emptiness check may not detect an
  # all-filtered dataset; also the docstring omits prompt_key, custom_batch_fn and the tuple return.
  """Applies post-initialization transformations to a dataset.

  This function filters, batches, and optionally limits the number of batches
  in a dataset.

  Args:
    dataset: The input dataset.
    tokenizer: The tokenizer used for prompt length filtering.
    batch_size: The size of each batch.
    num_batches: If not None, the maximum number of batches to yield.
    max_prompt_length: If not None and greater than 0, prompts longer than this
      will be filtered out.
    fraction: Fraction of the dataset to use (between 0.0 and 1.0), commonly
      used for splitting training and validation sets.
    num_epochs: Number of times to repeat the dataset.
  Returns:
    The processed dataset.
  """
  original_size = len(dataset)

  if prompt_key != "prompts":
    source_prompt_key = prompt_key

    def normalize_prompt_key(x):
      # [FUNCTION DOC]
      # HELPER: returns a copy of the record with "prompts" set to the sanitized (control tokens
      # neutralized) text of the configured source prompt key. KeyError if that key is missing.
      return {
          **x,
          "prompts": token_sanitization.sanitize_control_tokens(
              x[source_prompt_key]
          ),
      }

    dataset = dataset.map(normalize_prompt_key)
    prompt_key = "prompts"

  if max_prompt_length is not None and max_prompt_length > 0:

    def prompt_length_filter(x):
      # [FUNCTION DOC]
      # HELPER: filter predicate; true when the tokenizer encoding of the record's prompt has at most
      # max_prompt_length tokens.
      tokens = tokenizer.encode(x[prompt_key])
      return len(tokens) <= max_prompt_length

    dataset = dataset.filter(prompt_length_filter)

  filtered_size = len(dataset)
  if filtered_size == 0:
    raise ValueError(
        "Training dataset is empty after post_init_dataset filtering. "
        f"original_size={original_size}, max_prompt_length={max_prompt_length}, "
        f"prompt_key={prompt_key!r}. Consider increasing max_prompt_length "
        "or adjusting the prompt template/filter settings."
    )

  if num_batches is not None:
    target_size = min(num_batches * batch_size, len(dataset))
    dataset = dataset[:target_size]

  if fraction < 1.0 and fraction > 0.0:
    first_segment_size = int(len(dataset) * fraction)
    first_segment_dataset = dataset[:first_segment_size]
    second_segment_dataset = dataset[first_segment_size:]
  else:
    first_segment_dataset = dataset
    second_segment_dataset = None

  if len(first_segment_dataset) == 0:
    raise ValueError(
        "Training dataset is empty after post_init_dataset split. "
        f"filtered_size={filtered_size}, fraction={fraction}, "
        f"num_batches={num_batches}, batch_size={batch_size}."
    )

  first_segment_dataset = (
      first_segment_dataset.repeat(num_epochs)
      .to_iter_dataset()
      .batch(batch_size, batch_fn=custom_batch_fn)
  )
  if second_segment_dataset is not None:
    second_segment_dataset = (
        second_segment_dataset.repeat(num_epochs)
        .to_iter_dataset()
        .batch(batch_size, batch_fn=custom_batch_fn)
    )

  return first_segment_dataset, second_segment_dataset
