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
# Configuration loading and CLI interface for Tunix command-line pipelines (PEFT and GRPO mains).
# Loads a base YAML config with OmegaConf, applies an optional override YAML file and key=value
# CLI overrides (and T_-prefixed environment variable overrides), type-checks/coerces overridden
# values against the base config, propagates model_config fields into actor/reference/rollout
# model configs, validates tokenizer, model source, supported workflow and perf-metrics settings,
# and offers builders for optimizers (optax, with optional learning-rate schedules), mesh specs,
# training-config dictionaries and reward functions loaded from modules or files.
# Module state: environment-variable prefix "T_"; tuple of allowed model sources (kaggle,
# huggingface, gcs, internal, maxtext, empty string); a mapping from YAML value types to parser
# callables (str, int, float, boolean-string parser, OmegaConf dict config to dict, list config to list).
# Side effects at import: none (loading a .env file happens on construction).

"""Config and CLI launched interface."""
import ast
import collections
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
import copy
import importlib
import inspect
import os
import pathlib
import shutil
import stat
from typing import Any, Dict, Iterator, Sequence

from absl import logging
import dotenv
import jax
import omegaconf
import optax
import orbax.checkpoint as ocp
from tunix.perf import metrics as perf_metrics
from tunix.sft import metrics_logger
from tunix.sft import profiler
from tunix.utils import mesh as mesh_lib

# Define a prefix for environment variables that can override YAML keys
_TUNIX_PREFIX = "T_"
_SUPPORTED_MODEL_SOURCES = (
    "kaggle",
    "huggingface",
    "gcs",
    "internal",
    "maxtext",
    "",
)


def yaml_key_to_env_key(s: str) -> str:
  # [FUNCTION DOC]
  # HELPER: maps a YAML key to its overriding environment variable name: "T_" plus the key uppercased.
  return _TUNIX_PREFIX + s.upper()


def string_to_bool(s: str) -> bool:
  # [FUNCTION DOC]
  # HELPER: parses "true"/"false" case-insensitively to a boolean; any other string raises ValueError.
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")


_yaml_types_to_parser = {
    str: str,
    int: int,
    float: float,
    bool: string_to_bool,
    omegaconf.dictconfig.DictConfig: dict,
    omegaconf.listconfig.ListConfig: list,
}


def _normalize_cli_override(schema_value: Any, override_value: Any) -> Any:
  # [FUNCTION DOC]
  # HELPER: normalizes a CLI override against its schema value. If the override is None and the
  # schema value is a string, returns an empty string (OmegaConf parses key="" as None). If both
  # schema and override are mappings (dicts or OmegaConf dict configs), returns a new plain dict
  # whose entries are recursively normalized against the schema's same-named entries (missing
  # schema entries treated as None). Otherwise returns the override unchanged.
  """Restores empty string overrides that OmegaConf parses as None.

  OmegaConf.from_cli interprets CLI values like key="" as None. For string
  fields we want to preserve the user's intent and treat that as an empty
  string, including for nested dictionary overrides.

  Args:
    schema_value: Pre-existing schema value for reference.
    override_value: Proposed override value.

  Returns:
    The normalized override value.
  """
  if override_value is None and isinstance(schema_value, str):
    return ""
  if isinstance(
      schema_value, (collections.abc.Mapping, omegaconf.DictConfig)
  ) and isinstance(
      override_value, (collections.abc.Mapping, omegaconf.DictConfig)
  ):
    normalized = {}
    for key, value in override_value.items():
      normalized[key] = _normalize_cli_override(schema_value.get(key), value)
    return normalized
  return override_value


def _can_override_nullable_schema(override_value: Any) -> bool:
  # [FUNCTION DOC]
  # HELPER: returns True if an override for a key whose base value is null has an acceptable type:
  # string, int, float, bool, mapping, list, or OmegaConf dict/list config.
  """Returns whether a null-default schema key can accept the override.

  Nullable schema fields do not provide enough type information to route
  through `_yaml_types_to_parser`. In that case, preserve the CLI-parsed value
  directly, or the raw string from the environment.

  Args:
    override_value: Proposed override value.

  Returns:
    True if the override value is compatible.
  """
  return isinstance(
      override_value,
      (
          str,
          int,
          float,
          bool,
          collections.abc.Mapping,
          list,
          omegaconf.dictconfig.DictConfig,
          omegaconf.listconfig.ListConfig,
      ),
  )


def get_project_root() -> pathlib.Path:
  # [FUNCTION DOC]
  # Finds the project root: starting at this file's directory and walking up through all parents,
  # returns the first directory containing a file or directory named "LICENSE"; if none, returns the
  # current working directory. Output: a path. No side effects.
  # KNOWN ISSUE: the docstring mentions ".git", "pyproject.toml" and "setup.py" markers, but only
  # "LICENSE" is checked.
  """Returns the project root folder.

  It searches up from the current file until it finds a marker file like '.git',
  'pyproject.toml', or 'setup.py'.
  """
  current_path = pathlib.Path(__file__).resolve().parent
  # List of files that define the root of your project
  root_markers = [
      "LICENSE",
  ]

  # Iterate up through parent directories
  for parent in [current_path] + list(current_path.parents):
    # Check if any marker exists in this parent directory
    if any((parent / marker).exists() for marker in root_markers):
      return parent

  # Fallback: if no marker is found, return the current working directory
  return pathlib.Path.cwd()


def _dict_to_cli_args(
    d: collections.abc.Mapping[str, Any], parent_key: str = "", sep: str = "."
) -> Iterator[str]:
  # [FUNCTION DOC]
  # HELPER: generator flattening a nested mapping into "dotted.key=value" CLI strings. Keys are
  # joined with the separator (default "."); non-empty nested mappings recurse; empty mappings
  # yield "key={}"; None yields "key=null"; other values are formatted with their string form.
  # KNOWN ISSUE: values are not quoted or escaped, so strings containing spaces, "=" or OmegaConf
  # syntax may be re-parsed differently by the CLI parser.
  """Converts a dictionary to an Iterator string of CLI arguments."""
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, (collections.abc.Mapping, omegaconf.DictConfig)):
      if v:
        yield from _dict_to_cli_args(v, parent_key=new_key, sep=sep)
      else:
        yield f"{new_key}={{}}"
    else:
      if v is None:
        yield f"{new_key}=null"
      else:
        yield f"{new_key}={v}"


class HyperParameters:
  # [CLASS DOC]
  # Pipeline configuration holder and builder. Public attribute config: an ordered dict of the final
  # merged top-level config keys (values may be plain values, dicts or OmegaConf containers).
  # replace_keys: set of keys (lora_config, training_config, optimizer_config, profiler_options,
  # rl_training_config) whose dict overrides replace rather than deep-merge the base value.
  # Precedence: base YAML < override_config_file YAML < CLI key=value; environment variables
  # T_<KEY> may override top-level keys but conflicting CLI+env for the same key is an error.
  # Subclassed by the PEFT and GRPO pipelines.
  """Loads, merges, overrides, validates, and prepares the configuration for pipeline execution.

  Configurations are merged from multiple sources. The following order of
  precedence applies, with later sources overriding earlier ones:
  1. Base Config File: The first positional argument, path to a YAML config
  file.
  2. Config File Override: An optional `override_config_file=/path/to/file.yaml`
     argument. Values in this file override values in the base config file.
  3. CLI Arguments: `key=value` pairs provided as arguments override values
     from both the base config and the config file override.

  Environment variables prefixed with `T_` can also be used to set parameters,
  but it is an error to set a parameter via both an environment variable and
  a command-line argument or override file.
  """

  def __init__(self, argv: list[str], **kwargs):
    # [FUNCTION DOC]
    # Constructor: loads, merges and validates the config.
    # Inputs: argv - argv[0] the entry-point name, argv[1] the base YAML path, remaining items either
    # "override_config_file=<path>" (at most one) or key=value overrides; kwargs - passed through to
    # the override routine (unused there).
    # Logic: loads a .env file into the environment. Raises ValueError if there is no base path or
    # argv[1] contains "=". The special value "base_config.yaml" resolves next to this module. Loads the
    # base YAML (ValueError if missing), validates T_ environment variables against its keys, and
    # remembers the base model_config. Collects the override file (ValueError on a second one) and CLI
    # overrides; the override file is flattened to CLI strings and placed before the CLI overrides so
    # CLI wins. Applies overrides into the ordered dict and logs the updated keys. Then, for each of
    # actor_model_config, reference_model_config and rollout_model_config present, copies every key
    # from the final model_config whose value is missing there or equals the base YAML model_config
    # value (i.e. not explicitly customized). Validates tokenizer, model source, model/source workflow
    # and perf metrics (using argv[0]).
    # Output: none. Exceptions: ValueError/KeyError/TypeError from validation. Side effects: environment
    # mutation via dotenv, file reads, logging.
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.

    dotenv.load_dotenv()
    raw_keys = collections.OrderedDict()

    if len(argv) < 2 or "=" in argv[1]:
      raise ValueError(
          "The first argument must be a path to a base config file."
      )

    # Handle relative paths used in example scripts as a special case.
    # TODO(noghabi): Remove this once the example scripts are updated.
    if argv[1] == "base_config.yaml":
      base_config_file = pathlib.Path(__file__).parent / argv[1]
    else:
      base_config_file = argv[1]
    raw_data_from_yaml = self._load_config_from_yaml(base_config_file)
    self._validate_env_variable(raw_data_from_yaml)
    base_model_config = raw_data_from_yaml.get("model_config", {})

    config_file_override = None
    cli_overrides = []
    for arg in argv[2:]:
      if arg.startswith("override_config_file="):
        if config_file_override is not None:
          raise ValueError("Only one override_config_file argument is allowed.")
        _, config_file_override = arg.split("=", 1)
      else:
        cli_overrides.append(arg)

    self.replace_keys = {
        "lora_config",
        "training_config",
        "optimizer_config",
        "profiler_options",
        "rl_training_config",
    }

    all_overrides = []
    if config_file_override:
      next_conf = self._load_config_from_yaml(config_file_override)
      all_overrides.extend(_dict_to_cli_args(next_conf))

    # First we apply the file overrides, then the overrides from the command
    # line making the command line the highest priority.
    all_overrides.extend(cli_overrides)

    keys_from_env_and_command_line = self._update_from_env_and_command_line(
        raw_keys, raw_data_from_yaml, all_overrides, **kwargs
    )
    logging.info(
        "Updating keys from env and command line: %s",
        keys_from_env_and_command_line,
    )
    self.config = raw_keys
    # Inherit missing keys from model_config to actor_model_config, etc.
    # Also update keys that were not explicitly overridden
    current_model_config = self.config.get("model_config", {})
    for config_key in [
        "actor_model_config",
        "reference_model_config",
        "rollout_model_config",
    ]:
      if config_key in self.config:
        for k, v in current_model_config.items():
          if k not in self.config[config_key] or self.config[config_key][
              k
          ] == base_model_config.get(k):
            self.config[config_key][k] = v
    self._validate_tokenizer()
    self._validate_model_source(raw_keys)
    self.check_supported_workflow()
    self._validate_perf_metrics(entry_point=argv[0])

  def _config_mapping(self, key: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns config[key] as a plain dict copy; {} if missing or None; TypeError if the value
    # is not a mapping.
    """Returns a config section as a plain dictionary.

    This narrows nested config sections that may otherwise be inferred as broad
    unions of scalars and mappings by static type checkers.

    Args:
      key: Key of config section.

    Returns:
      The mapped dictionary config section.
    """
    value = self.config.get(key)
    if value is None:
      return {}
    if not isinstance(value, Mapping):
      raise TypeError(
          f"Expected config section {key!r} to be a mapping, got"
          f" {type(value).__name__}."
      )
    return dict(value)

  def _mutable_config_mapping(self, key: str) -> MutableMapping[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns config[key] as the live mutable mapping for in-place edits; if missing or None,
    # stores and returns a new empty dict; TypeError if it is not a mutable mapping.
    """Returns a mutable config section for in-place updates.

    Args:
      key: Key of config section.

    Returns:
      The mutable mapping of the config section.
    """
    value = self.config.get(key)
    if value is None:
      section: dict[str, Any] = {}
      self.config[key] = section
      return section
    if not isinstance(value, MutableMapping):
      raise TypeError(
          f"Expected config section {key!r} to be a mutable mapping, got"
          f" {type(value).__name__}."
      )
    return value

  def _config_string(self, key: str, default: str = "") -> str:
    # [FUNCTION DOC]
    # HELPER: returns config[key] as a string, using default when missing or None; TypeError when the
    # value is not a string.
    """Returns a string config value with validation.

    Args:
      key: Key of config value.
      default: Default fallback value if not set.

    Returns:
      The string config value.
    """
    value = self.config.get(key, default)
    if value is None:
      return default
    if not isinstance(value, str):
      raise TypeError(
          f"Expected config value {key!r} to be a string, got"
          f" {type(value).__name__}."
      )
    return value

  def _config_bool(self, key: str, default: bool = False) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns config[key] as a bool, using default when missing or None; TypeError when the
    # value is not a bool.
    """Returns a boolean config value with validation.

    Args:
      key: Key of config value.
      default: Default fallback value if not set.

    Returns:
      The boolean config value.
    """
    value = self.config.get(key, default)
    if value is None:
      return default
    if not isinstance(value, bool):
      raise TypeError(
          f"Expected config value {key!r} to be a bool, got"
          f" {type(value).__name__}."
      )
    return value

  def _validate_perf_metrics(self, entry_point: str):
    # [FUNCTION DOC]
    # HELPER: validates perf metrics settings. Reads training_config.perf_metrics_options (missing
    # treated as empty). If non-empty: ValueError unless entry_point ends with "grpo_main"; if
    # custom_export_fn_path or custom_export_fn_path_v2 is set and cannot be imported via the
    # dotted-path loader, raises ValueError. Side effects: imports the referenced modules.
    """Validates that perf metrics are only enabled for GRPO.

    Args:
      entry_point: The entry point of the pipeline.

    Raises:
      ValueError: If perf metrics are enabled but the entry point is not
        "grpo_main".
    """
    perf_config = self.config.get("training_config", {}).get(
        "perf_metrics_options", {}
    )

    if perf_config:
      if not entry_point.endswith("grpo_main"):
        raise ValueError(
            "Perf metrics are currently only supported for GRPO training"
            " (grpo_main)."
        )
      custom_export_fn_path = perf_config.get("custom_export_fn_path")
      if (
          custom_export_fn_path
          and self._get_function_from_path(custom_export_fn_path) is None
      ):
        raise ValueError(
            "Could not load custom export function from"
            f" {custom_export_fn_path}"
        )
      custom_export_fn_path_v2 = perf_config.get("custom_export_fn_path_v2")
      if (
          custom_export_fn_path_v2
          and self._get_function_from_path(custom_export_fn_path_v2) is None
      ):
        raise ValueError(
            "Could not load custom export function v2 from"
            f" {custom_export_fn_path_v2}"
        )

  def _validate_tokenizer(self):
    # [FUNCTION DOC]
    # HELPER: validates tokenizer_config: tokenizer_type must be "sentencepiece" or "huggingface"
    # (ValueError otherwise). For huggingface: if the path does not start with "/" or "gs://" and
    # HF_TOKEN is absent from the environment, ValueError; then an empty path raises ValueError.
    # KeyError if tokenizer_config or its type/path keys are missing.
    # KNOWN ISSUE: an empty huggingface tokenizer path reports the missing HF_TOKEN error first
    # when the token is unset, masking the more relevant empty-path error.
    """Validate the tokenizer configuration.

    Currently only sentencepiece and huggingface are supported. `HF_TOKEN` must
    be set if huggingface tokenizer is used.
    """
    tokenizer_config = self.config["tokenizer_config"]
    tokenizer_type = tokenizer_config["tokenizer_type"]
    tokenizer_path = tokenizer_config["tokenizer_path"]
    valid_tokenizer_type = {"sentencepiece", "huggingface"}
    if tokenizer_type not in valid_tokenizer_type:
      raise ValueError(
          f"tokenizer_type {tokenizer_type} is not supported, currently only"
          f" {valid_tokenizer_type} is supported"
      )
    if tokenizer_type == "huggingface":
      # Only require HF_TOKEN when loading from HuggingFace Hub (not a local
      # or CNS path).
      is_local_path = tokenizer_path.startswith(("/", "gs://"))
      if not is_local_path and "HF_TOKEN" not in os.environ:
        raise ValueError("Missing `HF_TOKEN` to access hf tokenizer")
      if not tokenizer_path:
        raise ValueError("tokenizer_path must be specified.")

  def clear_directory_contents(self):
    # [FUNCTION DOC]
    # Deletes everything inside the model download directory. Uses top-level model_download_path, or
    # model_config.model_download_path if the former is falsy. If it is not an existing directory,
    # logs an error and returns. Otherwise for each entry: files/links are chmod-ed to owner
    # read/write/execute (errors ignored) and removed; directories are removed recursively; failures
    # are logged as warnings with traceback and skipped. Logs progress. Output: none.
    # Side effects: filesystem deletion.
    # KNOWN ISSUE: if both paths are missing (None), the directory check raises TypeError instead of
    # logging; the docstring says "the specified directory" but the directory comes from config.
    """Removes all files, directories, and links within the specified directory."""
    model_download_path = self.config.get("model_download_path")
    if not model_download_path:
      model_download_path = self.config["model_config"].get(
          "model_download_path"
      )
    if not os.path.isdir(model_download_path):
      logging.error(
          "Error: '%r' is not a valid directory.", model_download_path
      )
      return

    logging.info("Clearing contents of '%s'...", model_download_path)
    for item in os.listdir(model_download_path):
      item_path = os.path.join(model_download_path, item)
      try:
        if os.path.isfile(item_path) or os.path.islink(item_path):
          # Attempt to make the file writable before removing,
          # in case of permission issues.
          try:
            os.chmod(item_path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
          except OSError:
            pass  # Continue and let os.remove() raise the error if it fails
          os.remove(item_path)
          logging.info("  Removed file/link: %s", item_path)
        elif os.path.isdir(item_path):
          # shutil.rmtree can also handle permission issues internally
          # by providing an onerror handler if needed.
          shutil.rmtree(item_path)
          logging.info("  Removed directory: %s", item_path)
      except (OSError, shutil.Error):
        logging.warning("  Failed to delete %s.", item_path, exc_info=True)
    logging.info("Finished clearing '%r'.", model_download_path)

  def _validate_model_source(self, raw_keys: collections.OrderedDict[str, Any]):
    # [FUNCTION DOC]
    # HELPER: validates model_config.model_source is one of the supported sources (ValueError
    # otherwise), and that kaggle/huggingface sources have a truthy intermediate_ckpt_dir (ValueError
    # otherwise). KeyError if model_config is missing.
    """Validate the checkpoint source and intermediate checkpoint."""
    model_config = raw_keys["model_config"]
    model_source = model_config.get("model_source")
    intermediate_ckpt = model_config.get("intermediate_ckpt_dir")

    if model_source not in _SUPPORTED_MODEL_SOURCES:
      raise ValueError(
          f"Invalid model_source: {model_source!r}. Must be one of"
          f" {_SUPPORTED_MODEL_SOURCES}."
      )

    if model_source in ["kaggle", "huggingface"] and not intermediate_ckpt:
      raise ValueError(
          "intermediate_ckpt must be specified when model_source is 'kaggle' or"
          " 'huggingface'"
      )

  def check_supported_workflow(self) -> None:
    # [FUNCTION DOC]
    # Validates that the model source suits the model name. Allowed sources: names starting with
    # "gemma3" or "gemma-3" need gcs/internal/maxtext; "gemma2"/"gemma-2" need kaggle/internal/maxtext;
    # other "gemma" names need kaggle/internal/maxtext; all other models need
    # huggingface/internal/maxtext. Raises ValueError naming the expected sources on mismatch;
    # KeyError if model_name/model_source are missing.
    """Checks if the model_source is supported for the given model_name.

    Raises:
      ValueError: If the model_source is not supported for the model_name.
    """
    model_config = self.config["model_config"]
    model_name = model_config["model_name"]
    model_source = model_config["model_source"]
    supported_sources = collections.defaultdict(
        lambda: ["huggingface", "internal", "maxtext"]
    )
    # TODO(b/467448875): Add support for other sources, such as kaggle for other
    # models.
    supported_sources["gemma"] = ["kaggle", "internal", "maxtext"]
    supported_sources["gemma2"] = ["kaggle", "internal", "maxtext"]
    supported_sources["gemma3"] = ["gcs", "internal", "maxtext"]

    if model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
      expected_sources = supported_sources["gemma3"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    elif model_name.startswith("gemma2") or model_name.startswith("gemma-2"):
      expected_sources = supported_sources["gemma2"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    elif model_name.startswith("gemma"):
      expected_sources = supported_sources["gemma"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    else:
      # Default case for other models
      expected_sources = supported_sources["other"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )

  def _get_nested_config(self, keys: Sequence[str]) -> Any:
    # [FUNCTION DOC]
    # HELPER: walks the config by a sequence of keys. TypeError if an intermediate value is not a
    # dict/DictConfig (message lists the path so far); KeyError (with the path) if a key is missing.
    # Output: the value found.
    """Helper to retrieve a value from a nested dictionary."""
    current_level = self.config
    for i, key in enumerate(keys):
      if not isinstance(current_level, omegaconf.dictconfig.DictConfig | dict):
        raise TypeError(
            f"Attempted to access key '{key}' on a non-dictionary element "
            f"at path: {' -> '.join(keys[:i])}"
        )
      try:
        current_level = current_level[key]
      except KeyError as exc:
        raise KeyError(
            f"Key '{key}' not found in config path: {' -> '.join(keys[:i+1])}"
        ) from exc
    return current_level

  def _extract_kwargs(
      self,
      func: Callable[..., Any],
      config: Dict[str, Any],
      config_path_info: str,
      learning_rate: Any | None = None,
  ) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: builds keyword args for a callable from a config mapping by inspecting its signature:
    # each parameter present in config takes the config value; otherwise, if a learning_rate value is
    # given and the parameter is named learning_rate, that value is used; otherwise a parameter with
    # no default raises ValueError naming the function and config path. Optional parameters absent
    # from config are omitted. Variadic parameters are treated like named ones (looked up by name).
    """Extracts and validates kwargs for a function from a config dictionary."""
    sig = inspect.signature(func)
    kwargs = {}
    for param in sig.parameters.values():
      param_name = param.name
      if param_name in config:
        kwargs[param_name] = config[param_name]
      elif learning_rate is not None and param_name == "learning_rate":
        kwargs[param_name] = learning_rate
      elif param.default is param.empty:
        # Safely get a name or representation for the callable
        func_name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"Missing required argument '{param_name}' for {func_name} "
            f"in config at {config_path_info}."
        )
    return kwargs

  def _get_schedule_fn(self, schedule_type: str, config_path_info: str):
    # [FUNCTION DOC]
    # HELPER: returns the optax.schedules attribute named schedule_type; re-raises AttributeError with
    # the config path if not found.
    """Dynamically imports a schedule function from optax.schedules.

    Args:
      schedule_type: The type of schedule (e.g., "constant_schedule",
        "warmup_cosine_decay_schedule").
      config_path_info: The path to the config file, used for error reporting.

    Returns:
      The corresponding function from optax.schedules.

    Raises:
      AttributeError: If the schedule type does not correspond to a valid
                      function in optax.schedules.
    """
    try:
      schedule_fn = getattr(optax.schedules, schedule_type)
      return schedule_fn
    except AttributeError as exc:
      raise AttributeError(
          f"Config {config_path_info}: '{schedule_type}' is not a valid"
          " function in optax.schedules."
      ) from exc

  def _create_learning_rate(
      self, optimizer_config: Dict[str, Any], config_path_info: str
  ) -> Any:
    # [FUNCTION DOC]
    # HELPER: builds the learning rate. If schedule_type is set, resolves the optax schedule, extracts
    # its kwargs from the optimizer config, logs and returns the schedule. Otherwise returns
    # optimizer_config.learning_rate, raising TypeError if it is set and not an int/float (may return None).
    # KNOWN ISSUE: the scalar learning rate is logged with an integer format specifier, so a float
    # like 1e-5 is logged as 0 and a None value produces a logging formatting error.
    """Creates a learning rate schedule based on the optimizer config."""
    schedule_type = optimizer_config.get("schedule_type")
    if schedule_type:
      schedule_func = self._get_schedule_fn(schedule_type, config_path_info)
      schedule_kwargs = self._extract_kwargs(
          schedule_func, optimizer_config, config_path_info
      )
      logging.info(
          "Creating learning rate with schedule_type: %s, and following"
          " kwargs: %s",
          schedule_type,
          schedule_kwargs,
      )
      return schedule_func(**schedule_kwargs)

    # Default: No schedule, learning_rate should be a scalar
    learning_rate = optimizer_config.get("learning_rate")
    if learning_rate is not None and not isinstance(
        learning_rate, (float, int)
    ):
      raise TypeError(
          "learning_rate must be a scalar when no schedule_type is specified, "
          f"got {type(learning_rate)} in config at {config_path_info}."
      )
    logging.info("Creating learning rate with learning_rate: %d", learning_rate)
    return learning_rate

  def create_optimizer(
      self, *optimizer_keys: str
  ) -> optax.GradientTransformation:
    # [FUNCTION DOC]
    # Creates an optax optimizer from a nested config section.
    # Inputs: one or more keys addressing the optimizer section (e.g. "optimizer_config" or
    # "rl_training_config", "actor_optimizer_config").
    # Logic: ValueError if no keys. Resolves the section (KeyError rewrapped); ValueError if it is not a
    # dict. Requires opt_type (ValueError otherwise) and looks up the lowercased name on optax. Builds
    # the learning rate (schedule or scalar); if it is None and the optimizer requires learning_rate,
    # raises ValueError. Extracts kwargs for the optimizer from the section (learning rate injected).
    # Wraps the optimizer with optax inject_hyperparams (float32 hyperparams) so the learning rate is
    # observable, and calls it; TypeError from the call is rewrapped with details.
    # Output: an optax GradientTransformation. Side effects: logging.
    # KNOWN ISSUE: the optimizer lookup catches ValueError, but a missing optax attribute raises
    # AttributeError, so an unknown opt_type surfaces as a raw AttributeError without the helpful message.
    """Creates the optimizer based on a config path.

    Args:
        *optimizer_keys: One or more strings representing the keys to navigate
          the nested self.config dictionary. For example, ('rl_training_config',
          'actor_optimizer_config') would access
          self.config['rl_training_config']['actor_optimizer_config'].

    Returns:
        An optimizer instance.

    Raises:
        ValueError: If no optimizer_keys are provided.
        KeyError: If a key in the path is not found.
        TypeError: If an intermediate element in the path is not a dictionary.
    """

    if not optimizer_keys:
      raise ValueError("At least one optimizer key must be provided.")
    config_path_info = " -> ".join(optimizer_keys)

    try:
      optimizer_config = self._get_nested_config(optimizer_keys)
    except KeyError as e:
      raise KeyError(f"Could not resolve optimizer config path: {e}") from e

    if not isinstance(optimizer_config, omegaconf.dictconfig.DictConfig | dict):
      raise ValueError("optimizer_config must be a dictionary")

    opt_type = optimizer_config.get("opt_type")
    if not opt_type:
      raise ValueError("Optimizer name is required")

    try:
      opt_func = getattr(optax, opt_type.lower())
    except ValueError as e:
      raise ValueError(
          f"Optimizer type '{opt_type}' not supported from {config_path_info}."
          " Available options, see"
          " https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers"
      ) from e

    # Handle learning rate, potentially creating a schedule
    learning_rate_val = self._create_learning_rate(
        optimizer_config, config_path_info
    )
    if learning_rate_val is None and (
        "learning_rate" in inspect.signature(opt_func).parameters
        and inspect.signature(opt_func).parameters["learning_rate"].default
        is inspect.Parameter.empty
    ):
      # learning_rate is required by opt_func but not provided and no schedule
      raise ValueError(
          "Missing required argument 'learning_rate' for optimizer"
          f" '{opt_type}' and no schedule defined in config at"
          f" {config_path_info}."
      )

    opt_kwargs = self._extract_kwargs(
        opt_func, optimizer_config, config_path_info, learning_rate_val
    )
    # Wrap the optimizer function with inject_hyperparams so that
    # the learning rate can be tracked and logged during training.
    injected_opt_func = optax.inject_hyperparams(
        opt_func, hyperparam_dtype=jax.numpy.float32
    )
    # Call the optimizer function with the extracted kwargs
    try:
      return injected_opt_func(**opt_kwargs)
    except TypeError as e:
      raise TypeError(
          f"Error calling {opt_type} with arguments {opt_kwargs}. "
          f"Check if the arguments match the signature of optax.{opt_type}: {e}"
      ) from e

  def parse_mesh_config(
      self, model_key: str
  ) -> tuple[tuple[int, ...], tuple[str, ...]]:
    # [FUNCTION DOC]
    # Parses the mesh spec of a model config section. Reads config[model_key].mesh, which must be a
    # non-empty mapping with non-empty "shape" and "axis_names" strings; each is literal-evaluated
    # (ValueError rewrapped). Shape must evaluate to a tuple of ints and axis_names to a tuple of
    # strings, of equal length; violations raise ValueError. Output: (axis_shapes, axis_names) tuples.
    # KNOWN ISSUE: the error text says shape may be "a list or tuple" but a list literal is rejected;
    # literal evaluation raising SyntaxError (malformed text) is not converted to ValueError.
    """Validates and parses the mesh shape and axis names for one model.

    Args:
      model_key: Config section name such as ``model_config`` or
        ``actor_model_config``.

    Returns:
      A tuple ``(axis_shapes, axis_names)`` ready to pass to
      ``tunix.utils.mesh.create_mesh``.

    Raises:
      ValueError: If the mesh config is missing, malformed, or internally
        inconsistent.
    """
    mesh_config = self.config[model_key].get("mesh")
    if not mesh_config:

      raise ValueError("Missing 'mesh' configuration in raw_keys.")

    if not isinstance(mesh_config, collections.abc.Mapping):
      raise ValueError(
          "The 'mesh' configuration must be a dictionary-like object, got"
          f" {type(mesh_config)}."
      )

    shape = mesh_config.get("shape")
    if not shape:
      raise ValueError("Missing 'shape' key in 'mesh' configuration.")
    names = mesh_config.get("axis_names")
    if not names:
      raise ValueError("Missing 'axis_names' key in 'mesh' configuration.")

    try:
      axis_shapes = ast.literal_eval(shape)
    except ValueError as e:
      raise ValueError(
          "Invalid 'shape' key in 'mesh' configuration:"
          f" {mesh_config.get('shape')}"
      ) from e
    try:
      axis_names = ast.literal_eval(names)
    except ValueError as e:
      raise ValueError(
          "Invalid 'axis_names' key in 'mesh' configuration:"
          f" {mesh_config.get('axis_names')}"
      ) from e

    if not isinstance(axis_shapes, tuple):
      raise ValueError(
          f"'mesh.shape' must be a list or tuple, got {type(axis_shapes)}."
      )
    if not all(isinstance(x, int) for x in axis_shapes):
      raise ValueError(
          f"All elements in mesh.shape must be integers, got {axis_shapes}."
      )

    if not isinstance(axis_names, tuple):
      raise ValueError(
          f"'mesh.axis_names' must be a tuple, got {type(axis_names)}."
      )
    if not all(isinstance(x, str) for x in axis_names):
      raise ValueError(
          f"All elements in mesh.axis_names must be strings, got {axis_names}."
      )

    if len(axis_shapes) != len(axis_names):
      raise ValueError(
          f"mesh.shape {axis_shapes} and mesh.axis_names {axis_names} "
          "must have the same length."
      )
    return tuple(axis_shapes), tuple(axis_names)

  def _parse_mesh_allocation_policy(self, model_key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the normalized mesh allocation policy (COMPACT or PERFORMANCE; default COMPACT
    # when unset) of config[model_key].mesh via the mesh utility; ValueError if the mesh section is
    # missing or not a mapping, or the policy is invalid.
    """Validates and returns the mesh allocation policy for one model.

    Mesh allocation policy controls how Tunix chooses device subsets when a
    mesh must be carved out of a larger device pool.

    Supported values are:

    * ``COMPACT``: prefer the smallest remaining region that can satisfy the
      request.
    * ``PERFORMANCE``: prefer the most cubical supported extracted shape.

    When ``mesh.allocation_policy`` is omitted, this defaults to ``COMPACT``.

    Args:
      model_key: Config section name such as ``model_config`` or
        ``actor_model_config``.

    Returns:
      The normalized allocation policy string.

    Raises:
      ValueError: If the mesh config is missing or the policy value is not
        supported.
    """
    mesh_config = self.config[model_key].get("mesh")
    if not mesh_config:
      raise ValueError("Missing 'mesh' configuration in raw_keys.")
    if not isinstance(mesh_config, collections.abc.Mapping):
      raise ValueError(
          "The 'mesh' configuration must be a dictionary-like object, got"
          f" {type(mesh_config)}."
      )
    return mesh_lib.normalize_allocation_policy(
        mesh_config.get("allocation_policy")
    )

  def obtain_training_config_dict(self, key):
    # [FUNCTION DOC]
    # Builds a training-config dict from config[key]. ValueError if the section is not a mutable
    # mapping. For each entry: non-empty checkpointing_options become orbax CheckpointManagerOptions;
    # non-empty metrics_logging_options become MetricsLoggerOptions; profiler_options become
    # ProfilerOptions or None when empty; perf_metrics_options become PerfMetricsOptions or None when
    # empty; any key containing "optimizer_config" is dropped; everything else (including empty
    # checkpointing/metrics options) is copied as is. ValueErrors from constructors are rewrapped.
    # Output: a defaultdict without a default factory (behaves like a dict).
    # KNOWN ISSUE: the loop variable shadows the key parameter, and the error message always says
    # 'training_config' regardless of the key; TypeError from bad option fields is not rewrapped.
    """Obtain training config dictionary from specified key in self.config.

    Check and construct each component in training config and return them in a
    dictionary, which is ready to be used to create the training config object.

    Args:
      key: The key of the training config in the self.config dictionary.

    Returns:
      A dictionary constructed training config.
    """
    training_config = self.config[key]
    if not isinstance(training_config, collections.abc.MutableMapping):
      raise ValueError(
          "Expected 'training_config' to be a dictionary, but got "
          f"{type(training_config).__name__}"
      )

    constructed_training_config = collections.defaultdict()
    for key, value in training_config.items():
      if key == "checkpointing_options" and value:
        try:
          constructed_training_config[key] = ocp.CheckpointManagerOptions(
              **value
          )
        except ValueError as e:
          raise ValueError(f"Invalid checkpointing options: {value}") from e
      elif key == "metrics_logging_options" and value:
        try:
          constructed_training_config[key] = (
              metrics_logger.MetricsLoggerOptions(**value)
          )
        except ValueError as e:
          raise ValueError(f"Invalid metrics logging options: {value}") from e
      elif key == "profiler_options":
        if value:
          try:
            constructed_training_config[key] = profiler.ProfilerOptions(**value)
          except ValueError as e:
            raise ValueError(f"Invalid profiler options: {value}") from e
        else:
          constructed_training_config[key] = None
      elif key == "perf_metrics_options":
        if value:
          try:
            constructed_training_config[key] = perf_metrics.PerfMetricsOptions(
                **value
            )
          except ValueError as e:
            raise ValueError(f"Invalid perf metrics options: {value}") from e
        else:
          constructed_training_config[key] = None
      elif "optimizer_config" in key:
        continue
      else:
        constructed_training_config[key] = value

    return constructed_training_config

  def _update_from_env_and_command_line(
      self,
      raw_keys: collections.OrderedDict[str, Any],
      raw_data_from_yaml: dict[str, Any],
      overrides: list[str],
      **_kwargs,
  ):
    # [FUNCTION DOC]
    # HELPER: merges YAML values with CLI and environment overrides into raw_keys.
    # Inputs: raw_keys (output ordered dict, mutated), the base YAML config, list of override strings,
    # ignored kwargs. Logic: parses overrides with OmegaConf from_cli into a plain container. Any
    # non-empty top-level CLI key not in the YAML raises ValueError. For each YAML key in order: a key
    # set by both CLI and env raises ValueError; a key set by neither copies the YAML value. Otherwise
    # the key is recorded as updated and the proposal is the CLI value, else the env string; it is
    # normalized (None to "" for string fields, recursively). If the YAML value is null: a None
    # proposal stays None, an acceptable-type proposal is deep-copied, else ValueError. If the proposal
    # is not an instance of the YAML value's type and that type has no parser, ValueError. Then: None
    # stays None; same-type values are taken as is; otherwise the type's parser converts it (e.g. env
    # string to int, "true"/"false" to bool, dict config to dict). Converted dicts replace the value
    # for replace_keys, otherwise are deep-merged into the YAML value via update_dict.
    # Output: list of updated key names. Exceptions: ValueError and parser errors.
    # KNOWN ISSUE: environment-variable overrides of dict/list keys are raw strings and are passed
    # to the dict/list constructor, which fails or splits the string into characters instead of parsing it.
    """Update the configuration from command line."""

    cli_cfg = omegaconf.OmegaConf.from_cli(overrides)

    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cli_cfg, resolve=True
    )

    updated_keys = []

    # Check for conflicts and unknown keys.
    for k in raw_data_from_cmd_line:
      if not k:
        continue
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    # Iterate over key from base yaml
    for k in raw_data_from_yaml:

      # Error out if same key defined in cmd line and environment
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(
            f"You are passing overrides by both CLI and ENV for `{k}`. This"
            " isn't allowed."
        )

      # Take value from base config yaml if key is not specified in command line
      # or environment.
      if (
          k not in raw_data_from_cmd_line
          and yaml_key_to_env_key(k) not in os.environ
      ):
        # take the config value from the YAML file.
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      #  Key is specified on either command line or enviornment
      updated_keys.append(k)

      # take updated value from command line or enviornment
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      new_proposal = _normalize_cli_override(
          raw_data_from_yaml[k], new_proposal
      )

      if raw_data_from_yaml[k] is None:
        if new_proposal is None:
          raw_keys[k] = None
          continue
        if _can_override_nullable_schema(new_proposal):
          raw_keys[k] = copy.deepcopy(new_proposal)
          continue
        raise ValueError(
            f"For key '{k}', nullable schema can't accept value of type"
            f" {type(new_proposal)} from the CLI or ENV"
        )

      # If specified value is not one of type in base config yaml or is not
      # consumed by to type parser, error out
      # TODO(b/477343879): ensure Type checking for values with no defaults such
      # as lora_config
      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
          type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in"
            f" {_yaml_types_to_parser.keys()}, can't pass at the CLI or ENV"
        )

      # Take the config value
      if new_proposal is None:
        raw_keys[k] = None
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        parsed_new_proposal = _yaml_types_to_parser[
            type(raw_data_from_yaml[k])
        ](
            new_proposal
        )  # take the command line value, but type it like the config value.

        if isinstance(parsed_new_proposal, dict):
          if k in self.replace_keys:
            raw_keys[k] = parsed_new_proposal
          else:
            # merge the dict recursively
            raw_keys[k] = self.update_dict(
                schema=raw_data_from_yaml[k], source=parsed_new_proposal
            )
        else:
          raw_keys[k] = parsed_new_proposal

    return updated_keys

  def update_dict(self, schema: dict[str, Any], source: dict[str, Any]):
    # [FUNCTION DOC]
    # Recursively merges source into a deep copy of schema. For keys in replace_keys the source value
    # replaces entirely (deep-copied); when both values are mappings, merges recursively; otherwise the
    # source value overwrites. Afterwards removes any replace_keys present in the output but absent
    # from source. Output: the new dict; inputs not mutated.
    # KNOWN ISSUE: removing replace_keys not present in the source drops base-config sections (e.g. a
    # nested lora_config) whenever a parent dict is overridden without mentioning them, which is
    # surprising for a "merge".
    """Recursively updates a dictionary with values from another dictionary.

    Uses the `self.replace_keys` set to determine which keys from the source
    should completely overwrite existing values in the schema.

    Args:
        schema (dict): The base dictionary to be updated.
        source (dict): The dictionary containing updates.

    Returns:
        dict: A new dictionary with updates applied.
    """
    output = copy.deepcopy(schema)
    for key, source_val in source.items():

      if key in self.replace_keys:
        # For keys in self.replace_keys, take the value from source entirely.
        output[key] = copy.deepcopy(source_val)
      else:
        output_val = output.get(key)
        # Check if both source and output values are dictionaries for merging.
        if isinstance(
            source_val,
            collections.abc.Mapping | omegaconf.dictconfig.DictConfig,
        ) and isinstance(
            output_val,
            collections.abc.Mapping | omegaconf.dictconfig.DictConfig,
        ):
          # Both are dictionaries, so recurse.
          # The recursive call uses the same self.replace_keys instance.
          output[key] = self.update_dict(output_val, source_val)
        else:
          # Otherwise (not both dictionaries), the source value overwrites.
          output[key] = copy.deepcopy(source_val)

    # Identify keys that are in self.replace_keys and were in the
    # original schema
    # (hence in output now) but are NOT in the source dictionary.
    keys_to_remove = []
    for key in self.replace_keys:
      if key in output and key not in source:
        keys_to_remove.append(key)

    # Remove these keys from the output dictionary.
    for key in keys_to_remove:
      del output[key]

    return output

  def _validate_env_variable(self, raw_data_from_yaml):
    # [FUNCTION DOC]
    # HELPER: for every environment variable starting with "T_", requires the lowercased remainder to
    # be a YAML key and the remainder to be all uppercase; otherwise raises ValueError.
    """Validate the environment variables."""
    for environment_var in os.environ:
      if environment_var[: len(_TUNIX_PREFIX)] == _TUNIX_PREFIX:
        proposed_key = environment_var[len(_TUNIX_PREFIX) :].lower()
        if proposed_key not in raw_data_from_yaml:
          raise ValueError(
              f"We received env {environment_var} but it doesn't match a key,"
              " so it is assumed a mistake."
          )
        if not environment_var[len(_TUNIX_PREFIX) :].isupper():
          raise ValueError(
              f"We received env {environment_var} but it isn't all uppercase."
          )

  def _load_config_from_yaml(self, config_path: str):
    # [FUNCTION DOC]
    # HELPER: loads a YAML file with OmegaConf; FileNotFoundError becomes ValueError. Output: an
    # OmegaConf config object.
    """Try Loading and validate the configuration from the YAML file."""

    try:
      config_oconf = omegaconf.OmegaConf.load(config_path)
    except FileNotFoundError as e:
      raise ValueError(f"Config {config_path} not found.") from e

    return config_oconf

  def obtain_reward_fn(self) -> list[Callable[..., Any]]:
    # [FUNCTION DOC]
    # Loads reward functions listed in config["reward_functions"].
    # Logic: for each entry, if it contains no "/" tries importing it as a dotted module (failures
    # logged as warnings). If that did not succeed, treats it as a path relative to the project root,
    # loads it as a source file module named after the file base name, and executes it (ImportError on
    # missing spec/loader or execution failure). If config["verl_compatible"] is true, appends one
    # wrapper per module that takes prompts, completions and reward_model (plus kwargs), reads the
    # "ground_truth" list from reward_model and returns the module's compute_score for each
    # completion/ground-truth pair. Otherwise appends every public (non-underscore) function defined
    # in that module itself (not imported ones), in alphabetical order.
    # Output: list of callables. Exceptions: ImportError, RuntimeError, KeyError for missing config.
    # Side effects: imports/executes arbitrary module code, logging.
    # KNOWN ISSUE: only the top-level importlib package is imported; the machinery and util submodules
    # are relied upon to have been imported elsewhere.
    """Obtain reward function from the config."""
    project_root = get_project_root()
    reward_fns = []
    for reward_fn_path in self.config["reward_functions"]:

      module = None
      # If the path is relative, try importing the module directly.
      if "/" not in reward_fn_path:
        try:
          module = importlib.import_module(reward_fn_path)
          module_name = module.__name__
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              "'%s' import failed: %s", reward_fn_path, e, exc_info=True
          )
          module = None

      # Try importing the module from the project root.
      if module is None:
        full_path = str(project_root / reward_fn_path)
        module_name = os.path.splitext(os.path.basename(full_path))[0]
        # load from source
        loader = importlib.machinery.SourceFileLoader(module_name, full_path)
        spec = importlib.util.spec_from_loader(module_name, loader)

        if spec is None:
          raise ImportError(f"Cannot find spec for module at {full_path}")
        if spec.loader is None:
          raise ImportError(f"Spec for module {module_name} has no loader")

        module = importlib.util.module_from_spec(spec)

        try:
          spec.loader.exec_module(module)
        except Exception as e:
          raise ImportError(
              f"Failed to execute module {module_name} from {full_path}"
          ) from e
      if module is None:
        raise RuntimeError(
            f"Module from path '{reward_fn_path}' failed to load."
        )

      loaded_module = module
      if self.config["verl_compatible"]:

        def reward_fn(
            prompts, completions, reward_model, *, lm=loaded_module, **kwargs
        ):
          # [FUNCTION DOC]
          # HELPER: verl-compatible reward wrapper; ignores prompts and kwargs, zips completions with
          # reward_model["ground_truth"] and returns compute_score(completion, ground_truth) from the bound module.
          del prompts, kwargs
          ground_truths = reward_model["ground_truth"]
          return [
              lm.compute_score(c, gt)
              for c, gt in zip(completions, ground_truths)
          ]

        reward_fns.append(reward_fn)

      else:
        # Get all defined functions in the file as reward functions.
        # We explicitly ignore functions whose names start with an underscore
        # (_), ensuring private helper functions are never mistaken for
        # reward functions.
        defined_functions = []
        for name, member in inspect.getmembers(module):
          if inspect.isfunction(member) and not name.startswith("_"):
            # Check if the function was defined in this module
            if member.__module__ == module_name:
              defined_functions.append(member)
        reward_fns.extend(defined_functions)
    return reward_fns

  def _get_function_from_path(self, path_str):
    # [FUNCTION DOC]
    # HELPER: imports "package.module.function" by splitting at the last dot, importing the module
    # and getting the attribute. Returns None (logging a warning) on ImportError, AttributeError or
    # ValueError (e.g. no dot).
    """Dynamically imports a function from a string path.

    Args:
      path_str: The string path to the function, e.g.
        "tunix.rl.reward_fn.check_answer"

    Returns:
      The dynamically imported function, or None if the import fails.
    """
    try:
      # Split the path into module and function name
      module_path, function_name = path_str.rsplit(".", 1)

      # Import the module
      module = importlib.import_module(module_path)

      # Get the function from the module
      function = getattr(module, function_name)
      return function
    except (ImportError, AttributeError, ValueError) as e:
      logging.warning("Error importing '%s': %s", path_str, e)
      return None


def initialize(argv, **kwargs):
  # [FUNCTION DOC]
  # Convenience factory returning a configuration object built from argv and kwargs.
  return HyperParameters(argv, **kwargs)
