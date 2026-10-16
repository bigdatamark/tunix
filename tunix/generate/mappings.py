# [MODULE DOC]
# Shared infrastructure for describing how trainer-side (flax/nnx) model
# weights map onto inference-backend weights (vLLM, SGLang-JAX, HuggingFace
# naming). Provides:
# - BackendMappingMixin: class-method helpers a model class mixes in to look up
#   per-backend mapping dictionaries from a BACKEND_MAPPINGS registry defined in
#   the model's package.
# - MappingConfig: a dataclass bundling the six mapping artifacts (name
#   mappings, LoRA name mappings, hook functions, transpose keys, LoRA transpose
#   keys, a state preprocessing callable), with constructors from arbitrary
#   objects, dicts or a model.
# No license header and no import-time side effects.
"""Shared helpers and dataclasses for model weight mappings."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Dict, Optional, Tuple


class BackendMappingMixin:
  # [CLASS DOC]
  # Mixin providing class methods to retrieve backend-specific weight mappings.
  # Class attributes: DEFAULT_BACKEND is the string vllm_jax, used when no backend
  # is given; BACKEND_PACKAGE_PATH is None by default and may be overridden with a
  # dotted module path used to locate the registry.
  # Registry contract: the parent package of the located module exposes a
  # module-level dict named BACKEND_MAPPINGS mapping backend name to a dict with
  # optional keys to_hf_mappings, lora_to_hf_mappings, to_hf_transpose_keys,
  # lora_to_hf_transpose_keys, to_hf_hook_fns, preprocess_src_state.
  """Provides helper methods to retrieve backend-specific weight mappings."""

  DEFAULT_BACKEND = 'vllm_jax'
  # Subclasses can override this to explicitly set the path
  BACKEND_PACKAGE_PATH = None

  @classmethod
  def _backend_registry(cls) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: returns the backend registry dict for the class. Uses
    # BACKEND_PACKAGE_PATH if truthy, otherwise the class's defining module name.
    # Strips the last dotted component (if any dot exists) to get the package name,
    # imports it with importlib, and returns its BACKEND_MAPPINGS attribute, or an
    # empty dict if absent.
    # Side effects: imports the package (executing its init code on first import).
    # Raises ModuleNotFoundError/ImportError if the package cannot be imported.
    # KNOWN ISSUE: when BACKEND_PACKAGE_PATH is given it is still treated as a
    # module path and its last component is stripped, so it must point at a module
    # inside the package, not at the package itself, despite the name.
    # Use the explicit path if provided, otherwise fallback to the module path
    module = cls.BACKEND_PACKAGE_PATH or cls.__module__

    package_name = module.rsplit('.', 1)[0] if '.' in module else module
    package = importlib.import_module(package_name)

    return getattr(package, 'BACKEND_MAPPINGS', {})

  @classmethod
  def mapping_for(cls, backend: str | None = None) -> Dict[str, Any]:
    # [FUNCTION DOC]
    # Returns the mapping dict for a backend (defaults to DEFAULT_BACKEND when the
    # argument is None or empty). Raises RuntimeError naming the backend and class
    # when the backend is not a key of the registry.
    backend = backend or cls.DEFAULT_BACKEND
    registry = cls._backend_registry()
    if backend not in registry:
      raise RuntimeError(
          f'{backend} mappings not available for {cls.__name__}.'
      )
    return registry[backend]

  @classmethod
  def to_hf_mappings(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the to_hf_mappings entry for the backend. Raises RuntimeError if the
    # backend is missing (via mapping_for) or if the entry is missing or None.
    # KNOWN ISSUE: the error message interpolates the raw backend argument, which
    # prints None when the default backend was used.
    mapping = cls.mapping_for(backend).get('to_hf_mappings')
    if mapping is None:
      raise RuntimeError(
          f'{backend} to_hf_mappings missing for {cls.__name__}.'
      )
    return mapping

  @classmethod
  def lora_to_hf_mappings(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the lora_to_hf_mappings entry for the backend, or None if absent.
    # Raises RuntimeError if the backend is unknown.
    return cls.mapping_for(backend).get('lora_to_hf_mappings')

  @classmethod
  def to_hf_transpose_keys(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the to_hf_transpose_keys entry for the backend; any falsy value
    # (missing, None, empty dict) is normalized to None. Raises RuntimeError if the
    # backend is unknown.
    result = cls.mapping_for(backend).get('to_hf_transpose_keys')
    return result or None

  @classmethod
  def lora_to_hf_transpose_keys(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the lora_to_hf_transpose_keys entry for the backend, with falsy
    # values normalized to None. Raises RuntimeError if the backend is unknown.
    result = cls.mapping_for(backend).get('lora_to_hf_transpose_keys')
    return result or None

  @classmethod
  def to_hf_hook_fns(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the to_hf_hook_fns entry for the backend or None if absent. Raises
    # RuntimeError if the backend is unknown.
    return cls.mapping_for(backend).get('to_hf_hook_fns')

  @classmethod
  def preprocess_src_state(cls, backend: str | None = None):
    # [FUNCTION DOC]
    # Returns the preprocess_src_state entry (a callable transforming the source
    # state before weight transfer) for the backend or None if absent. Raises
    # RuntimeError if the backend is unknown.
    return cls.mapping_for(backend).get('preprocess_src_state')


@dataclass
class MappingConfig:
  # [CLASS DOC]
  # Mutable dataclass describing how to translate trainer weights to backend
  # weights. All six fields default to None:
  # - to_hf_mappings: dict from source parameter path to target name/sharding.
  # - lora_to_hf_mappings: same for LoRA parameters.
  # - to_hf_hook_fns: dict of per-key transformation functions.
  # - to_hf_transpose_keys: dict from key to axis permutation tuple.
  # - lora_to_hf_transpose_keys: same for LoRA parameters.
  # - preprocess_src_state: callable applied to the source state before mapping.
  # Construct with build (generic) or from_model.
  """Describes how to translate trainer weights into backend weights.

  Prefer using `MappingConfig.build(...)`:
  * `MappingConfig.build(mapping_obj=..., model=..., backend=...)` accepts an
    existing config/dict/object and optionally falls back to extracting the
    data from `model`.
  """

  to_hf_mappings: Optional[Dict[str, Any]] = None
  lora_to_hf_mappings: Optional[Dict[str, Any]] = None
  to_hf_hook_fns: Optional[Dict[str, Any]] = None
  to_hf_transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None
  lora_to_hf_transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None
  preprocess_src_state: Optional[Callable[[Any], Any]] = None

  @classmethod
  def build(
      cls,
      mapping_obj: Any | None = None,
      model: Any | None = None,
      backend: str | None = None,
  ) -> 'MappingConfig':
    # [FUNCTION DOC]
    # Class method building a MappingConfig from flexible inputs.
    # Inputs: mapping_obj (a MappingConfig, a dict, any object with attributes of
    # the six field names, or None), model (used only when mapping_obj is None),
    # backend name (forwarded to from_model).
    # Behavior: asserts at least one of mapping_obj/model is given (AssertionError
    # otherwise). Returns mapping_obj unchanged if it is already an instance of
    # this class. If mapping_obj is None returns from_model(model, backend).
    # Otherwise collects values: from a dict, all its items; from other objects,
    # only attributes named like the six fields. For each of the six field names,
    # if the value is callable it is called with no arguments and the result
    # used; if that call raises TypeError the callable itself is kept. Returns a
    # new instance with those values (missing keys become None).
    # Side effects: calling the callables may have arbitrary side effects.
    # KNOWN ISSUE: preprocess_src_state is meant to be a callable, but because every
    # callable value is invoked with no arguments, a preprocess function taking a
    # zero-argument signature would be replaced by its result; only callables that
    # raise TypeError on a no-argument call are preserved. Also a TypeError raised
    # inside a getter for another reason silently keeps the getter itself.
    # KNOWN ISSUE: the backend argument is ignored when mapping_obj is supplied.
    """Build the MappingConfg from existing MappingConfig, Dict or other qualified datastructure, or load from model."""
    assert (
        mapping_obj is not None or model is not None
    ), 'Either mapping_obj or model must be provided (both cannot be None)'

    if isinstance(mapping_obj, cls):
      return mapping_obj

    if mapping_obj is None:
      return cls.from_model(model, backend)

    keys = (
        'to_hf_mappings',
        'lora_to_hf_mappings',
        'to_hf_hook_fns',
        'to_hf_transpose_keys',
        'lora_to_hf_transpose_keys',
        'preprocess_src_state',
    )

    values: Dict[str, Any] = {}
    if isinstance(mapping_obj, dict):
      values.update(mapping_obj)
    else:
      for key in keys:
        if hasattr(mapping_obj, key):
          values[key] = getattr(mapping_obj, key)

    resolved: Dict[str, Any] = {}
    for key in keys:
      value = values.get(key)
      if callable(value):
        try:
          value = value()
        except TypeError:
          resolved[key] = value
          continue
      resolved[key] = value

    return cls(
        to_hf_mappings=resolved.get('to_hf_mappings'),
        lora_to_hf_mappings=resolved.get('lora_to_hf_mappings'),
        to_hf_hook_fns=resolved.get('to_hf_hook_fns'),
        to_hf_transpose_keys=resolved.get('to_hf_transpose_keys'),
        lora_to_hf_transpose_keys=resolved.get('lora_to_hf_transpose_keys'),
        preprocess_src_state=resolved.get('preprocess_src_state'),
    )

  @classmethod
  def from_model(
      cls,
      model: Any,
      backend: str = 'vllm_jax',
      **overrides: Any,
  ) -> MappingConfig:
    # [FUNCTION DOC]
    # Class method building a MappingConfig by reading attributes from a model.
    # Inputs: model object, backend name (default vllm_jax), and keyword overrides.
    # For each of the six field names, reads the attribute from the model (None if
    # missing); if callable, calls it with the backend name, and on TypeError calls
    # it again with no arguments; non-callables are used as-is. Then for each
    # override whose key is an existing attribute of the config, sets it; unknown
    # override keys are silently ignored. Returns the config.
    # Side effects: the model's attribute callables are invoked.
    # KNOWN ISSUE: preprocess_src_state on a model is typically itself the
    # preprocessing function, but it is called here (with backend), so the stored
    # value is its return value rather than the function, unless the model exposes
    # it as a factory.
    """Constructs a MappingConfig for the given model and backend."""

    def maybe_call(attr: str):
      # [FUNCTION DOC]
      # HELPER: reads attribute attr from the enclosing model. Returns None if
      # missing or None; if callable returns its result when called with backend,
      # falling back to a no-argument call when the first raises TypeError (a
      # TypeError from the second call propagates); otherwise returns the value.
      value = getattr(model, attr, None)
      if value is None:
        return None
      if callable(value):
        try:
          return value(backend)
        except TypeError:
          return value()
      return value

    config = MappingConfig(
        to_hf_mappings=maybe_call('to_hf_mappings'),
        lora_to_hf_mappings=maybe_call('lora_to_hf_mappings'),
        to_hf_hook_fns=maybe_call('to_hf_hook_fns'),
        to_hf_transpose_keys=maybe_call('to_hf_transpose_keys'),
        lora_to_hf_transpose_keys=maybe_call('lora_to_hf_transpose_keys'),
        preprocess_src_state=maybe_call('preprocess_src_state'),
    )

    for key, value in overrides.items():
      if hasattr(config, key):
        setattr(config, key, value)

    return config
