# [MODULE DOC]
# HELPER: Small utility module for converting PyTorch checkpoint parameter
# names into JAX/Flax parameter names during checkpoint loading.
# Contents: one function, torch_key_to_jax_key. No module-level state.
# Dependencies: standard library regular expressions only.
"""Common utilities for torch."""
import re


def torch_key_to_jax_key(mapping, source_key):
  # [FUNCTION DOC]
  # HELPER: Maps one PyTorch state-dict key to its JAX key plus reshape info.
  # Inputs: mapping - dict whose keys are regex patterns and whose values are
  #   2-tuples (replacement string, reshape descriptor; the descriptor is opaque
  #   here and returned unchanged); source_key - the torch parameter name.
  # Behavior: for every pattern that matches at the START of source_key
  #   (regex match semantics, not full-match), build a candidate tuple of
  #   (source_key with all occurrences of the pattern substituted by the
  #   replacement, reshape descriptor). Exactly one candidate must exist.
  # Output: the single (new_key, reshape) tuple.
  # Exceptions: ValueError if zero or more than one pattern matches; the message
  #   lists the candidates and the source key.
  # Side effects: none.
  """Convert torch key to jax key using the provided mapping."""
  subs = [
      (re.sub(pat, repl, source_key), reshape)
      for pat, (repl, reshape) in mapping.items()
      if re.match(pat, source_key)
  ]
  if len(subs) != 1:
    raise ValueError(
        f"Only one key should be found. Found: {subs} for {source_key}"
    )
  else:
    return subs[0]
