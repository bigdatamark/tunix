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
# HELPER: Small utility module with a metaclass that marks instances as
# "locked" once construction is complete, enabling classes to implement
# immutability in their own attribute-setting hook.
"""Utilities for enforcing immutability in Python objects."""

import abc


class ImmutableMeta(abc.ABCMeta):
  # [CLASS DOC]
  # Metaclass deriving from the abstract-base-class metaclass (so it is usable
  # with ABCs). It only overrides instance creation: after the normal creation
  # and full initializer chain run, it sets the private attribute "_locked" to
  # True on the instance via the base object attribute setter (bypassing any
  # overridden setattr). Enforcement (raising on assignment when locked) is the
  # responsibility of the class using the metaclass.
  """Metaclass that sets `self._locked = True` after `__init__` completes.

  By overriding `__call__`, this metaclass ensures that the `_locked` flag is
  only set after the entire instantiation process (including the full
  `__init__` chain of subclasses) has finished. This allows subclasses to
  perform their own initialization normally before the immutability constraint
  is applied.

  Note that this metaclass alone does not enforce immutability; it only provides
  the signal. Classes must explicitly implement `__setattr__` to check the
  `_locked` flag and raise an error (as shown in the usage example).

  Usage:
    class MyImmutableClass(metaclass=ImmutableMeta):

      def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
          raise AttributeError("Immutable!")
        super().__setattr__(name, value)
  """

  def __call__(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # Instance creation hook. Inputs: cls (the class being instantiated) plus
    # arbitrary positional/keyword constructor arguments, forwarded unchanged to
    # the default creation. Output: the fully initialized instance.
    # Side effects: sets instance attribute "_locked" = True, bypassing the
    # class's own setattr. Exceptions: whatever construction raises.
    """Create and initialize a new instance, then lock it against modification."""
    # This calls __new__ and __init__, completing the full initialization chain
    instance = super().__call__(*args, **kwargs)
    # Lock the instance after init is done
    object.__setattr__(instance, "_locked", True)
    return instance
