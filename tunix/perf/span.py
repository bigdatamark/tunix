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
# Span-based data model for performance tracing. A Span is a named time
# interval (begin, end in seconds as floats; end is +infinity until ended). A
# SpanGroup is a named interval holding an ordered list of child Spans and
# SpanGroups, forming a tree with parent ("outer") links; the root's outer is
# itself. Also provides tree printing, stack cloning, batch queries over lists
# of groups, deep cloning, and merging of two trees with overlap validation.
# JaxDevice is an alias of Any used as a type placeholder by other modules.
"""Span based data model."""

from __future__ import annotations

import math
from typing import Any

from absl import logging


JaxDevice = Any


class Span:
  # [CLASS DOC]
  # Leaf time interval with attributes name, begin, end. No validation that
  # end >= begin.
  name: str
  begin: float
  end: float

  def __init__(self, name: str, begin: float):
    # [FUNCTION DOC]
    # Stores name and begin; end initialized to +infinity (not yet ended).
    self.name = name
    self.begin = begin
    self.end = float("inf")

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns "name: begin, end" with times formatted to 6 decimals.
    out = f"{self.name}: {self.begin:.6f}, {self.end:.6f}"
    return out

  @property
  def ended(self) -> bool:
    # [FUNCTION DOC]
    # Property: True if end is not +infinity.
    return self.end != float("inf")

  @property
  def duration(self) -> float:
    # [FUNCTION DOC]
    # Property: end minus begin (infinity if not ended).
    return self.end - self.begin


class SpanGroup:
  # [CLASS DOC]
  # Tree node: name, begin (default 0), end (default +infinity), outer
  # (parent; self for the root) and inner (ordered children list). Lookup
  # helpers search only direct children by exact name.
  """Organizes spans into a tree structure."""

  name: str
  begin: float
  end: float
  outer: SpanGroup
  inner: list[Span | SpanGroup]

  def __init__(self, name: str, outer: SpanGroup | None = None):
    # [FUNCTION DOC]
    # Inputs: name, optional outer parent. Sets begin 0, end +infinity, empty
    # inner. If outer is given, links to it and appends self to the parent's
    # inner list (side effect on parent); otherwise outer refers to self.
    self.name = name
    self.begin = 0
    self.end = float("inf")
    self.inner = []
    if outer is not None:
      self.outer = outer
      outer.inner.append(self)
    else:
      self.outer = self

  def __repr__(self) -> str:
    # [FUNCTION DOC]
    # Returns "name: begin, end" with times formatted to 6 decimals.
    out = f"{self.name}: {self.begin:.6f}, {self.end:.6f}"
    return out

  @property
  def duration(self) -> float:
    # [FUNCTION DOC]
    # Property: end minus begin.
    return self.end - self.begin

  def find_first_inner_group(self, name: str) -> SpanGroup | None:
    # [FUNCTION DOC]
    # Returns the first direct child SpanGroup with the given name, or None.
    for item in self.inner:
      if isinstance(item, SpanGroup) and item.name == name:
        return item
    return None

  def find_last_inner_group(self, name: str) -> SpanGroup | None:
    # [FUNCTION DOC]
    # Returns the last direct child SpanGroup with the given name, or None.
    for item in reversed(self.inner):
      if isinstance(item, SpanGroup) and item.name == name:
        return item
    return None

  def find_all_inner_groups(self, name: str) -> list[SpanGroup]:
    # [FUNCTION DOC]
    # Returns all direct child SpanGroups with the given name, in order.
    return [
        item
        for item in self.inner
        if isinstance(item, SpanGroup) and item.name == name
    ]

  def find_last_inner_span(self, name: str) -> Span | None:
    # [FUNCTION DOC]
    # Returns the last direct child Span (leaf) with the given name, or None.
    for item in reversed(self.inner):
      if isinstance(item, Span) and item.name == name:
        return item
    return None

  def find_all_inner_spans(self, name: str) -> list[Span]:
    # [FUNCTION DOC]
    # Returns all direct child Spans (leaves) with the given name, in order.
    return [
        item
        for item in self.inner
        if isinstance(item, Span) and item.name == name
    ]


def span_group_tostring(group: SpanGroup, born: float = 0.0) -> str:
  # [FUNCTION DOC]
  # Renders a group tree as text: one line per node, "- name (begin, end)" with
  # times relative to born (6 decimals), indented two spaces per depth level,
  # depth-first in child order, each line newline-terminated. Output: string.
  """Converts a span group tree to a string."""

  def _tostring_recursive(out: str, group: SpanGroup, indent: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Appends the group line at the given indent, then for each child
    # either recurses (groups) or appends a leaf line one level deeper.
    # Returns the accumulated string.
    out += "  " * indent
    out += f"- {group.name} ({group.begin-born:.6f}, {group.end-born:.6f})\n"
    for item in group.inner:
      if isinstance(item, SpanGroup):
        out += _tostring_recursive("", item, indent + 1)
      elif isinstance(item, Span):
        out += "  " * (indent + 1)
        out += f"- {item.name} ({item.begin-born:.6f}, {item.end-born:.6f})\n"
    return out

  return _tostring_recursive("", group, 0)


def span_group_print(group: SpanGroup, born: float = 0.0) -> None:
  # [FUNCTION DOC]
  # Prints span_group_tostring(group, born) to stdout.
  """Prints the span group tree."""
  print(span_group_tostring(group, born))


def span_group_stack_clone(stack: list[SpanGroup]) -> list[SpanGroup]:
  # [FUNCTION DOC]
  # Clones a root-to-leaf stack (list of nested groups) into a fresh chain of
  # SpanGroups with the same names/begin/end, each new group linked as a child
  # of the previous one; children not on the stack are not copied.
  # Output: list of new groups in the same order.
  # Exceptions: IndexError on an empty stack.
  outer = SpanGroup(stack[0].name, None)
  outer.begin = stack[0].begin
  outer.end = stack[0].end
  stack_clone = [outer]
  for group in stack[1:]:
    inner = SpanGroup(group.name, outer)
    inner.begin = group.begin
    inner.end = group.end
    stack_clone.append(inner)
    outer = inner
  return stack_clone


# TODO(yangmu) create a new class SpanGroupBatch.


def span_group_batch_query_first(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  # [FUNCTION DOC]
  # For each group in batch, collects its first direct child group named name
  # (skipping groups without one). Output: list.
  out_batch = []
  for group in batch:
    inner = group.find_first_inner_group(name)
    if inner is not None:
      out_batch.append(inner)
  return out_batch


def span_group_batch_query_last(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  # [FUNCTION DOC]
  # For each group in batch, collects its last direct child group named name
  # (skipping groups without one). Output: list.
  out_batch = []
  for group in batch:
    inner = group.find_last_inner_group(name)
    if inner is not None:
      out_batch.append(inner)
  return out_batch


def span_group_batch_query_nth(
    batch: list[SpanGroup], name: str, index: int
) -> list[SpanGroup]:
  # [FUNCTION DOC]
  # For each group in batch, collects its index-th (0-based, non-negative)
  # direct child group named name when it exists. Output: list.
  out_batch = []
  for group in batch:
    inners = group.find_all_inner_groups(name)
    if 0 <= index < len(inners):
      out_batch.append(inners[index])
  return out_batch


def span_group_batch_query_all(
    batch: list[SpanGroup], name: str
) -> list[SpanGroup]:
  # [FUNCTION DOC]
  # Concatenates all direct child groups named name from every group in batch.
  out_batch = []
  for group in batch:
    out_batch.extend(group.find_all_inner_groups(name))
  return out_batch


def clone_span_or_group(
    node: Span | SpanGroup, outer: SpanGroup | None = None
) -> Span | SpanGroup:
  # [FUNCTION DOC]
  # Deep-clones a Span or SpanGroup (names, begin, end, children recursively),
  # appending the clone to outer's children when outer is given. A cloned
  # group without outer becomes its own root. Output: the clone.
  """Clones a span or span group recursively.

  Args:
    node: The span or span group to clone.
    outer: The parent span group to attach the cloned node to.

  Returns:
    The cloned span or span group.
  """
  if isinstance(node, SpanGroup):
    new_group = SpanGroup(node.name, outer)
    new_group.begin = node.begin
    new_group.end = node.end
    for child in node.inner:
      clone_span_or_group(child, new_group)
    return new_group
  else:
    new_span = Span(node.name, node.begin)
    new_span.end = node.end
    if outer is not None:
      outer.inner.append(new_span)
    return new_span


def _are_nodes_shallowly_identical(
    node1: Span | SpanGroup, node2: Span | SpanGroup
) -> bool:
  # [FUNCTION DOC]
  # HELPER: True if node2 is an instance of node1's type, names are equal, and
  # begin and end are each close (relative tolerance 1e-15, absolute 1e-9).
  # Children are not compared.
  """Checks if two nodes are identical in type, name, start, and end.

  This only checks the root node, not the entire tree.

  Args:
    node1: The first node to compare.
    node2: The second node to compare.

  Returns:
    True if the node roots are identical, False otherwise.
  """

  return (
      isinstance(node2, type(node1))
      and node1.name == node2.name
      and math.isclose(node1.begin, node2.begin, rel_tol=1e-15, abs_tol=1e-9)
      and math.isclose(node1.end, node2.end, rel_tol=1e-15, abs_tol=1e-9)
  )


def _merge_span_group_trees_inplace(
    target: SpanGroup, source: SpanGroup
) -> None:
  # [FUNCTION DOC]
  # HELPER: Merges source's children into target in place: appends them to
  # target (re-parenting groups), stable-sorts by begin, then sweeps: when the
  # current node's end is after the next node's begin (overlap), the two must
  # be shallowly identical (else ValueError); identical groups are merged
  # recursively and the duplicate dropped, identical spans keep only the first.
  # Non-overlapping nodes are kept. Side effects: mutates target and moves
  # source's child objects (source's inner list itself is left unchanged).
  # KNOWN ISSUE: unended nodes (end = infinity) overlap everything after them
  # and trigger ValueError; identical zero-length nodes are not deduplicated.
  """Merges source into target in-place.

  Args:
    target: The target span group tree (mutable).
    source: The source span group tree (mutable).

  Raises:
    ValueError: If overlapping nodes are not identical.
  """
  # Append source children to target. Parent pointers are updated later.
  for child in source.inner:
    target.inner.append(child)
    if isinstance(child, SpanGroup):
      child.outer = target

  target.inner.sort(key=lambda x: x.begin)

  # Resolve overlaps
  merged_inner = []
  if target.inner:
    current = target.inner[0]
    for next_node in target.inner[1:]:
      if current.end > next_node.begin:
        if _are_nodes_shallowly_identical(current, next_node):
          if isinstance(current, SpanGroup) and isinstance(
              next_node, SpanGroup
          ):
            _merge_span_group_trees_inplace(current, next_node)
          # If Spans, keep current (drop duplicate next_node)
        else:
          raise ValueError(
              f"Merge is not possible: Overlap detected between {current!r}"
              f" and {next_node!r}"
          )
      else:
        merged_inner.append(current)
        current = next_node
    merged_inner.append(current)

  target.inner = merged_inner


def merge_span_group_trees(
    tree1: SpanGroup | Span, tree2: SpanGroup | Span
) -> SpanGroup | Span:
  # [FUNCTION DOC]
  # Merges two trees into a new tree. ValueError if roots are not shallowly
  # identical. Deep-clones both; if both are groups, merges the second clone
  # into the first (ValueError on non-identical overlaps). For spans, returns
  # the clone of the first. Inputs are not modified.
  """Merges two SpanGroup or Span trees into a new SpanGroup or Span.

  Validates that the input trees have identical root nodes and no overlapping
  spans at the same level. Does not modify the input trees.

  Args:
    tree1: The first SpanGroup or Span tree.
    tree2: The second SpanGroup or Span tree.

  Returns:
    A new SpanGroup or Span containing the merged result.

  Raises:
    ValueError: If the roots are not identical or if the merge results in
      overlapping spans.
  """
  if not _are_nodes_shallowly_identical(tree1, tree2):
    raise ValueError(
        f"Roots are not identical: {tree1!r} vs {tree2!r}. Merge expected"
        " identical roots."
    )

  target = clone_span_or_group(tree1)
  source = clone_span_or_group(tree2)

  # Merge if both are SpanGroups. For Spans, simply return target.
  if isinstance(target, SpanGroup) and isinstance(source, SpanGroup):
    _merge_span_group_trees_inplace(target, source)

  return target
