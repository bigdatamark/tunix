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
# Client-facing RL cluster abstraction. Defines Mode (train/eval), Role
# (actor, critic, reference, reward, rollout), RLTrainingConfig (training
# config with RL batch-size knobs), ClusterConfig (meshes per role, rollout
# engine choice and configs, CPU offload), and RLCluster, which loads and
# places all models on their meshes, decides which roles share weights,
# builds the rollout engine (vanilla, vLLM, sglang-jax or a custom
# BaseRollout), an inference worker for critic/reference/reward models and
# RL trainers for actor and critic, and exposes generation, log-prob, value
# and reward computation, weight sync, CPU offloading and step-keyed metric
# buffering/logging. Aliases: ModelOrPath, MetricsT, MetricsBuffer.
"""Client facing abstraction for interacting with RL training cluster."""

import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import gc
import itertools
import operator
import os
from typing import Any, Callable, Mapping

from absl import logging
import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import jaxtyping
import numpy as np
import optax
from tunix.generate import tokenizer_adapter
# Internal placeholder for sglang_jax rollout worker stub, don't change this line.
# Internal placeholder for vllm rollout worker stub, don't change this line.
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import trainer as rl_trainer
from tunix.rl import utils as rl_utils
from tunix.rl.inference import inference_worker
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import sharding_utils
from tunix.sft import utils as sft_utils

ModelOrPath = nnx.Module | str
MetricsT = perf_metrics.MetricsT
MetricsBuffer = perf_metrics.MetricsBuffer


class Mode(enum.Enum):
  # [CLASS DOC]
  # Enum of workload mode with values "train" and "eval"; used to select the
  # rollout config and the metrics buffer.
  """Mode of RolloutConfig."""

  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the enum value string ("train" or "eval").
    return self.value


class Role(enum.Enum):
  # [CLASS DOC]
  # Enum of model roles: ACTOR (policy being trained), CRITIC (value model, PPO
  # only), REFERENCE (frozen policy for KL), REWARD (reward model), ROLLOUT
  # (generation copy of the policy). Values are lowercase names.
  """Role of the model."""

  ACTOR = "actor"  # policy model
  CRITIC = "critic"  # value model (only for PPO-style algos, not for GRPO)
  REFERENCE = "reference"  # kept fixed during training
  REWARD = "reward"
  ROLLOUT = "rollout"


@dataclasses.dataclass(slots=True, kw_only=True)
class RLTrainingConfig(peft_trainer.TrainingConfig):
  # [CLASS DOC]
  # Slotted keyword-only dataclass extending peft_trainer.TrainingConfig.
  # Fields: actor_optimizer (required optax transform), critic_optimizer
  # (optional), mini_batch_size (prompts per optimizer update),
  # train_micro_batch_size (prompts per gradient-accumulation micro step),
  # rollout_micro_batch_size and compute_logps_micro_batch_size (prompts per
  # generation / log-prob chunk). All optional sizes must be positive ints.
  # KNOWN ISSUE: the docstring states the divisibility relations backwards (the
  # full batch must be divisible by mini_batch_size, and mini_batch_size by
  # train_micro_batch_size); it also says the critic uses the actor optimizer
  # when critic_optimizer is None, but the critic trainer is given None.
  """RLTraining config.

  Attributes:
    actor_optimizer: Optimizer for the actor model.
    critic_optimizer: Optimizer for the critic model. If None, the critic model
      will be trained in the same optimizer as the actor model.
    mini_batch_size: The mini-batch size used for policy weight updates. One
      mini-batch corresponds to one optimizer update. `mini_batch_size` must be
      divisible by the global batch size.
    train_micro_batch_size: The micro-batch size used for gradient accumulation
      at training time. `train_micro_batch_size` must be divisible by
      `mini_batch_size`.
    rollout_micro_batch_size: The micro-batch size used for model rollouts.
    compute_logps_micro_batch_size: The micro-batch size used for computing log
      probabilities (e.g. for reference and old policy models).
  """

  actor_optimizer: optax.GradientTransformation
  critic_optimizer: optax.GradientTransformation | None = None
  mini_batch_size: int | None = None
  train_micro_batch_size: int | None = None
  rollout_micro_batch_size: int | None = None
  compute_logps_micro_batch_size: int | None = None

  def __post_init__(self):
    # [FUNCTION DOC]
    # Validates each optional batch size is a positive int (ValueError
    # otherwise). Raises ValueError if gradient_accumulation_steps was set by the
    # user. If train_micro_batch_size is set: requires mini_batch_size
    # (ValueError), requires mini_batch_size divisible by it (ValueError), and
    # sets gradient_accumulation_steps to their quotient (mutation).
    # Does not call the parent __post_init__.
    """Validates the configuration after initialization."""
    for name in [
        "mini_batch_size",
        "train_micro_batch_size",
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
    ]:
      rl_utils.is_positive_integer(getattr(self, name), name)

    if self.gradient_accumulation_steps is not None:
      raise ValueError(
          "For RL training, gradient_accumulation_steps should be None. It is "
          "automatically derived from: "
          "`mini_batch_size // train_micro_batch_size`."
      )

    if self.train_micro_batch_size is not None:
      if self.mini_batch_size is None:
        raise ValueError(
            "For RL training, `mini_batch_size` must be set when"
            " `train_micro_batch_size` is set."
        )
      rl_utils.check_divisibility(
          self.train_micro_batch_size,
          self.mini_batch_size,
          f"{self.train_micro_batch_size=}",
          f"{self.mini_batch_size=}",
      )
      self.gradient_accumulation_steps = (
          self.mini_batch_size // self.train_micro_batch_size
      )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClusterConfig:
  # [CLASS DOC]
  # Frozen keyword-only dataclass. Fields: role_to_mesh (dict Role to Mesh;
  # same meshes mean colocated roles), role_to_logical_axis_rule (optional
  # dict Role to logical axis rules), rollout_engine ("vanilla", "vllm",
  # "sglang_jax", a BaseRollout subclass, or a functools.partial of one;
  # default "vanilla"), offload_to_cpu (default False), training_config
  # (RLTrainingConfig, required), rollout_config (RolloutConfig or dict Mode
  # to RolloutConfig; must include TRAIN).
  # KNOWN ISSUE: the docstring documents rollout_vllm_* attributes that are not
  # fields here (they live in RolloutConfig). The role_to_mesh dict is
  # mutated by RLCluster despite the frozen dataclass.
  """Cluster config.

  Attributes:
    role_to_mesh: Mapping from model role to mesh. Key config for colocated vs
      disaggregated setup.
    role_to_logical_axis_rule: Mapping from model role to logical axis rule.
      This is used when models are sharded with logical axis and expects a
      logical to physical axis mapping at runtime.
    rollout_engine: Rollout engine to use. E.g. "vanilla", "vllm", "sglang_jax".
      Alternatively, if a subclass of `base_rollout.BaseRollout` is provided, it
      will be used as the rollout engine.
    offload_to_cpu: Whether to offload models to CPU at each step..
    training_config: RL training config.
    rollout_config: Rollout config. It may be different for different modes,
      e.g. TRAIN vs EVAL.
    rollout_vllm_model_version: Model version for vllm rollout engine.
    rollout_vllm_lora_config: LoRA config for vllm rollout engine.
    rollout_vllm_hbm_utilization: The percentage of TPU/GPU HBM allocated the
      vllm rollout engine.
    rollout_vllm_init_with_random_weights: Init the vllm TPU backend model with
      random weights instead of loading from the given path.
    rollout_vllm_tpu_backend_type: The TPU Jax backend type for vllm rollout
      engine, E.g. "jax", "torchax" or "pytorch_xla".
  """

  role_to_mesh: dict[Role, Mesh]
  role_to_logical_axis_rule: dict[Role, flax.typing.LogicalRules] | None = None
  rollout_engine: str | type[base_rollout.BaseRollout] = "vanilla"
  offload_to_cpu: bool = False

  training_config: RLTrainingConfig
  rollout_config: (
      dict[Mode, base_rollout.RolloutConfig] | base_rollout.RolloutConfig
  )


class RLCluster:
  # [CLASS DOC]
  # Owns all models and services for an RL run. Public attributes:
  # cluster_config, perf_config, r2m (role to mesh dict, same object as the
  # config's), tokenizer (TokenizerAdapter), global_steps (int, starts at 0).
  # Properties: rollout, inference_worker, actor_trainer, critic_trainer
  # (AttributeError when no separate critic trainer exists), perf, perf_v2.
  # Private state: backbone sharing map (role to list of roles sharing the
  # same weights), anchor policy state (pinned-host snapshot of actor weights
  # from the last sync, used for old log-probs), default memory kind, metrics
  # logger, buffered train/eval metric buffers (list of MetricsBuffer keyed by
  # global step), optional external metrics logger.
  """RLCluster."""

  def __init__(
      self,
      *,
      actor: ModelOrPath,
      critic: ModelOrPath | None = None,
      reference: ModelOrPath | None = None,
      reward: ModelOrPath | None = None,
      tokenizer: Any | None,
      cluster_config: ClusterConfig,
      perf_config: perf_metrics.PerfMetricsConfig | None = None,
  ):
    # [FUNCTION DOC]
    # Keyword-only inputs: actor (nnx module; paths unsupported), optional
    # critic, reference and reward models, tokenizer, cluster_config, optional
    # perf_config. Steps: build the backbone sharing map; load the train actor
    # on the actor mesh. Raises ValueError if rollout_config is None or is a dict
    # lacking a TRAIN entry. Rollout actor: same object as train actor if actor
    # and rollout meshes match; for the vanilla engine a separately loaded copy
    # on the rollout mesh cast to the TRAIN rollout data_type; otherwise the
    # train actor (initial weights for external engines). Loads reference (warns
    # if colocated with a LoRA actor but not sharing backbone), critic (and, if
    # critic shared backbone with the actor, copies non-LoRA actor state into
    # it) and reward models. Wraps the tokenizer, creates the metrics logger and
    # empty buffers, calls _init_cluster, runs garbage collection, and sets
    # global_steps to 0.
    # KNOWN ISSUE: Role.CRITIC is never added to the sharing map, so the critic
    # backbone-copy branch is dead code.
    self.cluster_config = cluster_config
    self.perf_config = perf_config
    self.r2m = cluster_config.role_to_mesh
    self._init_backbone_sharing_map(actor, reference)
    self._anchor_policy_state = None

    self._default_memory_kind = jax.devices()[0].default_memory().kind
    self.train_actor = self._load_model(actor, self.r2m[Role.ACTOR])

    if self.cluster_config.rollout_config is None:
      raise ValueError("`cluster_config.rollout_config` cannot be None.")
    if isinstance(
        self.cluster_config.rollout_config, dict
    ) and not self.cluster_config.rollout_config.get(Mode.TRAIN):
      raise ValueError(
          "Rollout config is a dict but missing a train config. Provided"
          f" config: {self.cluster_config.rollout_config}"
      )

    if Role.ROLLOUT in self._backbone_sharing_map[Role.ACTOR]:
      self.rollout_actor = self.train_actor
    elif self.cluster_config.rollout_engine == "vanilla":
      rollout_data_type = (
          self.cluster_config.rollout_config[Mode.TRAIN].data_type
          if isinstance(self.cluster_config.rollout_config, dict)
          else self.cluster_config.rollout_config.data_type
      )
      self.rollout_actor = self._load_model(
          actor,
          self.r2m[Role.ROLLOUT],
          rollout_data_type,
      )
    else:
      # Provide initial weights for non vanilla rollout engines.
      self.rollout_actor = self.train_actor

    if reference:
      self.reference = self._load_model(reference, self.r2m[Role.REFERENCE])
      if Role.REFERENCE in self._backbone_sharing_map[Role.ACTOR]:
        if not rl_utils.is_sharing_backbone(self.reference, self.train_actor):
          logging.warning(
              "Reference model and actor model are colocated but do not share"
              " the same backbone. This will result in an unnecessary model"
              " copy and increased HBM usage."
          )
    else:
      self.reference = None
    self.critic = (
        self._load_model(critic, self.r2m[Role.CRITIC]) if critic else None
    )
    if Role.CRITIC in self._backbone_sharing_map[Role.ACTOR]:
      critic_state = nnx.state(self.train_actor, filterlib.Not(nnx.LoRAParam))
      nnx.update(self.critic, critic_state)
    self.reward = (
        self._load_model(reward, self.r2m[Role.REWARD]) if reward else None
    )

    self.tokenizer = tokenizer_adapter.TokenizerAdapter(tokenizer)
    self._rl_metrics_logger = metrics_logger.MetricsLogger(
        self.cluster_config.training_config.metrics_logging_options
    )
    self._buffered_train_metrics: list[MetricsBuffer] = []
    self._buffered_eval_metrics: list[MetricsBuffer] = []
    self._external_metrics_logger = None

    self._init_cluster()
    gc.collect()

    # NB: global steps should be adjusted properly based on the actual RL
    # algorithm. E.g. when loading from a checkpoint with additional inner loops
    # that update the model, we should properly update the global steps.
    self.global_steps = 0

  def _init_backbone_sharing_map(
      self,
      actor: ModelOrPath,
      reference: ModelOrPath | None = None,
  ):
    # [FUNCTION DOC]
    # HELPER: Creates a defaultdict(list) sharing map. If actor and rollout meshes
    # are equal, marks ACTOR and ROLLOUT as sharing. Returns early if actor (or
    # a given reference) is not an nnx module. If the actor has LoRA enabled and
    # a reference is given on the same mesh as the actor, marks ACTOR and
    # REFERENCE as sharing (they share the base weights). Then propagates so
    # every role sharing with ACTOR also shares with the others.
    """Initializes the backbone sharing map."""
    self._backbone_sharing_map: dict[Role, list[Role]] = (
        collections.defaultdict(list)
    )

    if self.r2m[Role.ACTOR] == self.r2m[Role.ROLLOUT]:
      # Given that we load both actor trainer and rollout from `actor`,
      # if the meshes are the same, they are able to share the same model.
      # TODO(linchai): We may want to enable different shardings for actor
      # trainer and rollout even when they are colocated.
      self._backbone_sharing_map[Role.ACTOR].append(Role.ROLLOUT)
      self._backbone_sharing_map[Role.ROLLOUT].append(Role.ACTOR)

    # TODO(linchai): support loadding model from path and backbone sharing for
    # such case.
    if not isinstance(actor, nnx.Module) or (
        reference and not isinstance(reference, nnx.Module)
    ):
      return
    if sft_utils.is_lora_enabled(actor):
      if reference and self.r2m[Role.ACTOR] == self.r2m[Role.REFERENCE]:
        self._backbone_sharing_map[Role.ACTOR].append(Role.REFERENCE)
        self._backbone_sharing_map[Role.REFERENCE].append(Role.ACTOR)
      # TODO(linchai): maybe support critic backbone sharing.

    self._propagate_backbone_sharing_map()

  def _load_model(
      self,
      model_or_path: ModelOrPath,
      mesh: Mesh,
      data_type: jnp.dtype | None = None,
  ) -> nnx.Module:
    # [FUNCTION DOC]
    # HELPER: Inputs: model (must be an nnx module, else NotImplementedError),
    # mesh, optional data_type. Determines the model's current mesh and whether
    # any parameter is in default (device) memory. If the target mesh is
    # non-empty and differs, builds NamedShardings on the target mesh from the
    # variables' partition specs (empty spec when None), keeping device memory
    # or pinned host, optionally casts to data_type (if different from the first
    # leaf's dtype), reshards and merges a new model. Then, if on device and
    # offload_to_cpu is set, moves parameters to pinned host. Returns the model.
    # KNOWN ISSUE: data_type casting only happens when resharding is needed.
    """Loads model with given mesh to the given memory_kind.

    If input is already an NNX model, check if the model is sharded on the
    target mesh. If not, reshard the model.

    Args:
      model_or_path: either a nnx.Module or a path to a model.
      mesh: the mesh to load the model on.
      data_type: optional data type to cast the model parameters to.

    Returns:
      The model loaded on the given mesh.
    """
    if isinstance(model_or_path, nnx.Module):
      model_mesh = rl_utils.get_pytree_mesh_info(nnx.state(model_or_path))
      original_shardings = jax.tree_util.tree_map(
          lambda x: x.sharding, nnx.state(model_or_path)
      )
      is_on_device = jax.tree_util.tree_reduce(
          operator.or_,
          jax.tree.map(
              lambda x: x.memory_kind == self._default_memory_kind,
              original_shardings,
          ),
      )
      if not mesh.empty and model_mesh != mesh:
        logging.warning("Resharding model from %s to %s", model_mesh, mesh)
        graph, state = nnx.split(model_or_path)
        dst_shardings = jax.tree_util.tree_map(
            lambda x: jax.sharding.NamedSharding(
                mesh,
                x if x is not None else jax.sharding.PartitionSpec(),
                memory_kind=self._default_memory_kind
                if is_on_device
                else "pinned_host",
            ),
            nnx.get_partition_spec(state),
        )
        if data_type and data_type != jax.tree.leaves(state)[0].dtype:
          tmp_state = jax.tree.map(lambda x: x.astype(data_type), state)
        else:
          tmp_state = state
        model_or_path = nnx.merge(
            graph, reshard.reshard_pytree(tmp_state, dst_shardings)
        )
        del tmp_state
        gc.collect()
      if is_on_device and self.cluster_config.offload_to_cpu:
        graph, state = nnx.split(model_or_path)
        new_params = rl_utils.put_params_on_memory_kind(state, "pinned_host")
        model_or_path = nnx.merge(graph, new_params)
      return model_or_path
    else:
      raise NotImplementedError("Loading from path is not supported yet.")

  def _init_cluster(self):
    # [FUNCTION DOC]
    # HELPER: Builds services. 1) Validates a string rollout_engine is vanilla,
    # vllm or sglang_jax (ValueError). KV cache size is the max over TRAIN and
    # EVAL configs. vanilla: requires the rollout actor to have a config
    # (ValueError), loads it from CPU if offloading, creates VanillaRollout with
    # a CacheConfig from the model config (layers, kv heads, head dim), then
    # offloads. vllm: uses the TRAIN config, creates VllmRollout within the
    # rollout mesh / logical-axis context with max_model_len = KV cache size.
    # sglang_jax: raises ValueError if the actor uses LoRA without static LoRA
    # enabled; creates SglangJaxRollout. Custom class or partial of a
    # BaseRollout subclass: called with rollout_actor, tokenizer, mesh and the
    # TRAIN config. Anything else raises NotImplementedError. If the engine
    # exposes a non-None mesh, it replaces the rollout mesh. 2) Perf tracers are
    # no-op unless perf_config provides v1/v2 export functions, in which case
    # tracers over all devices of all meshes are created (v2 treats rollout and
    # environment spans as concurrent). 3) Creates the InferenceWorker with the
    # available critic, reference and reward models (deleting the reference and
    # reward attributes). 4) If a critic exists and does not share the actor
    # backbone, creates the critic Trainer (deep-copied config with prefix
    # "critic", checkpoint subdirectory "critic", critic optimizer, metadata
    # global_step + 1 and role) and offloads it; deletes the critic attribute.
    # 5) Creates the actor Trainer likewise (prefix/subdirectory "actor"), deletes
    # rollout_actor and train_actor attributes, offloads the actor if enabled,
    # and snapshots the actor state to pinned host as the anchor policy state.
    # KNOWN ISSUE: the vLLM check for a None model version can never trigger
    # because the default is an empty string.
    """Initializes the RL cluster."""
    # 1. Initialize rollout.
    if isinstance(
        self.cluster_config.rollout_engine, str
    ) and self.cluster_config.rollout_engine not in [
        "vanilla",
        "vllm",
        "sglang_jax",
    ]:
      raise ValueError(
          "`cluster_config.rollout_engine` should be one of `'vanilla'`, "
          "`'vllm'`, or `'sglang_jax'`. Received:"
          f" '{self.cluster_config.rollout_engine}'."
      )

    if isinstance(self.cluster_config.rollout_config, dict):
      # train_cfg should always be provided.
      train_cfg = self.cluster_config.rollout_config[Mode.TRAIN]
      eval_cfg = self.cluster_config.rollout_config.get(Mode.EVAL)
      max_kv_cache_size = max(
          train_cfg.kv_cache_size,
          eval_cfg.kv_cache_size if eval_cfg is not None else 0,
      )
    else:
      max_kv_cache_size = self.cluster_config.rollout_config.kv_cache_size

    if self.cluster_config.rollout_engine == "vanilla":
      if not hasattr(self.rollout_actor, "config"):
        raise ValueError("`self.rollout_actor` must have a config attribute.")
      # We must load the model from CPU before initializing the rollout,
      # otherwise the prefill and decode programs might be initialized on CPU.
      self._maybe_load_model_from_cpu(self.rollout_actor, Role.ROLLOUT)
      self._rollout = vanilla_rollout.VanillaRollout(
          self.rollout_actor,
          self.tokenizer,
          cache_config_or_size=base_rollout.CacheConfig(
              cache_size=max_kv_cache_size,
              num_layers=self.rollout_actor.config.num_layers,
              num_kv_heads=self.rollout_actor.config.num_kv_heads,
              head_dim=self.rollout_actor.config.head_dim,
          ),
      )
      self._maybe_offload_model_to_cpu(self._rollout.model(), Role.ROLLOUT)
    elif self.cluster_config.rollout_engine == "vllm":
      from tunix.rl.rollout import vllm_rollout

      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_vllm_config = self.cluster_config.rollout_config[Mode.TRAIN]
      else:
        loaded_vllm_config = self.cluster_config.rollout_config

      if loaded_vllm_config.rollout_vllm_model_version is None:
        raise ValueError("Rollout vllm model version or path is missing!")

      # TODO(linchai): maybe support offloading for vllm rollout.
      with self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT):
        # vLLM handles model initialization and loading internally, so we need
        # to provide logical axis rules for vLLM to correctly shard the model on
        # the rollout mesh. This is important for out-of-tree models in vLLM
        # that are implemented with custom logical axis rules, like is the case
        # for MaxText models.
        self._rollout = vllm_rollout.VllmRollout(
            self.rollout_actor,
            self.tokenizer,
            cache_config_or_size=max_kv_cache_size,
            mesh=self.r2m[Role.ROLLOUT],
            rollout_config=loaded_vllm_config,
        )
    elif self.cluster_config.rollout_engine == "sglang_jax":
      from tunix.rl.rollout import sglang_jax_rollout

      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_sglang_jax_config = self.cluster_config.rollout_config[
            Mode.TRAIN
        ]
      else:
        loaded_sglang_jax_config = self.cluster_config.rollout_config

      if (
          sft_utils.is_lora_enabled(self.rollout_actor)
          and not loaded_sglang_jax_config.rollout_sglang_jax_enable_static_lora
      ):
        raise ValueError(
            "Rollout sglang jax lora config is missing: must set"
            " rollout_sglang_jax_lora_target_modules,"
            " rollout_sglang_jax_enable_static_lora,"
            " rollout_sglang_jax_max_lora_rank,"
            " rollout_sglang_jax_lora_scaling."
        )

      self._rollout = sglang_jax_rollout.SglangJaxRollout(
          self.rollout_actor,
          self.tokenizer,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=loaded_sglang_jax_config,
      )
    elif (
        isinstance(self.cluster_config.rollout_engine, type)
        and issubclass(
            self.cluster_config.rollout_engine, base_rollout.BaseRollout
        )
    ) or (
        isinstance(self.cluster_config.rollout_engine, functools.partial)
        and issubclass(
            self.cluster_config.rollout_engine.func,
            base_rollout.BaseRollout,
        )
    ):
      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_config = self.cluster_config.rollout_config[Mode.TRAIN]
      else:
        loaded_config = self.cluster_config.rollout_config

      self._rollout = self.cluster_config.rollout_engine(
          rollout_actor=self.rollout_actor,
          tokenizer=self.tokenizer,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=loaded_config,
      )
    else:
      raise NotImplementedError(
          f"Rollout engine {self.cluster_config.rollout_engine} not supported"
      )

    # If the rollout engine constructs its own mesh, it could potentially
    # rearanges the devices for better performance. Use that mesh instead of the
    # one provided in the cluster config.
    if hasattr(self._rollout, "mesh") and self._rollout.mesh is not None:
      self.r2m[Role.ROLLOUT] = self._rollout.mesh

    # Initialize the performance tracer after we have all the meshes
    self._perf = perf_trace.NoopTracer()
    self._perf_v2 = perf_tracer_v2.NoopTracer()

    if self.perf_config:
      export_fn_v1 = self.perf_config.custom_export_fn
      export_fn_v2 = self.perf_config.custom_export_fn_v2

      if export_fn_v1 or export_fn_v2:
        devices = list(
            itertools.chain.from_iterable(
                mesh.devices.flatten().tolist()
                for mesh in self.cluster_config.role_to_mesh.values()
            )
        )

        if export_fn_v1:
          self._perf = perf_trace.PerfTracer(devices, export_fn_v1)

        if export_fn_v2:
          self._perf_v2 = perf_tracer_v2.PerfTracer(
              devices,
              export_fn=export_fn_v2,
              concurrent_device_spans=[
                  perf_constants.ROLLOUT,
                  perf_constants.ENVIRONMENT,
              ],
          )

    # 2. Initialize inference worker.
    inference_models = {}
    if self.critic is not None:
      inference_models["critic"] = self.critic
    if self.reference is not None:
      inference_models["reference"] = self.reference
      del self.reference
    if self.reward is not None:
      inference_models["reward"] = self.reward
      del self.reward
    self._inference_worker = inference_worker.InferenceWorker(inference_models)

    # 3. Initialize trainer.
    if (
        self.critic
        and Role.CRITIC not in self._backbone_sharing_map[Role.ACTOR]
    ):
      critic_config = copy.deepcopy(self.cluster_config.training_config)
      critic_config.metrics_prefix = "critic"
      critic_config.pbar_description = "Critic Training"
      if critic_config.checkpoint_root_directory is not None:
        critic_config.checkpoint_root_directory = os.path.join(
            critic_config.checkpoint_root_directory, "critic"
        )
      with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
        self._critic_trainer = rl_trainer.Trainer(
            model=self.critic,
            optimizer=self.cluster_config.training_config.critic_optimizer,
            training_config=critic_config,
            custom_checkpoint_metadata_fn=lambda: {
                "global_step": self.global_steps + 1,
                "role": Role.CRITIC.value,
            },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-long
            metrics_logger=self._rl_metrics_logger,
            perf_tracer=self._perf,
            perf_tracer_v2=self._perf_v2,
        )
      del self.critic
      self._maybe_offload_model_to_cpu(self._critic_trainer.model, Role.CRITIC)

    self._maybe_load_model_from_cpu(self.train_actor, Role.ACTOR)
    actor_config = copy.deepcopy(self.cluster_config.training_config)
    actor_config.metrics_prefix = "actor"
    actor_config.pbar_description = "Actor Training"
    if actor_config.checkpoint_root_directory is not None:
      actor_config.checkpoint_root_directory = os.path.join(
          actor_config.checkpoint_root_directory, "actor"
      )
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
      self._actor_trainer = rl_trainer.Trainer(
          model=self.train_actor,
          optimizer=self.cluster_config.training_config.actor_optimizer,
          training_config=actor_config,
          custom_checkpoint_metadata_fn=lambda: {
              "global_step": self.global_steps + 1,
              "role": Role.ACTOR.value,
          },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-long
          metrics_logger=self._rl_metrics_logger,
          perf_tracer=self._perf,
          perf_tracer_v2=self._perf_v2,
      )
    del self.rollout_actor
    del self.train_actor
    self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
    self._anchor_policy_state = rl_utils.put_params_on_memory_kind(
        nnx.state(self.actor_trainer.model), "pinned_host"
    )

  def _propagate_backbone_sharing_map(self):
    # [FUNCTION DOC]
    # HELPER: For every pair of distinct roles that both share with ACTOR, adds
    # each to the other's sharing list (may add duplicates if called twice).
    """Propagates backbone sharing map."""
    for role in self._backbone_sharing_map[Role.ACTOR]:
      for other_role in self._backbone_sharing_map[Role.ACTOR]:
        if other_role != role:
          self._backbone_sharing_map[role].append(other_role)

  def _put_model_on_memory_kind(self, model: nnx.Module, memory_kind: str):
    # [FUNCTION DOC]
    # HELPER: Moves all of the model's variables to memory_kind ("pinned_host" or
    # "device"; ValueError otherwise) via put_params_on_memory_kind and updates
    # the model in place.
    """Puts model on the given memory kind."""
    if memory_kind not in ["pinned_host", "device"]:
      raise ValueError(f"Unsupported memory kind. Received: {memory_kind}")
    original_variables = nnx.variables(model)
    new_variables = rl_utils.put_params_on_memory_kind(
        original_variables, memory_kind
    )
    nnx.update(model, new_variables)

  def _update_models_sharing_weights(
      self,
      params: jaxtyping.PyTree,
      role: Role,
  ):
    # [FUNCTION DOC]
    # HELPER: Propagates params of the model playing role to every role sharing
    # weights with it: ROLLOUT gets an nnx update of rollout_actor if it still
    # exists, else rollout.update_params; REFERENCE gets the non-LoRA part of the
    # params (from the attribute or the inference worker model); ACTOR gets an
    # nnx update of the train actor or trainer model.
    """Updates models sharing weights."""
    for role in self._backbone_sharing_map[role]:
      if role == Role.ROLLOUT:
        if hasattr(self, "rollout_actor"):
          nnx.update(self.rollout_actor, params)
        else:
          self.rollout.update_params(params)
      elif role == Role.REFERENCE:
        ref_model = (
            self.reference
            if hasattr(self, "reference")
            else self.inference_worker.get_model("reference")
        )
        if ref_model:
          nnx.update(
              ref_model,
              statelib.filter_state(params, filterlib.Not(nnx.LoRAParam)),
          )
      elif role == Role.ACTOR:
        actor_model = (
            self.train_actor
            if hasattr(self, "train_actor")
            else self.actor_trainer.model
        )
        nnx.update(actor_model, params)

  def _is_state_on_device(self, state: jaxtyping.PyTree) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns true if any leaf of the state has a sharding whose memory
    # kind is the default (device) memory kind; false for empty states.
    shardings = jax.tree.map(
        lambda x: x.sharding if hasattr(x, "sharding") else None, state
    )
    return jax.tree_util.tree_reduce(
        operator.or_,
        jax.tree.map(
            lambda x: x is not None
            and x.memory_kind == self._default_memory_kind,
            shardings,
        ),
        initializer=False,
    )

  def _maybe_load_model_from_cpu(self, model: nnx.Module, role: Role):
    # [FUNCTION DOC]
    # HELPER: If offload_to_cpu is enabled, moves the model to device memory and
    # pushes its state to roles sharing weights; otherwise does nothing.
    """Loads model from CPU if needed."""
    if not self.cluster_config.offload_to_cpu:
      return
    self._put_model_on_memory_kind(model, "device")
    self._update_models_sharing_weights(nnx.state(model), role)

  def _maybe_offload_model_to_cpu(self, model: nnx.Module, role: Role):
    # [FUNCTION DOC]
    # HELPER: If offload_to_cpu is enabled, moves the model to pinned host memory
    # and pushes its state to roles sharing weights; otherwise does nothing.
    """Offloads model to CPU if needed."""
    if not self.cluster_config.offload_to_cpu:
      return
    self._put_model_on_memory_kind(model, "pinned_host")
    self._update_models_sharing_weights(nnx.state(model), role)

  @property
  def rollout(self) -> base_rollout.BaseRollout:
    # [FUNCTION DOC]
    # Property: the rollout engine.
    return self._rollout

  @property
  def inference_worker(self) -> inference_worker.InferenceWorker:
    # [FUNCTION DOC]
    # Property: the inference worker holding critic/reference/reward models.
    return self._inference_worker

  @property
  def actor_trainer(self) -> rl_trainer.Trainer:
    # [FUNCTION DOC]
    # Property: the actor Trainer.
    return self._actor_trainer

  @property
  def critic_trainer(self) -> rl_trainer.Trainer:
    # [FUNCTION DOC]
    # Property: the critic Trainer; raises AttributeError if none was created.
    return self._critic_trainer

  @property
  def perf(self) -> perf_trace.Tracer:
    # [FUNCTION DOC]
    # Property: the v1 perf tracer (no-op by default).
    """The v1 performance tracer."""
    return self._perf

  @property
  def perf_v2(self) -> perf_tracer_v2.Tracer:
    # [FUNCTION DOC]
    # Property: the v2 perf tracer (no-op by default).
    """The v2 performance tracer."""
    return self._perf_v2

  def close(self):
    # [FUNCTION DOC]
    # Logs all remaining buffered train and eval metrics, then closes the actor
    # trainer and, if present, the critic trainer.
    for m in self._buffered_train_metrics + self._buffered_eval_metrics:
      self._log_metrics(m)
    self.actor_trainer.close()
    if getattr(self, "critic_trainer", None):
      self.critic_trainer.close()

  def _log_metrics(self, metrics_buffer: MetricsBuffer) -> None:
    # [FUNCTION DOC]
    # HELPER: For each metric (name to pair of value list and aggregation):
    # converts values to a NumPy array (skips with a warning if that fails);
    # string arrays, or object arrays whose first element is a string, are
    # logged as info text and skipped; otherwise applies the aggregation when
    # present and the array is non-empty. Splits the name at the first "/" into
    # prefix and name (prefix "global" if none) and logs via the metrics logger
    # with the buffer's mode and global step. Finally passes the whole buffer to
    # the external metrics logger if set.
    """Log metrics."""
    for metric_name, (value, op) in metrics_buffer.metrics.items():
      # Convert to numpy array immediately.
      # This handles nested lists, mixed types, and JAX arrays automatically.
      try:
        agg_value = np.array(value)
      except Exception:
        logging.warning(
            "Skipping metric %s: Could not convert to numpy array.", metric_name
        )
        continue

      if agg_value.dtype.kind in {"U", "S"}:
        logging.info(
            "Rollout string metric %s: %s",
            metric_name,
            agg_value,
        )
        continue

      if agg_value.dtype.kind == "O":
        # Try to infer if it contains strings by checking the first flattened element
        if agg_value.size > 0 and isinstance(
            agg_value.ravel()[0], (str, np.str_)
        ):
          logging.info("Rollout string metric %s: %s", metric_name, agg_value)
          continue

      # Apply aggregation and Log
      if op is not None:
        # Ensure op doesn't crash on empty arrays
        if agg_value.size > 0:
          agg_value = op(agg_value)

      if "/" in metric_name:
        prefix, metric_name = metric_name.split("/", maxsplit=1)
      else:
        prefix = "global"
      self._rl_metrics_logger.log(
          prefix,
          metric_name,
          agg_value,
          metrics_buffer.mode,
          metrics_buffer.global_steps,
      )

    if self._external_metrics_logger is not None:
      self._external_metrics_logger(metrics_buffer)

  def with_external_metrics_logger(
      self, external_metrics_logger: Callable[[MetricsBuffer], None]
  ):
    # [FUNCTION DOC]
    # Sets the external metrics logger callback (receives each MetricsBuffer
    # after logging) and returns self for chaining.
    self._external_metrics_logger = external_metrics_logger
    return self

  def buffer_metrics(
      self,
      metrics: MetricsT,
      mode: Mode = Mode.TRAIN,
  ) -> None:
    # [FUNCTION DOC]
    # Inputs: metrics dict (name to pair of value and aggregation or None),
    # mode. Selects the train or eval buffer list; creates a buffer for the
    # current global step if the list is empty. If the oldest train buffer's
    # step differs from the current global step, appends a new train buffer for
    # the current step and logs (and removes) the oldest train buffer and the
    # oldest eval buffer if any. Then appends each value to the latest buffer of
    # the selected list (creating the entry with the aggregation on first use).
    # KNOWN ISSUE: in EVAL mode with no train buffer yet, indexing the train list
    # raises IndexError; the new train buffer created on step change is labeled
    # with the caller's mode (may be "eval"); and eval values are appended to
    # the latest eval buffer, which may belong to an older step.
    """Buffers rl metrics to be logged.

    Actual logging will happen when global steps are incremented.

    Args:
      metrics: A dictionary mapping metric names to a tuple containing the
        metric value and an optional aggregation function.
      mode: The mode of the workload, either TRAIN or EVAL.
    """
    if mode == Mode.TRAIN:
      buffered_metrics = self._buffered_train_metrics
    else:
      buffered_metrics = self._buffered_eval_metrics

    if not buffered_metrics:
      buffered_metrics.append(MetricsBuffer(self.global_steps, mode=str(mode)))

    # Global steps are incremented, log the previous metrics.
    if self._buffered_train_metrics[0].global_steps != self.global_steps:
      self._buffered_train_metrics.append(
          MetricsBuffer(self.global_steps, mode=str(mode))
      )
      for m in [self._buffered_train_metrics.pop(0)] + (
          [self._buffered_eval_metrics.pop(0)]
          if self._buffered_eval_metrics
          else []
      ):
        self._log_metrics(m)

    cur_metrics = buffered_metrics[-1]
    for metric_name, (value, op) in metrics.items():
      if metric_name not in cur_metrics.metrics:
        cur_metrics.metrics[metric_name] = (
            [value],
            op,
        )
      else:
        cur_metrics.metrics[metric_name][0].append(value)

  def buffer_metrics_async(
      self,
      metrics: MetricsT,
      mode: Mode = Mode.TRAIN,
      step: int = 0,
  ) -> None:
    # [FUNCTION DOC]
    # Like buffer_metrics but keyed by an explicit step (for async producers):
    # creates a buffer for the current global step if the list is empty, or a new
    # buffer for step if it differs from the latest buffer's step; appends the
    # values; then logs and removes the oldest train buffer and the oldest eval
    # buffer if their step is below the current global step (at most one each
    # per call).
    """Buffers rl metrics to be logged for async training.

    Actual logging will happen when global steps are incremented.

    Args:
      metrics: A dictionary mapping metric names to a tuple containing the
        metric value and an optional aggregation function.
      mode: The mode of the workload, either TRAIN or EVAL.
      step: The step number for the metrics. Only used in TRAIN mode.
    """
    if mode == Mode.TRAIN:
      buffered_metrics = self._buffered_train_metrics
    else:
      buffered_metrics = self._buffered_eval_metrics

    if not buffered_metrics:
      buffered_metrics.append(MetricsBuffer(self.global_steps, mode=str(mode)))
    else:
      if step != buffered_metrics[-1].global_steps:
        buffered_metrics.append(MetricsBuffer(step, mode=str(mode)))

    cur_metrics = buffered_metrics[-1]
    for metric_name, (value, op) in metrics.items():
      if metric_name not in cur_metrics.metrics:
        cur_metrics.metrics[metric_name] = (
            [value],
            op,
        )
      else:
        cur_metrics.metrics[metric_name][0].append(value)

    # Global steps are incremented, log the previous metrics.
    if (
        self._buffered_train_metrics
        and self._buffered_train_metrics[0].global_steps < self.global_steps
    ):
      for m in [self._buffered_train_metrics.pop(0)]:
        self._log_metrics(m)
    if (
        self._buffered_eval_metrics
        and self._buffered_eval_metrics[0].global_steps < self.global_steps
    ):
      for m in [self._buffered_eval_metrics.pop(0)]:
        self._log_metrics(m)

  def update_actor(self, train_ds, eval_ds, skip_jit=False):
    # [FUNCTION DOC]
    # Within the actor mesh/axis-rule context: loads the actor from CPU if
    # offloading, trains the actor trainer on train_ds with optional eval_ds
    # (inside a perf span group), and offloads again. Returns None.
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
      self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
      with self._perf.span_group("actor_training"):
        self.actor_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)

  def update_critic(self, train_ds, eval_ds, skip_jit=False):
    # [FUNCTION DOC]
    # Same as update_actor for the critic trainer within the critic mesh context.
    with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
      self._maybe_load_model_from_cpu(self.critic_trainer.model, Role.CRITIC)
      with self._perf.span_group("critic_training"):
        self._critic_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.critic_trainer.model, Role.CRITIC)

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Mode = Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # Inputs: prompts (strings, or chat conversations if apply_chat_template),
    # apply_chat_template flag, mode, optional micro_batch_size, optional perf
    # trace tags, optional max_generation_steps override. With chat template,
    # renders each conversation via the tokenizer with a generation prompt and
    # thinking disabled (ValueError if no tokenizer). Raises ValueError for an
    # empty prompt list. Within the rollout mesh context: loads the rollout
    # model from CPU if offloading (and pushes its state into the engine),
    # selects the rollout config for mode (optionally replacing
    # max_tokens_to_generate), and calls rollout.generate on consecutive chunks
    # of micro_batch_size prompts inside perf spans tagged with role "rollout";
    # then offloads again. Concatenates per-chunk text, logprobs and logits
    # (None if the first chunk has none), tokens, and left-padded prompt tokens
    # (concatenated along batch; requires equal widths). Returns a
    # RolloutOutput.
    """Generates text from the given prompts.

    Args:
      prompts: A list of prompts to generate text from. If `apply_chat_template`
        is True, this should be a list of conversations (each a list of
        dictionaries with 'role' and 'content'). Otherwise, it should be a list
        of strings.
      apply_chat_template: Whether to apply chat template to the prompts.
      mode: The mode of rollout, either TRAIN or EVAL.
      micro_batch_size: The micro-batch size for generation. If None, no
        micro-batching is performed.
      trace_tags: Optional tags to add to the performance tracer.

    Returns:
      A `RolloutOutput` object containing the generated text and other info.
    """
    if apply_chat_template:
      if self.tokenizer is None:
        raise ValueError("Tokenizer must be initialized to use chat templates.")
      string_prompts = [
          self.tokenizer.apply_chat_template(
              prompt,  # pytype: disable=wrong-arg-types
              add_generation_prompt=True,
              tokenize=False,
              enable_thinking=False,
          )
          for prompt in prompts
      ]
    else:
      string_prompts = prompts  # pytype: disable=annotation-type-mismatch

    if len(string_prompts) == 0:
      raise ValueError("Cannot generate from an empty list of prompts.")
    micro_batch_size = micro_batch_size or len(string_prompts)

    with self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT) as (mesh, _):
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))

      if isinstance(self.cluster_config.rollout_config, dict):
        rollout_config = self.cluster_config.rollout_config[mode]
      else:
        rollout_config = self.cluster_config.rollout_config

      if max_generation_steps is not None:
        rollout_config = dataclasses.replace(
            rollout_config,
            max_tokens_to_generate=max_generation_steps,
        )

      perf_tags = {
          perf_constants.ROLE: Role.ROLLOUT.value,
      }
      if trace_tags:
        perf_tags.update(trace_tags)

      with self._perf.span("rollout", mesh.devices) as span, self._perf_v2.span(
          perf_constants.ROLLOUT,
          mesh.devices,
          tags=perf_tags,
      ) as span_v2:
        outputs = [
            self.rollout.generate(string_prompts[s], rollout_config)
            for s in rl_utils.chunk_slices_by_size(
                stop=len(string_prompts), step=micro_batch_size
            )
        ]
        span.device_end([o.tokens for o in outputs])
        span_v2.async_end([o.tokens for o in outputs])
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))

    texts = list(itertools.chain.from_iterable(out.text for out in outputs))

    logprobs = None
    if outputs[0].logprobs is not None:
      logprobs = list(
          itertools.chain.from_iterable(out.logprobs for out in outputs)
      )

    logits = None
    if outputs[0].logits is not None:
      logits = list(
          itertools.chain.from_iterable(out.logits for out in outputs)
      )

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits,
        tokens=list(
            itertools.chain.from_iterable(out.tokens for out in outputs)
        ),
        left_padded_prompt_tokens=np.concatenate(
            [out.left_padded_prompt_tokens for out in outputs], axis=0
        ),
        logprobs=logprobs,
    )

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt and completion token arrays, pad and eos ids, optional micro
    # batch size. Raises ValueError for an empty batch. Within the reference
    # mesh context, shards inputs along the training data_sharding_axis, loads
    # the reference model from CPU if needed, and computes reference per-token
    # logps chunk by chunk via the inference worker using the TRAIN rollout
    # temperature; concatenates along batch, offloads, and returns the array
    # (batch by completion length).
    """Gets the per-token logps of the reference model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError(
          "Cannot get reference log probabilities from an empty batch."
      )
    micro_batch_size = micro_batch_size or batch_size

    with self._get_mesh_and_logical_axis_rules_cm(Role.REFERENCE):
      # This assumes reference model shards same data sharding as actor, which
      # should be true as ref model and policy model shares same architecture.
      dest_prompt_tokens = sharding_utils.shard_input(
          prompt_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_completion_tokens = sharding_utils.shard_input(
          completion_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      self._maybe_load_model_from_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      temperature = self.get_rollout_config(mode=Mode.TRAIN).temperature
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            self.inference_worker.get_ref_per_token_logps(
                dest_prompt_tokens[batch_slice],
                dest_completion_tokens[batch_slice],
                pad_id,
                eos_id,
                temperature=temperature,
            )
        )
      ref_per_token_logps = jnp.concatenate(outs, axis=0)
      self._maybe_offload_model_to_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      return ref_per_token_logps

  def get_old_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      micro_batch_size: int | None = None,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt and completion tokens, optional micro batch size. Raises
    # ValueError for an empty batch. Within the rollout mesh context, loads the
    # rollout model from CPU if offloading (syncing the engine), computes
    # per-token logps chunk by chunk with rollout.get_per_token_logps,
    # concatenates, offloads again, and returns them. These are the "old"
    # policy logps from the rollout copy of the model.
    """Gets the per-token logps of the current policy model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError("Cannot get old log probabilities from an empty batch.")
    micro_batch_size = micro_batch_size or batch_size

    with self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT):
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            self.rollout.get_per_token_logps(
                prompt_tokens[batch_slice],
                completion_tokens[batch_slice],
            )
        )
      per_token_logps = jnp.concatenate(outs, axis=0)
      model = self.rollout.model()
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      return per_token_logps

  def get_actor_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      temperature: float | None = None,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt and completion tokens, pad/eos ids, optional micro batch
    # size, optional temperature (defaults to the TRAIN rollout temperature).
    # Raises ValueError for an empty batch or when no anchor policy state exists.
    # Within the actor mesh context, shards inputs, temporarily offloads the
    # live actor weights to host if they are on device and offloading is
    # enabled, moves the anchor (last-synced) state to device memory if needed,
    # computes stop-gradient per-token logps chunk by chunk with the actor
    # graph and the anchor state at the given temperature, frees the temporary
    # copy, restores the actor to device memory if it was moved, and returns
    # the concatenated logps.
    """Gets per-token logps from the actor model on the trainer side.

    Mirrors `get_ref_per_token_logps` — must pass through the rollout temperature
    so the actor's recomputed logps match the temperature scaling used at
    sampling time (otherwise log_softmax(logits/T_sample) vs log_softmax(logits)
    yields a multi-nat artifact diff vs vllm's `processed_logprobs`).
    """
    if temperature is None:
      temperature = self.get_rollout_config(mode=Mode.TRAIN).temperature
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError(
          "Cannot get actor log probabilities from an empty batch."
      )
    if self._anchor_policy_state is None:
      raise ValueError(
          "Anchor policy state is not initialized. Please run `sync_weights`"
          " first."
      )
    micro_batch_size = micro_batch_size or batch_size
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR) as (mesh, _):
      dest_prompt_tokens = sharding_utils.shard_input(
          prompt_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_completion_tokens = sharding_utils.shard_input(
          completion_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )

      # Use the anchor (start-of-global-step) actor weights so old_per_token_logps
      # reference the same policy vllm sampled with even when mini_batch_size <
      # full_batch_size or num_iterations > 1. Only offload the live actor when
      # `offload_to_cpu` is enabled cluster-wide; otherwise the host round-trip
      # was both unnecessary and risked leaving stray weights pinned to host.
      actor_trainer_state_on_device = self._is_state_on_device(
          nnx.state(self.actor_trainer.model)
      )
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        self._put_model_on_memory_kind(self.actor_trainer.model, "pinned_host")
        gc.collect()
      graphdef, _ = nnx.split(self.actor_trainer.model)
      anchor_on_device = self._is_state_on_device(self._anchor_policy_state)
      if anchor_on_device:
        anchor_policy_state = self._anchor_policy_state
      else:
        anchor_policy_state = rl_utils.put_params_on_memory_kind(
            self._anchor_policy_state, self._default_memory_kind
        )
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            common.compute_per_token_logps(
                graphdef,
                anchor_policy_state,
                prompt_tokens=dest_prompt_tokens[batch_slice],
                completion_tokens=dest_completion_tokens[batch_slice],
                pad_id=pad_id,
                eos_id=eos_id,
                stop_gradient=True,
                return_logits=False,
                temperature=temperature,
            )
        )
      actor_per_token_logps = jnp.concatenate(outs, axis=0)
      if not anchor_on_device:
        del anchor_policy_state
      gc.collect()
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        self._put_model_on_memory_kind(
            self.actor_trainer.model, self._default_memory_kind
        )
      return actor_per_token_logps

  def sync_weights(self):
    # [FUNCTION DOC]
    # Pushes trainer weights to the rollout engine. On non-TPU/GPU backends
    # (e.g. CPU) wraps the operation in transfer guards disallowing explicit
    # host/device transfers; otherwise no guard. Filters actor state to LoRA
    # params if LoRA is enabled, else all nnx Params; calls
    # rollout.update_params with them and the filter type tuple; re-snapshots
    # the full actor state to pinned host as the anchor policy state.
    # Increments global_steps by one. Returns None.
    # KNOWN ISSUE: the transfer guards are only active on non-accelerator
    # backends, which looks inverted relative to their purpose.
    """Syncs the weights of between the sampler model and trainer model."""
    if jax.devices() and jax.default_backend() not in ["tpu", "gpu"]:
      cm = contextlib.ExitStack()
      cm.enter_context(jax.transfer_guard_device_to_host("disallow_explicit"))
      cm.enter_context(jax.transfer_guard_host_to_device("disallow_explicit"))
    else:
      cm = contextlib.nullcontext()
    with cm:
      filter_types = (
          nnx.LoRAParam
          if sft_utils.is_lora_enabled(self.actor_trainer.model)
          else nnx.Param,
      )
      src_filtered_params = nnx.state(self.actor_trainer.model, filter_types)
      self.rollout.update_params(src_filtered_params, filter_types)
      # The anchor policy state is snapshotted from actor_trainer.model.
      self._anchor_policy_state = rl_utils.put_params_on_memory_kind(
          nnx.state(self.actor_trainer.model), "pinned_host"
      )

    # sync weights marks the end of a full batch, so increment the global steps.
    self.global_steps += 1

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Within the critic mesh context returns inference_worker.get_values for the
    # tokens (per-token values over prompt+completion).
    with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
      return self.inference_worker.get_values(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
      )

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Within the reward mesh context returns inference_worker.get_rewards for the
    # tokens (per-token reward-model scores over prompt+completion).
    with self._get_mesh_and_logical_axis_rules_cm(Role.REWARD):
      return self.inference_worker.get_rewards(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
      )

  def get_rollout_config(self, mode: Mode) -> base_rollout.RolloutConfig:
    # [FUNCTION DOC]
    # Returns the rollout config for mode when rollout_config is a dict
    # (KeyError if missing), else the single config.
    """Returns the rollout config for the given mode."""
    if isinstance(self.cluster_config.rollout_config, dict):
      return self.cluster_config.rollout_config[mode]
    else:
      return self.cluster_config.rollout_config

  @contextlib.contextmanager
  def _get_mesh_and_logical_axis_rules_cm(self, role: Role):
    # [FUNCTION DOC]
    # HELPER: Generator-based context manager for role: enters the role's mesh
    # and, if logical axis rules exist for the role, the flax logical axis rules
    # context (else a null context). Yields a pair of (mesh, rules context
    # value). Raises KeyError if the role has no mesh.
    """Returns a context manager for the mesh and logical axis rules.

    This is used for models that uses logical sharding, so XLA can generate the
    correct graph based on physical mesh.

    Args:
      role: The role of the model (e.g., ACTOR, CRITIC, REFERENCE, etc.).
    """
    role_logical_axis_rule = self.cluster_config.role_to_logical_axis_rule
    logical_axis_rule_ctx = contextlib.nullcontext()
    if role_logical_axis_rule and role in role_logical_axis_rule:
      logical_axis_rule_ctx = nn_partitioning.axis_rules(
          role_logical_axis_rule[role]
      )
    with contextlib.ExitStack() as stack:
      yield (
          stack.enter_context(self.cluster_config.role_to_mesh[role]),
          stack.enter_context(logical_axis_rule_ctx),
      )
