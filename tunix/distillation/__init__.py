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
# [PACKAGE DOC]
# Package tunix.distillation: knowledge-distillation training for Tunix.
# Behavior: re-exports the public trainer API from distillation_trainer:
#   DistillationTrainer (a PeftTrainer subclass that trains a student model
#   against a frozen teacher via a pluggable strategy), TrainingConfig and
#   TrainingInput (input record that additionally carries teacher_output).
# Subpackages: strategies (loss strategies: logit, attention, feature pooling,
#   feature projection) and feature_extraction (sowing/pooling/projection
#   utilities used by the feature strategies).
# Inputs/outputs: none at import time. Exceptions: only import errors.
# Side effects: importing loads distillation_trainer (and transitively
#   tunix.sft.peft_trainer, flax, optax, jax).
"""Module containing the distillation trainers."""

# pylint: disable=g-importing-member

from tunix.distillation.distillation_trainer import DistillationTrainer
from tunix.distillation.distillation_trainer import TrainingConfig
from tunix.distillation.distillation_trainer import TrainingInput
