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
# Package tunix.distillation.strategies: distillation loss strategies.
# Behavior: re-exports BaseStrategy (abstract strategy contract) and the
#   ModelForwardCallable type alias from base_strategy, plus the concrete
#   strategies: LogitStrategy (KL on softened logits blended with task loss),
#   AttentionTransferStrategy and AttentionProjectionStrategy (match
#   intermediate attention maps), FeaturePoolingStrategy (match pooled
#   intermediate features) and FeatureProjectionStrategy (match features
#   after a learned projection of student features).
# Every strategy turns (student model, teacher output, inputs) into a scalar
#   loss (optionally with aux metrics) and may rewrite models before/after
#   training via pre_process_models / post_process_models.
# Inputs/outputs/exceptions: none at import. Side effects: imports submodules.
"""Module containing the distillation strategies."""

# pylint: disable=g-importing-member

from tunix.distillation.strategies.attention import AttentionProjectionStrategy
from tunix.distillation.strategies.attention import AttentionTransferStrategy
from tunix.distillation.strategies.base_strategy import BaseStrategy
from tunix.distillation.strategies.base_strategy import ModelForwardCallable
from tunix.distillation.strategies.feature_pooling import FeaturePoolingStrategy
from tunix.distillation.strategies.feature_projection import FeatureProjectionStrategy
from tunix.distillation.strategies.logit import LogitStrategy
