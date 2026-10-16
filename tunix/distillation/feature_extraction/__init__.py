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
# Package tunix.distillation.feature_extraction: helpers to capture and
#   adapt intermediate activations of nnx models for feature distillation.
# Re-exports:
#   - avg_pool_array_to_target_shape (pooling): average-pool an array down
#     to a target shape.
#   - SowedModule, wrap_model_with_sowed_modules, unwrap_sowed_modules,
#     pop_sowed_intermediate_outputs (sowed_module): wrap selected sublayers
#     so their outputs are stored ("sown") as nnx intermediates, then
#     collect/remove them.
#   - ModelWithFeatureProjection, setup_models_with_feature_projection,
#     remove_feature_projection_from_models (projection): attach/detach a
#     learned linear projection mapping student features to teacher shape.
# Inputs/outputs/exceptions: none at import. Side effects: imports submodules.
"""Feature extraction utilities for distillation."""

# pylint: disable=g-importing-member

from tunix.distillation.feature_extraction.pooling import avg_pool_array_to_target_shape
from tunix.distillation.feature_extraction.projection import ModelWithFeatureProjection
from tunix.distillation.feature_extraction.projection import remove_feature_projection_from_models
from tunix.distillation.feature_extraction.projection import setup_models_with_feature_projection
from tunix.distillation.feature_extraction.sowed_module import pop_sowed_intermediate_outputs
from tunix.distillation.feature_extraction.sowed_module import SowedModule
from tunix.distillation.feature_extraction.sowed_module import unwrap_sowed_modules
from tunix.distillation.feature_extraction.sowed_module import wrap_model_with_sowed_modules
