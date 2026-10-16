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
# Standalone script (no functions) that downloads Llama-3.2-1B-Instruct from Hugging Face into
# <system tempdir>/models/<repo id>, builds a 1 x N_devices JAX mesh with axes ("fsdp","tp") (Auto axis
# types), constructs the tunix Llama3 model from safetensors with the llama3p2_1b config, displays it,
# loads the HF tokenizer (pad_token_id forced to 0), chat-templatizes three prompts, and runs the tunix
# Sampler (KV cache size 256 with layer/head dims from config) for 128 generation steps with echo and
# no top-p, printing each output text followed by a line of 30 asterisks.
# Inputs: none (network access to HF required). Outputs: printed text. Side effects: network download,
# disk writes to tempdir, device memory use. Raises: any download/model-loading errors propagate.
# KNOWN ISSUE: the name sampler is rebound from the module to the Sampler instance; harmless here but
# the module can no longer be referenced afterwards.
"""Example of using tunix to load and run Llama3 models."""

import os
import tempfile

from flax import nnx
import jax
import transformers
from tunix.generate import sampler
from tunix.models.llama3 import model
from tunix.models.llama3 import params
from tunix.tests import test_common as tc


MODEL_VERSION = "meta-llama/Llama-3.2-1B-Instruct"

# Consider switch to tempfile after figuring out how it works
temp_dir = tempfile.gettempdir()
MODEL_CP_PATH = os.path.join(temp_dir, "models", MODEL_VERSION)

all_files = tc.download_from_huggingface(repo_id=MODEL_VERSION, model_path=MODEL_CP_PATH)

mesh = jax.make_mesh(
    (1, len(jax.devices())),
    ("fsdp", "tp"),
    axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
)
config = (
    model.ModelConfig.llama3p2_1b()
)  # pick corresponding config based on model version
llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(llama3)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer.pad_token_id = 0


inputs = tc.batch_templatize([
    "tell me about world war 2",
    "印度的首都在哪里",
    "tell me your name, respond in Chinese",
], tokenizer)

sampler = sampler.Sampler(
    llama3,
    tokenizer,
    sampler.CacheConfig(
        cache_size=256, num_layers=config.num_layers, num_kv_heads=config.num_kv_heads, head_dim=config.head_dim
    ),
)
out = sampler(inputs, max_generation_steps=128, echo=True, top_p=None)

for t in out.text:
  print(t)
  print("*" * 30)
