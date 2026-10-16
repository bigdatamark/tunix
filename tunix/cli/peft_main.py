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
# Command-line entry point for parameter-efficient fine-tuning (PEFT / LoRA / full-weight SFT).
# Defines an absl string flag "pathways_bns" (default None) holding the Pathways server address,
# a pipeline class that builds a mesh, model, tokenizer, optimizer and PeftTrainer from a parsed
# CLI config and trains on a translation dataset, a helper that switches JAX to the Pathways
# backend, and a main function run via absl app when executed as a script.
# Side effects at import: registers the absl flag.

"""Main entry point for PEFT training."""
from collections.abc import Callable
from typing import Any
from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli.utils import model as model_lib
from tunix.examples.data import translation_dataset as data_lib
from tunix.sft import peft_trainer
from tunix.sft import utils
from tunix.utils import mesh as mesh_lib

_PATHWAYS_BNS = flags.DEFINE_string(
    'pathways_bns', None, 'BNS address of the Pathways server.'
)


class PeftPipeline(config.HyperParameters):
  # [CLASS DOC]
  # PEFT training pipeline. Inherits config parsing (self.config dict, mesh parsing, optimizer
  # creation, training-config dict extraction) from the CLI HyperParameters base class; the
  # constructor is inherited (takes argv and override kwargs). Adds a single method that runs training.
  """Class for running the Peft trainer."""

  def run_peft_trainer(self):
    # [FUNCTION DOC]
    # Runs PEFT training end to end. Input: self.config with keys model_config, tokenizer_config,
    # optimizer_config, training_config, dataset_name, batch_size, max_target_length,
    # num_train_epochs, tfds_download.
    # Logic: parses mesh axis shapes/names from model_config and creates a JAX mesh. Creates the
    # model (and resolved tokenizer path) via the CLI model utility; raises ValueError if the model
    # is None. Creates the tokenizer from tokenizer_config and the resolved path. Creates the optimizer
    # from optimizer_config. Constructs a PeftTrainer with a TrainingConfig built from the
    # training_config dict, and installs an input function that derives padding mask (token id not
    # equal 0), positions and a causal attention mask from the batch tokens. Builds train/eval
    # datasets via the translation dataset loader and trains inside the mesh context.
    # Output: none. Exceptions: ValueError for a missing model; config KeyErrors and training errors
    # propagate. Side effects: model download/loading, device allocation, training, checkpointing/logging
    # done by the trainer.
    """Run the PEFT trainer."""
    axis_shapes, axis_names = self.parse_mesh_config('model_config')
    mesh: jax.sharding.Mesh = mesh_lib.create_mesh(axis_shapes, axis_names)
    model: nnx.Module | None = None
    tokenizer: Any | None = None
    my_gen_model_input_fn: (
        Callable[[peft_trainer.TrainingInput], dict[str, Any]] | None
    ) = None
    model, tokenizer_path = model_lib.create_model(
        self.config['model_config'], self.config['tokenizer_config'], mesh
    )
    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
        self.config['tokenizer_config'], tokenizer_path
    )
    optimizer = self.create_optimizer('optimizer_config')
    trainer = peft_trainer.PeftTrainer(
        model,
        optimizer,
        peft_trainer.TrainingConfig(
            **self.obtain_training_config_dict('training_config')
        ),
    )

    def gen_model_input_fn(x: peft_trainer.TrainingInput):
      # [FUNCTION DOC]
      # HELPER: converts a TrainingInput batch into model keyword inputs. Pad mask is input_tokens != 0;
      # positions are built from the mask (cumulative count of non-pad tokens) and the attention mask
      # is the causal mask combined with the pad mask. Output: dict with input_tokens, input_mask,
      # positions, attention_mask. No side effects.
      pad_mask = x.input_tokens != 0

      positions = utils.build_positions_from_mask(pad_mask)
      attention_mask = utils.make_causal_attn_mask(pad_mask)
      return {
          'input_tokens': x.input_tokens,
          'input_mask': x.input_mask,
          'positions': positions,
          'attention_mask': attention_mask,
      }

    my_gen_model_input_fn = gen_model_input_fn
    trainer = trainer.with_gen_model_input_fn(my_gen_model_input_fn)

    train_ds, eval_ds = data_lib.create_datasets(
        dataset_name=self.config['dataset_name'],
        global_batch_size=self.config['batch_size'],
        max_target_length=self.config['max_target_length'],
        num_train_epochs=self.config['num_train_epochs'],
        tokenizer=tokenizer,
        tfds_download=self.config['tfds_download'],
    )

    with mesh:
      trainer.train(train_ds, eval_ds)


def _setup_jax_pathways(pathways_bns: str):
  # [FUNCTION DOC]
  # HELPER: configures JAX to use a Pathways backend. Input: pathways_bns server address. Sets the
  # absl flag pathways_ifrt to True, the JAX config jax_xla_backend to "pathways" and
  # jax_backend_target to the address. Output: none. Side effects: global flag/JAX config mutation;
  # raises if the pathways_ifrt flag is not defined.
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update('jax_xla_backend', 'pathways')
  jax.config.update('jax_backend_target', pathways_bns)


def main(argv, **kwargs):
  # [FUNCTION DOC]
  # CLI main. Inputs: argv (command-line args, forwarded to the pipeline for config parsing) and
  # extra kwargs (config overrides). If the pathways_bns flag is set, sets up Pathways first.
  # Builds the pipeline, logs its config at INFO and runs PEFT training. Output: none.
  # Exceptions/side effects: as for the pipeline.
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  pipeline = PeftPipeline(argv, **kwargs)
  logging.info(
      '--- Launching PEFT pipeline with following config ---\n'
      '%r\n--------------------------',
      pipeline.config,
  )
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)
