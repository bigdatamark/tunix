# [MODULE DOC]
# Numerical alignment tests between HuggingFace PyTorch Qwen models and Tunix JAX implementations
# (DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-1.5B-Instruct via qwen2, Qwen3-0.6B via qwen3).
# Downloads each checkpoint to the temp dir under models/<name>, loads it in both frameworks in
# float32, checks the first query weight is identical, then compares all-decoder-layer hidden states
# and full-model logits for input tokens 1..128 with causal masks within a per-model tolerance.
# Side effects: network downloads, prints, and deletion of the downloaded directory on success.
# The constant K_MASK is the large negative additive mask value used for PyTorch masks.
"""Check the model correctness for Tunix nnx implemented models.

The test will compare the first N decoder layer output between Tunix model and
HF PyTorch model, typically we will expect the logits differnece to be within
1e-3 in fp32.
"""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from tunix.models.qwen2 import model as qwen2_model
from tunix.models.qwen2 import params as qwen2_params
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.sft import utils
from tunix.tests import test_common as tc

K_MASK = -2.3819763e38


def create_pytorch_causal_mask(seq_len):
  # [FUNCTION DOC]
  # HELPER: builds a (seq_len, seq_len) float additive causal mask: 0 where column index is at most the
  # row index, K_MASK elsewhere.
  # KNOWN ISSUE: the docstring says a boolean True/False tensor is returned, but the result is a float
  # tensor of 0 and K_MASK values.
  """Creates a causal attention mask for a sequence of a given length.

  Args:
    seq_len: The length of the sequence.

  Returns:
    A boolean tensor of shape (seq_len, seq_len) where:
    - mask[i, j] is True if token i can attend to token j (j <= i).
    - mask[i, j] is False if token i cannot attend to token j (j > i).
  """
  # Create a lower triangular matrix of ones
  mask = torch.ones(seq_len, seq_len, dtype=torch.float).tril(diagonal=0)
  mask = mask.masked_fill(mask == 0, K_MASK)
  mask = mask.masked_fill(mask == 1, 0)
  return mask


def get_hf_output(model, seq_len: int):
  # [FUNCTION DOC]
  # HELPER: runs the HF model on tokens 1..seq_len (batch 1) with positions 0..seq_len-1 and a
  # [1, 1, seq, seq] additive causal mask passed positionally; returns logits as a numpy array.
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
  return model(x, attn_mask, position_ids).logits.detach().numpy()


def get_jax_output(model, seq_len: int):
  # [FUNCTION DOC]
  # HELPER: runs the Tunix model on tokens 1..seq_len with positions 0..seq_len-1, no cache, and a
  # causal mask from utils.make_causal_attn_mask over an all-ones input mask; returns the logits.
  x = (jnp.arange(seq_len) + 1).reshape(1, -1)
  positions = jnp.arange(seq_len).reshape(1, -1)
  attn_mask = utils.make_causal_attn_mask(jnp.ones((1, seq_len)))
  output, _ = model(x, positions, None, attn_mask)
  return output


def get_per_layer_hf_output(model, seq_len: int, num_layer_to_run: int = 1):
  # [FUNCTION DOC]
  # HELPER: computes HF decoder hidden states after the first num_layer_to_run layers (default 1): embeds
  # tokens 1..seq_len, computes rotary position embeddings, applies each layer in turn with the additive
  # causal mask and position ids, and returns the first output of the last layer for batch element 0 as
  # numpy.
  # KNOWN ISSUE: the docstring says it returns the first layer's output, but it runs num_layer_to_run
  # layers. Indexing the layer result with zero assumes the layer returns a tuple; on versions where it
  # returns a tensor this selects the first batch row instead (which still works for batch 1).
  """Get the first decoder layer output from the HF model."""
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)

  m = model.get_decoder()
  emb = m.embed_tokens(x)
  position_embeddings = m.rotary_emb(emb, position_ids)

  logits = emb
  for i in range(num_layer_to_run):
    logits = m.layers[i](
        logits, attn_mask, position_ids, position_embeddings=position_embeddings
    )

  return logits[0].detach().numpy()


def get_per_layer_jax_output(
    model_name: str, model, seq_len: int, num_layer_to_run: int = 1
):
  # [FUNCTION DOC]
  # HELPER: computes Tunix hidden states after num_layer_to_run decoder layers. For the two qwen2-based
  # model names it generates sin/cos rotary embeddings via qwen2_model._generate_pos_embeddings and calls
  # each layer with (hidden, no cache, mask, sin, cos); for Qwen3-0.6B it calls each layer with (hidden,
  # positions, no cache, mask). Each layer returns (cache, hidden). Returns the final hidden states.
  # Raises ValueError for any other model name.
  """Get the first decoder layer output from the Tunix model."""
  x = (jnp.arange(seq_len) + 1).reshape(1, -1)
  positions = jnp.arange(seq_len).reshape(1, -1)
  attn_mask = utils.make_causal_attn_mask(jnp.ones((1, seq_len)))

  if model_name in [
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
      "Qwen/Qwen2.5-1.5B-Instruct",
  ]:
    sin, cos = qwen2_model._generate_pos_embeddings(  # pylint: disable=protected-access
        positions, model.config.head_dim, model.config.rope_theta
    )

    logits = model.embedder.encode(x)
    for i in range(num_layer_to_run):
      _, logits = model.layers[i](logits, None, attn_mask, sin, cos)

    return logits
  elif model_name == "Qwen/Qwen3-0.6B":
    logits = model.embedder.encode(x)
    for i in range(num_layer_to_run):
      _, logits = model.layers[i](logits, positions, None, attn_mask)

    return logits
  else:
    raise ValueError(f"Unsupported model: {model_name}")


class QwenAlignTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized alignment test across Qwen-family checkpoints.

  @parameterized.named_parameters(
      dict(
          testcase_name="deepseek_r1_distill_qwen_1p5b",
          model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
          model_config=qwen2_model.ModelConfig.deepseek_r1_distill_qwen_1p5b,
          model_params=qwen2_params,
          tolerance=2e-3,
      ),
      dict(
          testcase_name="qwen2p5_1p5b_instruct",
          model_name="Qwen/Qwen2.5-1.5B-Instruct",
          model_config=qwen2_model.ModelConfig.qwen2p5_1p5b,
          model_params=qwen2_params,
          tolerance=1e-3,
      ),
      dict(
          testcase_name="qwen3_0p6b",
          model_name="Qwen/Qwen3-0.6B",
          model_config=qwen3_model.ModelConfig.qwen3_0p6b,
          model_params=qwen3_params,
          tolerance=1e-3,
      ),
  )
  def test_qwen_model_alignment(
      self, model_name, model_config, model_params, tolerance
  ):
    # [FUNCTION DOC]
    # Verifies alignment for one model. Inputs: model_name (HF repo id), model_config (config factory),
    # model_params (Tunix params module with create_model_from_safe_tensors), tolerance (atol and rtol).
    # Loads both models, asserts exact equality of the HF layer-0 q_proj weight with the Tunix q_proj
    # weight reshaped to (embed, heads*head_dim) and transposed, compares hidden states after all layers
    # and then full logits with assert_allclose, prints success and deletes the model directory.
    model_path = os.path.join(tempfile.gettempdir(), "models", model_name)

    tc.download_from_huggingface(repo_id=model_name, model_path=model_path)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    )
    print("HF model loaded.")

    jax_model = model_params.create_model_from_safe_tensors(
        model_path,
        model_config(),
        mesh=jax.make_mesh(
            (1, 1),
            ("fsdp", "tp"),
            axis_types=(jax.sharding.AxisType.Auto,) * len(("fsdp", "tp")),
        ),
        dtype=jnp.float32,
    )
    print("JAX model loaded.")

    # Make sure model weights are the same (only check the first query weight)
    hf_query_weight = (
        hf_model.get_decoder()
        .layers[0]
        .self_attn.q_proj.weight.detach()
        .numpy()
    )
    jax_query_weight = jax_model.layers[0].attn.q_proj.w
    d, _, _ = jax_query_weight.shape
    jax_query_weight = jax_query_weight.reshape(d, -1).transpose()
    np.testing.assert_equal(
        hf_query_weight,
        jax_query_weight,
        err_msg=(
            "Query weights are not equal, are you sure the loaded model weight"
            " between HF and JAX is identical?"
        ),
    )

    seq_len = 128

    layer_to_run = model_config().num_layers
    hf_logits = get_per_layer_hf_output(hf_model, seq_len, layer_to_run)
    jax_logits = get_per_layer_jax_output(
        model_name, jax_model, seq_len, layer_to_run
    )
    np.testing.assert_allclose(
        hf_logits.squeeze(),
        jax_logits.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )

    # Do a check on entire model output
    hf_output = get_hf_output(hf_model, seq_len)
    jax_output = get_jax_output(jax_model, seq_len)
    np.testing.assert_allclose(
        hf_output.squeeze(),
        jax_output.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )

    print("Logits are close! Model alignment check passed :)")

    # clean up
    tc.delete_directory(model_path)


if __name__ == "__main__":
  absltest.main()
