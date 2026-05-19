"""Registered model loader for pre-sharded JAX checkpoints.

This module registers the "jax_preshard" load format with vLLM's model loader
system. The actual loading logic is in preshard_utils.py and is invoked at the
get_flax_model() level to completely bypass _get_nnx_model().
"""

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


@register_model_loader("jax_preshard")
class JaxPreshardModelLoader(BaseModelLoader):
    """Model loader for pre-sharded orbax checkpoints.

    Skips ALL weight loading, preprocessing, and quantization. The preshard
    checkpoint contains the final model state (post re-quant, post Qwix, etc.).

    Note: This loader's load_weights() is not called in the normal flow.
    Preshard loading is handled at the get_flax_model() level to completely
    bypass _get_nnx_model(). This class exists only so that
    get_model_loader("jax_preshard") resolves without error.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model, model_config: ModelConfig) -> None:
        raise RuntimeError(
            "JaxPreshardModelLoader.load_weights should not be called. "
            "Preshard loading is handled at the get_flax_model() level, "
            "bypassing _get_nnx_model() entirely.")
