"""Save a pre-sharded JAX checkpoint for fast model loading on TPU.

This script loads a model through the full pipeline (HF safetensors → preprocess
→ quantize → shard to mesh), then saves the final state as a preshard checkpoint.
Subsequent loads with load_format="jax_preshard" skip all preprocessing.

Usage:
    python examples/save_jax_preshard.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --tensor-parallel-size 4 \
        --output /path/to/preshard/output

    # With FP8 quantization:
    python examples/save_jax_preshard.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --tensor-parallel-size 4 \
        --quantization fp8 \
        --output /path/to/preshard/output

Then load with:
    PRESHARD_CHECKPOINT_PATH=/path/to/preshard/output \
    python examples/offline_inference.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --load-format jax_preshard \
        --tensor-parallel-size 4
"""

import os
import shutil
from pathlib import Path

# Skip precompilation — preshard only needs the loaded weights, not compiled graphs.
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
# Disable multiprocessing so we can directly access model_executor/worker.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.preshard_utils import (
    save_preshard_checkpoint)

logger = init_logger(__name__)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Save a pre-sharded JAX checkpoint for fast model loading")
    EngineArgs.add_cli_args(parser)
    # Minimal settings for save-only: small context, minimal KV cache allocation.
    parser.set_defaults(max_model_len=128, num_gpu_blocks_override=1)
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=str,
        help="Output directory for the preshard checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output

    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        logger.warning("Output directory %s already exists and is not empty",
                       output_dir)

    # Pop our custom arg before passing to LLM
    delattr(args, "output")

    # Create LLM — this triggers the full model loading pipeline
    engine_args = EngineArgs.from_cli_args(args)
    model_path = engine_args.model
    logger.info("Loading model from %s with full preprocessing...", model_path)

    llm = LLM(**vars(engine_args))

    # Access the internal model runner to get the jit_model.
    # In non-MP mode, model_executor is exposed directly on llm_engine.
    worker = llm.llm_engine.model_executor.driver_worker
    model_runner = worker.model_runner

    # The jit_model is stored as model_runner.model (set in load_model)
    jit_model = model_runner.model
    mesh = model_runner.mesh
    vllm_config = model_runner.vllm_config

    if jit_model is None:
        raise RuntimeError("Model not loaded. Check model loading logs.")

    logger.info("Model loaded successfully. Saving preshard checkpoint...")

    # Save the preshard checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_preshard_checkpoint(jit_model, output_dir, vllm_config, mesh)

    # Copy non-weight files (config.json, tokenizer, etc.) from the model dir
    if Path(model_path).is_dir():
        _copy_model_metadata(model_path, output_dir)

    logger.info("Preshard checkpoint saved to %s", output_dir)


def _copy_model_metadata(model_path: str, output_dir: str) -> None:
    """Copy non-weight metadata files from the model directory."""
    weight_extensions = {".bin", ".pt", ".safetensors", ".pkl"}
    preshard_files = {
        "preshard_metadata.json", "sharding.json", "state"
    }

    for item in os.listdir(model_path):
        if item in preshard_files:
            continue
        src = os.path.join(model_path, item)
        dst = os.path.join(output_dir, item)
        if os.path.exists(dst):
            continue
        ext = os.path.splitext(item)[1].lower()
        if ext in weight_extensions:
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    logger.info("Copied model metadata files to %s", output_dir)


if __name__ == "__main__":
    main()
