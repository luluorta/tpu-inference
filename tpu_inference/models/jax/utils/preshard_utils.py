"""Utilities for saving and loading pre-sharded JAX model checkpoints.

A preshard checkpoint stores the final, fully-processed model state (including
all weight reshaping, transposition, padding, FP8 re-quantization, and Qwix
quantization). Loading from a preshard checkpoint bypasses all preprocessing
and directly restores sharded arrays onto the TPU mesh.

Uses orbax-checkpoint for serialization, which natively supports multi-host
save/restore — each process only reads/writes its own addressable shards.
"""

import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from vllm.config import VllmConfig

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

PRESHARD_METADATA_FILENAME = "preshard_metadata.json"
PRESHARD_STATE_SUBDIR = "state"
PRESHARD_SHARDING_FILENAME = "sharding.json"
PRESHARD_METADATA_VERSION = 1


def save_preshard_checkpoint(
    jit_model: nnx.Module,
    directory: str,
    vllm_config: VllmConfig,
    mesh: Mesh,
) -> None:
    """Save the fully-processed model to a preshard checkpoint.

    This must be called AFTER all weight preprocessing is complete (load_weights,
    process_weights_after_loading, create_jit_model, Qwix quantization, etc.).

    For multi-host: all processes must call this simultaneously. orbax coordinates
    so each process writes only its addressable shards. The directory must be on
    a shared filesystem (GCS or NFS).

    Args:
        jit_model: The fully loaded, quantized, and jitted nnx.Module.
        directory: Output directory path.
        vllm_config: Current VllmConfig for metadata.
        mesh: The device mesh used for sharding.
    """
    t0 = time.perf_counter()

    _, state = nnx.split(jit_model)

    if jax.process_index() == 0:
        os.makedirs(directory, exist_ok=True)
        # Remove stale state dir from a previous failed save
        state_dir = os.path.join(directory, PRESHARD_STATE_SUBDIR)
        if os.path.exists(state_dir):
            import shutil
            shutil.rmtree(state_dir)

    # Barrier to ensure directory exists before other processes proceed
    if jax.process_count() > 1:
        jax.experimental.multihost_utils.sync_global_devices(
            "preshard_save_mkdir")

    # Save metadata and sharding info (process 0 only)
    if jax.process_index() == 0:
        metadata = _build_metadata(vllm_config, mesh, state)
        metadata_path = os.path.join(directory, PRESHARD_METADATA_FILENAME)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved preshard metadata to %s", metadata_path)

        sharding_info = _extract_sharding_info(state, mesh)
        sharding_path = os.path.join(directory, PRESHARD_SHARDING_FILENAME)
        with open(sharding_path, "w") as f:
            json.dump(sharding_info, f, indent=2)
        logger.info("Saved sharding info to %s", sharding_path)

    # All processes save state via orbax (each writes local shards)
    # Disable per-leaf sharding file and array metadata writes to avoid
    # O(n_params) small IO that causes timeouts on large MoE models.
    # Our sharding.json already records this info for validation, and
    # restore uses target-provided sharding (not the _sharding file).
    state_dir = os.path.join(directory, PRESHARD_STATE_SUBDIR)
    checkpointer = _create_save_checkpointer()
    checkpointer.save(state_dir, state)
    checkpointer.wait_until_finished()
    checkpointer.close()

    elapsed = time.perf_counter() - t0
    logger.info("Preshard checkpoint saved to %s in %.2fs", directory, elapsed)


def _create_save_checkpointer() -> ocp.AsyncCheckpointer:
    """Create a checkpointer optimized for preshard saves.

    Disables per-leaf sharding file writes (serialize_shardings) and array
    metadata store, which cause O(n_params) small IO operations that timeout
    on large MoE models. The pytree metadata file is still written (single
    JSON write, fast).
    """
    import jax
    from orbax.checkpoint._src.handlers import (
        base_pytree_checkpoint_handler as base_handler)
    from orbax.checkpoint._src.serialization import (
        jax_array_handlers, tensorstore_utils as ts_utils,
        type_handler_registry)

    # Increase tensorstore IO concurrency for faster writes on networked storage.
    # Patch the module-level default before handler creation (handler reads it
    # internally via get_ts_context).
    file_io_concurrency = int(os.environ.get(
        "PRESHARD_FILE_IO_CONCURRENCY", "512"))
    data_copy_concurrency = int(os.environ.get(
        "PRESHARD_DATA_COPY_CONCURRENCY", "128"))
    ts_utils._DEFAULT_OCDBT_TS_CONTEXT['file_io_concurrency'] = {
        'limit': file_io_concurrency}
    ts_utils._DEFAULT_OCDBT_TS_CONTEXT['data_copy_concurrency'] = {
        'limit': data_copy_concurrency}
    logger.info(
        "Tensorstore concurrency: file_io=%d, data_copy=%d",
        file_io_concurrency, data_copy_concurrency)

    array_handler = jax_array_handlers.ArrayHandler(
        enable_write_sharding_file=False,
        array_metadata_store=None,
    )
    registry = type_handler_registry.create_type_handler_registry(
        (jax.Array, array_handler),
    )
    handler = base_handler.BasePyTreeCheckpointHandler(
        type_handler_registry=registry,
    )
    return ocp.AsyncCheckpointer(
        handler,
        async_options=ocp.options.AsyncOptions(timeout_secs=7200),
    )


def _fill_abstract_weights_to_load(model: nnx.Module) -> None:
    """Populate `_weights_to_load` lists on MoE Params with abstract placeholders.

    Mirrors the dummy loader's transposed convention (weight_utils.py:1035-1059):
    storage shape is (E, axis1, axis2); each chunk is (1, axis2, axis1).
    Required so that `process_weights_after_loading` can run under eval_shape
    and produce the post-load (fused) module structure.
    """
    for _name, param in model.named_parameters():
        if not hasattr(param, '_weights_to_load'):
            continue
        E, axis1, axis2 = param.value.shape
        chunk_shape = (1, axis2, axis1)
        param._weights_to_load[:] = [
            jnp.zeros(chunk_shape, dtype=param.value.dtype)
            for _ in range(E)
        ]


def _mark_abstract_params_loaded(model: nnx.Module) -> None:
    """Set `_is_loaded=True` metadata on every Param.

    Required so that `Fp8BlockwiseLinearMethod.process_weights_after_loading`
    (fp8.py:267-273) does not early-return on its loaded-flag guard. Harmless
    for other quant methods — only Fp8BlockwiseLinearMethod reads this flag.
    """
    for _name, param in model.named_parameters():
        param.set_metadata("_is_loaded", True)


def _build_post_load_abstract_model(
    model_class: Any,
    vllm_config: VllmConfig,
    mesh: Mesh,
    rng: jax.Array,
) -> nnx.Module:
    """Build an abstract model whose structure matches the post-load state.

    Constructs the abstract module, fills `_weights_to_load` and `_is_loaded`
    so that `process_weights_after_loading` proceeds rather than early-returns,
    then walks the module tree calling each `quant_method`'s
    `process_weights_after_loading`. All under a single `nnx.eval_shape` so
    that delattr/setattr/`nnx.Param(...)` mutations land in the returned model
    and jnp ops trace abstractly.
    """
    from tpu_inference.layers.jax import JaxModuleList
    from tpu_inference.layers.jax.quantization import QuantizeMethodBase
    from tpu_inference.models.jax.utils.qwix.qwix_utils import (
        apply_qwix_on_abstract_model, apply_qwix_quantization)

    def create_concrete():
        return model_class(vllm_config, rng, mesh)

    abstract_fn = create_concrete
    if apply_qwix_on_abstract_model(vllm_config):
        abstract_fn = apply_qwix_quantization(
            vllm_config, create_concrete, rng, mesh,
            apply_to_abstract_model=True)

    def _process(module):
        # Same recursion as JaxDummyModelLoader._process_weights_after_loading
        # (weight_utils.py:1080); kept local to avoid a refactor of that
        # private method just for this single new caller.
        if (qm := getattr(module, 'quant_method', None)) is not None:
            assert isinstance(qm, QuantizeMethodBase)
            qm.process_weights_after_loading(module)
            return
        if isinstance(module, JaxModuleList):
            for sub in module:
                _process(sub)
        else:
            for _name, sub in module.named_children():
                _process(sub)

    def build():
        model = abstract_fn()
        _fill_abstract_weights_to_load(model)
        _mark_abstract_params_loaded(model)
        _process(model)
        return model

    with jax.set_mesh(mesh):
        return nnx.eval_shape(build)


def load_preshard_model(
    directory: str,
    mesh: Mesh,
    vllm_config: VllmConfig,
    model_class: Any,
    rng: jax.Array,
) -> nnx.Module:
    """Load a preshard checkpoint and return the fully-reconstructed model.

    Completely bypasses the normal HF loading + preprocessing pipeline.
    The returned model is equivalent to what _get_nnx_model() would produce
    after all weight loading, re-quantization, and JIT wrapping.

    For multi-host: all processes must call this simultaneously. orbax restores
    each process's local shards from the shared filesystem.

    Args:
        directory: Path to preshard checkpoint directory.
        mesh: The device mesh for sharding (must match the one used for save).
        vllm_config: Current VllmConfig for validation.
        model_class: The model class to instantiate for graphdef reconstruction.
        rng: JAX random key for model initialization.

    Returns:
        The restored nnx.Module with all weights on the mesh.

    Raises:
        ValueError: If metadata or sharding validation fails.
        FileNotFoundError: If checkpoint files are missing.
    """
    t0 = time.perf_counter()

    # 1. Load and validate metadata
    _load_and_validate_metadata(directory, vllm_config, mesh)

    # 2. Reconstruct abstract model whose structure matches the post-load
    #    state (after process_weights_after_loading mutations such as
    #    Fp8FusedMoEMethod's gate+up_proj fusion).
    abstract_model = _build_post_load_abstract_model(
        model_class, vllm_config, mesh, rng)
    graphdef, abstract_state = nnx.split(abstract_model)
    logger.info("Reconstructed graphdef from model class %s",
                model_class.__name__)

    # 3. Validate against saved sharding info
    sharding_path = os.path.join(directory, PRESHARD_SHARDING_FILENAME)
    if os.path.exists(sharding_path):
        with open(sharding_path, "r") as f:
            saved_sharding = json.load(f)
        _validate_sharding_compatibility(abstract_state, saved_sharding, mesh)
    else:
        logger.warning(
            "sharding.json not found in %s, skipping compatibility check. "
            "If the checkpoint is incompatible, orbax restore will fail.",
            directory)

    # 4. Build abstract restore target from model's partition specs
    abstract_target = _build_abstract_target(abstract_state, mesh)

    # 5. Restore state via orbax (all processes participate)
    state_dir = os.path.join(directory, PRESHARD_STATE_SUBDIR)
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(state_dir, target=abstract_target)
    logger.info("Restored state from %s", state_dir)

    # 6. Reconstruct the model
    model = nnx.merge(graphdef, restored_state)

    elapsed = time.perf_counter() - t0
    logger.info("Preshard model loaded from %s in %.2fs", directory, elapsed)

    return model


def _extract_sharding_info(state: nnx.State, mesh: Mesh) -> list:
    """Extract per-leaf sharding specs from the state.

    Returns a JSON-serializable list of dicts with path, shape, dtype, and
    sharding_spec for each parameter leaf.
    """
    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(state)

    leaves = []
    for path, leaf in flat_with_path:
        path_str = jax.tree_util.keystr(path)
        sharding_spec = None
        if hasattr(leaf, "sharding") and isinstance(leaf.sharding,
                                                    NamedSharding):
            spec = leaf.sharding.spec
            sharding_spec = [
                list(s) if isinstance(s, (list, tuple)) else s
                for s in spec
            ]
        leaves.append({
            "path": path_str,
            "shape": list(leaf.shape),
            "dtype": str(leaf.dtype),
            "sharding_spec": sharding_spec,
        })

    return leaves


def _build_abstract_target(
    abstract_state: nnx.State,
    mesh: Mesh,
) -> Any:
    """Build an abstract restore target from the model's partition annotations.

    Derives sharding from the abstract model's nnx partition specs, so no
    external sharding file is needed for the restore itself.
    """
    pspecs = nnx.get_partition_spec(abstract_state)
    flat_state, treedef = jax.tree_util.tree_flatten(abstract_state)
    flat_pspecs = jax.tree_util.tree_leaves(pspecs)

    abstract_leaves = []
    for leaf, pspec in zip(flat_state, flat_pspecs):
        sharding = NamedSharding(mesh, pspec)
        abstract_leaves.append(
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding))

    return treedef.unflatten(abstract_leaves)


def _validate_sharding_compatibility(
    abstract_state: nnx.State,
    saved_sharding: list,
    mesh: Mesh,
) -> None:
    """Validate that the current model's structure matches the saved checkpoint.

    Checks leaf count, shapes, dtypes, and sharding specs. Raises ValueError
    on any mismatch — a mismatch means the checkpoint is incompatible with
    the current model code or configuration.
    """
    flat_with_path, _ = jax.tree_util.tree_flatten_with_path(abstract_state)
    pspecs = jax.tree_util.tree_leaves(
        nnx.get_partition_spec(abstract_state))

    if len(flat_with_path) != len(saved_sharding):
        raise ValueError(
            f"Preshard checkpoint incompatible: model has "
            f"{len(flat_with_path)} parameters, checkpoint has "
            f"{len(saved_sharding)}")

    for i, ((path, leaf), saved, pspec) in enumerate(
            zip(flat_with_path, saved_sharding, pspecs)):
        current_path = jax.tree_util.keystr(path)
        saved_path = saved.get("path", f"<index {i}>")

        if current_path != saved_path:
            raise ValueError(
                f"Preshard path mismatch at index {i}: "
                f"model has '{current_path}', checkpoint has '{saved_path}'")

        saved_shape = tuple(saved["shape"])
        if leaf.shape != saved_shape:
            raise ValueError(
                f"Preshard shape mismatch at '{current_path}': "
                f"model expects {leaf.shape}, checkpoint has {saved_shape}")

        saved_dtype = saved["dtype"]
        if str(leaf.dtype) != saved_dtype:
            raise ValueError(
                f"Preshard dtype mismatch at '{current_path}': "
                f"model expects {leaf.dtype}, checkpoint has {saved_dtype}")

        # Compare sharding specs (normalize: None and [] both mean replicated)
        saved_spec = saved.get("sharding_spec")
        current_spec = [
            list(s) if isinstance(s, (list, tuple)) else s
            for s in pspec
        ] if pspec else None
        if not saved_spec:
            saved_spec = None
        if not current_spec:
            current_spec = None

        if saved_spec != current_spec:
            raise ValueError(
                f"Preshard sharding mismatch at '{current_path}': "
                f"model expects {current_spec}, checkpoint has {saved_spec}. "
                f"This may indicate a TP/EP configuration change.")


def _build_metadata(
    vllm_config: VllmConfig,
    mesh: Mesh,
    state: nnx.State,
) -> dict:
    """Build metadata dict for a preshard checkpoint."""
    num_params = len(jax.tree_util.tree_leaves(state))

    model_config = vllm_config.model_config
    architectures = getattr(model_config.hf_config, "architectures", [])
    architecture = architectures[0] if architectures else "unknown"

    quant_config = None
    if hasattr(model_config.hf_config, "quantization_config"):
        qc = model_config.hf_config.quantization_config
        if isinstance(qc, dict):
            quant_config = qc
        elif hasattr(qc, "to_dict"):
            quant_config = qc.to_dict()

    return {
        "version": PRESHARD_METADATA_VERSION,
        "model_name": model_config.model,
        "architecture": architecture,
        "mesh_shape": dict(mesh.shape),
        "tp_size": vllm_config.parallel_config.tensor_parallel_size,
        "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        "model_dtype": str(model_config.dtype),
        "num_params": num_params,
        "quantization": quant_config,
        "jax_process_count": jax.process_count(),
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
    }


def _load_and_validate_metadata(
    directory: str,
    vllm_config: VllmConfig,
    mesh: Mesh,
) -> dict:
    """Load and validate preshard metadata against current config."""
    metadata_path = os.path.join(directory, PRESHARD_METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Preshard metadata not found at {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    version = metadata.get("version", 0)
    if version != PRESHARD_METADATA_VERSION:
        raise ValueError(
            f"Preshard metadata version mismatch: got {version}, "
            f"expected {PRESHARD_METADATA_VERSION}")

    saved_mesh_shape = metadata.get("mesh_shape", {})
    current_mesh_shape = dict(mesh.shape)
    if saved_mesh_shape != current_mesh_shape:
        raise ValueError(
            f"Mesh shape mismatch: checkpoint saved with {saved_mesh_shape}, "
            f"current mesh is {current_mesh_shape}")

    saved_tp = metadata.get("tp_size")
    current_tp = vllm_config.parallel_config.tensor_parallel_size
    if saved_tp != current_tp:
        raise ValueError(
            f"Tensor parallel size mismatch: checkpoint tp={saved_tp}, "
            f"current tp={current_tp}")

    saved_pp = metadata.get("pp_size", 1)
    current_pp = vllm_config.parallel_config.pipeline_parallel_size
    if saved_pp != 1 or current_pp != 1:
        raise ValueError(
            "Pipeline parallelism > 1 not supported for preshard. "
            f"Checkpoint pp={saved_pp}, current pp={current_pp}")

    saved_procs = metadata.get("jax_process_count", 1)
    current_procs = jax.process_count()
    if saved_procs != current_procs:
        raise ValueError(
            f"Process count mismatch: checkpoint saved with {saved_procs} "
            f"processes, current run has {current_procs}")

    logger.info(
        "Preshard metadata validated: model=%s, arch=%s, tp=%d, mesh=%s",
        metadata.get("model_name"), metadata.get("architecture"),
        saved_tp, saved_mesh_shape)

    return metadata


def get_preshard_path(vllm_config: VllmConfig) -> str:
    """Get preshard checkpoint path from config or environment.

    Checks in order:
    1. PRESHARD_CHECKPOINT_PATH environment variable
    2. additional_config["preshard_checkpoint"] in VllmConfig

    Raises:
        ValueError: If no preshard path is configured.
    """
    path = os.environ.get("PRESHARD_CHECKPOINT_PATH", "")
    if not path:
        additional = getattr(vllm_config, "additional_config", None) or {}
        path = additional.get("preshard_checkpoint", "")
    if not path:
        raise ValueError(
            "Preshard checkpoint path not specified. "
            "Set PRESHARD_CHECKPOINT_PATH env var or pass "
            "--additional-config '{\"preshard_checkpoint\": \"/path/to/dir\"}'")
    return path
