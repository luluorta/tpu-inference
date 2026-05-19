"""Unit tests for preshard save/load utilities.

Tests the full save→load roundtrip on CPU with a tiny dummy model,
validating that weights are correctly preserved and metadata validation works.

NOTE: orbax StandardCheckpointer has async-thread issues on macOS CPU, so
roundtrip tests mock the orbax layer and test our save/load logic directly.
"""

import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.models.jax.utils.preshard_utils import (
    PRESHARD_METADATA_FILENAME,
    PRESHARD_METADATA_VERSION,
    PRESHARD_SHARDING_FILENAME,
    PRESHARD_STATE_SUBDIR,
    _build_abstract_target,
    _build_metadata,
    _build_post_load_abstract_model,
    _extract_sharding_info,
    _load_and_validate_metadata,
    _validate_sharding_compatibility,
    get_preshard_path,
    load_preshard_model,
    save_preshard_checkpoint,
)


class DummyLinear(JaxModule):
    """Minimal linear layer for testing."""

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features))
        )
        self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x):
        return x @ self.kernel.value + self.bias.value


class DummyModel(JaxModule):
    """Tiny two-layer model for testing preshard save/load.

    Accepts the same (vllm_config, rng, mesh) signature as real model classes
    so it can be used with load_preshard_model's abstract model reconstruction.
    """

    def __init__(self, vllm_config=None, rng=None, mesh=None, hidden: int = 8):
        rngs = nnx.Rngs(0)
        self.layer1 = DummyLinear(hidden, hidden, rngs)
        self.layer2 = DummyLinear(hidden, hidden, rngs)

    def __call__(self, x):
        x = self.layer1(x)
        x = nnx.relu(x)
        x = self.layer2(x)
        return x


def _make_mock_vllm_config(
    model_name: str = "dummy-model",
    tp_size: int = 1,
    pp_size: int = 1,
    dtype: str = "float32",
    additional_config: dict = None,
) -> MagicMock:
    """Create a minimal mock VllmConfig for testing."""
    config = MagicMock()
    config.model_config.model = model_name
    config.model_config.dtype = dtype
    config.model_config.hf_config.architectures = ["DummyForCausalLM"]
    config.model_config.hf_config.quantization_config = None
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = pp_size
    config.additional_config = additional_config or {}
    # For load_config
    config.load_config.load_format = "jax_preshard"
    return config


def _make_cpu_mesh(axis_name: str = "tp", size: int = 1) -> Mesh:
    """Create a 1-device CPU mesh."""
    devices = jax.devices("cpu")[:size]
    return Mesh(devices, axis_names=(axis_name,))


class TestExtractShardingInfo(unittest.TestCase):

    def test_extracts_shape_dtype_sharding(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        _, state = nnx.split(model)

        info = _extract_sharding_info(state, mesh)

        self.assertIsInstance(info, list)
        self.assertGreater(len(info), 0)

        for leaf_info in info:
            self.assertIn("path", leaf_info)
            self.assertIn("shape", leaf_info)
            self.assertIn("dtype", leaf_info)
            self.assertIn("sharding_spec", leaf_info)

    def test_leaf_shapes_match_model_params(self):
        model = DummyModel(hidden=4)
        mesh = _make_cpu_mesh()
        _, state = nnx.split(model)

        info = _extract_sharding_info(state, mesh)

        # DummyModel has 4 params: layer1.kernel, layer1.bias, layer2.kernel, layer2.bias
        self.assertEqual(len(info), 4)

        shapes = [tuple(l["shape"]) for l in info]
        self.assertIn((4, 4), shapes)  # kernel
        self.assertIn((4,), shapes)  # bias

    def test_json_serializable(self):
        model = DummyModel(hidden=4)
        mesh = _make_cpu_mesh()
        _, state = nnx.split(model)

        info = _extract_sharding_info(state, mesh)
        # Should not raise
        serialized = json.dumps(info)
        restored = json.loads(serialized)
        self.assertEqual(len(restored), len(info))


class TestBuildAbstractTarget(unittest.TestCase):

    def test_roundtrip_structure(self):
        model = DummyModel(hidden=4)
        mesh = _make_cpu_mesh()
        _, state = nnx.split(model)

        abstract_target = _build_abstract_target(state, mesh)

        original_leaves = jax.tree_util.tree_leaves(state)
        abstract_leaves = jax.tree_util.tree_leaves(abstract_target)
        self.assertEqual(len(original_leaves), len(abstract_leaves))

        for orig, abstract in zip(original_leaves, abstract_leaves):
            self.assertEqual(orig.shape, abstract.shape)
            self.assertEqual(orig.dtype, abstract.dtype)


class TestBuildMetadata(unittest.TestCase):

    def test_metadata_fields(self):
        model = DummyModel(hidden=4)
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()
        _, state = nnx.split(model)

        metadata = _build_metadata(vllm_config, mesh, state)

        self.assertEqual(metadata["version"], PRESHARD_METADATA_VERSION)
        self.assertEqual(metadata["model_name"], "dummy-model")
        self.assertEqual(metadata["architecture"], "DummyForCausalLM")
        self.assertEqual(metadata["tp_size"], 1)
        self.assertEqual(metadata["num_params"], 4)
        self.assertIn("mesh_shape", metadata)
        self.assertIn("creation_timestamp", metadata)


class _FakeCheckpointer:
    """Pickle-based stand-in for orbax StandardCheckpointer (avoids macOS issues)."""

    def __init__(self, **kwargs):
        self.handler = type('FakeHandler', (), {})()

    def save(self, path, state):
        os.makedirs(path, exist_ok=True)
        leaves, treedef = jax.tree_util.tree_flatten(state)
        data = {
            "leaves": [np.array(l) for l in leaves],
            "treedef": treedef,
        }
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)

    def restore(self, path, target=None):
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        leaves = [jnp.array(l) for l in data["leaves"]]
        return data["treedef"].unflatten(leaves)

    def wait_until_finished(self):
        pass

    def close(self):
        pass


class TestSaveLoadRoundtrip(unittest.TestCase):
    """Integration test: save a dummy model and load it back, verify weights match.

    Mocks orbax StandardCheckpointer to avoid macOS CPU async-thread issues.
    """

    @patch(
        "tpu_inference.models.jax.utils.preshard_utils.ocp.StandardCheckpointer",
        _FakeCheckpointer,
    )
    @patch(
        "tpu_inference.models.jax.utils.preshard_utils._create_save_checkpointer",
        lambda: _FakeCheckpointer(),
    )
    @patch(
        "tpu_inference.models.jax.utils.qwix.qwix_utils.apply_qwix_on_abstract_model",
        return_value=False,
    )
    def test_save_and_load_roundtrip(self, _mock_qwix):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=8)
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.key(0)

        original_params = {
            "layer1_kernel": np.array(model.layer1.kernel.value),
            "layer1_bias": np.array(model.layer1.bias.value),
            "layer2_kernel": np.array(model.layer2.kernel.value),
            "layer2_bias": np.array(model.layer2.bias.value),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preshard_checkpoint(model, tmpdir, vllm_config, mesh)

            # Verify checkpoint files exist (no graphdef.pkl anymore)
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, PRESHARD_METADATA_FILENAME))
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmpdir, PRESHARD_SHARDING_FILENAME)
                )
            )
            self.assertTrue(
                os.path.isdir(os.path.join(tmpdir, PRESHARD_STATE_SUBDIR))
            )

            # Load — now passes model_class and rng
            restored_model = load_preshard_model(
                tmpdir, mesh, vllm_config, DummyModel, rng)

            # Verify weights match
            np.testing.assert_array_equal(
                np.array(restored_model.layer1.kernel.value),
                original_params["layer1_kernel"],
            )
            np.testing.assert_array_equal(
                np.array(restored_model.layer1.bias.value),
                original_params["layer1_bias"],
            )
            np.testing.assert_array_equal(
                np.array(restored_model.layer2.kernel.value),
                original_params["layer2_kernel"],
            )
            np.testing.assert_array_equal(
                np.array(restored_model.layer2.bias.value),
                original_params["layer2_bias"],
            )

    @patch(
        "tpu_inference.models.jax.utils.preshard_utils.ocp.StandardCheckpointer",
        _FakeCheckpointer,
    )
    @patch(
        "tpu_inference.models.jax.utils.preshard_utils._create_save_checkpointer",
        lambda: _FakeCheckpointer(),
    )
    @patch(
        "tpu_inference.models.jax.utils.qwix.qwix_utils.apply_qwix_on_abstract_model",
        return_value=False,
    )
    def test_save_and_load_preserves_model_forward(self, _mock_qwix):
        """Verify the loaded model produces same output as original."""
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=8)
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.key(0)

        x = jnp.ones((2, 8))
        original_output = np.array(model(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preshard_checkpoint(model, tmpdir, vllm_config, mesh)
            restored_model = load_preshard_model(
                tmpdir, mesh, vllm_config, DummyModel, rng)

            restored_output = np.array(restored_model(x))
            np.testing.assert_array_equal(restored_output, original_output)

    @patch(
        "tpu_inference.models.jax.utils.preshard_utils.ocp.StandardCheckpointer",
        _FakeCheckpointer,
    )
    @patch(
        "tpu_inference.models.jax.utils.preshard_utils._create_save_checkpointer",
        lambda: _FakeCheckpointer(),
    )
    def test_metadata_content_is_valid_json(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        vllm_config = _make_mock_vllm_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preshard_checkpoint(model, tmpdir, vllm_config, mesh)

            metadata_path = os.path.join(tmpdir, PRESHARD_METADATA_FILENAME)
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.assertEqual(metadata["version"], PRESHARD_METADATA_VERSION)
            self.assertEqual(metadata["model_name"], "dummy-model")


class TestLoadAndValidateMetadata(unittest.TestCase):

    def _save_metadata(self, tmpdir, **overrides):
        metadata = {
            "version": PRESHARD_METADATA_VERSION,
            "model_name": "dummy-model",
            "mesh_shape": {"tp": 1},
            "tp_size": 1,
            "pp_size": 1,
            "jax_process_count": 1,
        }
        metadata.update(overrides)
        with open(os.path.join(tmpdir, PRESHARD_METADATA_FILENAME), "w") as f:
            json.dump(metadata, f)

    def test_valid_metadata_passes(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config(tp_size=1, pp_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_metadata(tmpdir)
            result = _load_and_validate_metadata(tmpdir, vllm_config, mesh)
            self.assertEqual(result["model_name"], "dummy-model")

    def test_version_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_metadata(tmpdir, version=999)
            with self.assertRaises(ValueError) as ctx:
                _load_and_validate_metadata(tmpdir, vllm_config, mesh)
            self.assertIn("version mismatch", str(ctx.exception))

    def test_mesh_shape_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_metadata(tmpdir, mesh_shape={"tp": 4})
            with self.assertRaises(ValueError) as ctx:
                _load_and_validate_metadata(tmpdir, vllm_config, mesh)
            self.assertIn("Mesh shape mismatch", str(ctx.exception))

    def test_tp_size_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config(tp_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_metadata(tmpdir, tp_size=4)
            with self.assertRaises(ValueError) as ctx:
                _load_and_validate_metadata(tmpdir, vllm_config, mesh)
            self.assertIn("Tensor parallel size mismatch", str(ctx.exception))

    def test_pp_greater_than_one_raises(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config(pp_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_metadata(tmpdir, pp_size=2)
            with self.assertRaises(ValueError) as ctx:
                _load_and_validate_metadata(tmpdir, vllm_config, mesh)
            self.assertIn("Pipeline parallelism", str(ctx.exception))

    def test_missing_metadata_file_raises(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _load_and_validate_metadata(tmpdir, vllm_config, mesh)


class TestGetPreshardPath(unittest.TestCase):

    def test_from_env_var(self):
        vllm_config = _make_mock_vllm_config()
        with patch.dict(os.environ, {"PRESHARD_CHECKPOINT_PATH": "/tmp/ckpt"}):
            self.assertEqual(get_preshard_path(vllm_config), "/tmp/ckpt")

    def test_from_additional_config(self):
        vllm_config = _make_mock_vllm_config(
            additional_config={"preshard_checkpoint": "/data/preshard"}
        )
        with patch.dict(os.environ, {}, clear=True):
            # Ensure env var is not set
            os.environ.pop("PRESHARD_CHECKPOINT_PATH", None)
            self.assertEqual(get_preshard_path(vllm_config), "/data/preshard")

    def test_env_var_takes_precedence(self):
        vllm_config = _make_mock_vllm_config(
            additional_config={"preshard_checkpoint": "/data/preshard"}
        )
        with patch.dict(
            os.environ, {"PRESHARD_CHECKPOINT_PATH": "/env/path"}
        ):
            self.assertEqual(get_preshard_path(vllm_config), "/env/path")

    def test_missing_path_raises(self):
        vllm_config = _make_mock_vllm_config()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PRESHARD_CHECKPOINT_PATH", None)
            with self.assertRaises(ValueError) as ctx:
                get_preshard_path(vllm_config)
            self.assertIn("not specified", str(ctx.exception))


class TestLoadPreshardModelErrors(unittest.TestCase):

    @patch(
        "tpu_inference.models.jax.utils.qwix.qwix_utils.apply_qwix_on_abstract_model",
        return_value=False,
    )
    @patch(
        "tpu_inference.models.jax.utils.preshard_utils.ocp.StandardCheckpointer",
        _FakeCheckpointer,
    )
    @patch(
        "tpu_inference.models.jax.utils.preshard_utils._create_save_checkpointer",
        lambda: _FakeCheckpointer(),
    )
    def test_shape_mismatch_raises(self, _mock_qwix):
        """Checkpoint saved with different hidden size should fail validation."""
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.key(0)

        # Save a model with hidden=8
        model = DummyModel(hidden=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_preshard_checkpoint(model, tmpdir, vllm_config, mesh)

            # Try to load with a model class that produces hidden=4
            class SmallDummyModel(JaxModule):
                def __init__(self, vllm_config=None, rng=None, mesh=None):
                    rngs = nnx.Rngs(0)
                    self.layer1 = DummyLinear(4, 4, rngs)
                    self.layer2 = DummyLinear(4, 4, rngs)

            with self.assertRaises(ValueError) as ctx:
                load_preshard_model(
                    tmpdir, mesh, vllm_config, SmallDummyModel, rng)
            self.assertIn("shape mismatch", str(ctx.exception))


class TestValidateShardingCompatibility(unittest.TestCase):

    def test_matching_state_passes(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        _, state = nnx.split(model)
        sharding_info = _extract_sharding_info(state, mesh)
        # Should not raise
        _validate_sharding_compatibility(state, sharding_info, mesh)

    def test_leaf_count_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        _, state = nnx.split(model)
        # Remove one entry
        sharding_info = _extract_sharding_info(state, mesh)[:3]
        with self.assertRaises(ValueError) as ctx:
            _validate_sharding_compatibility(state, sharding_info, mesh)
        self.assertIn("parameters", str(ctx.exception))

    def test_shape_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        _, state = nnx.split(model)
        sharding_info = _extract_sharding_info(state, mesh)
        sharding_info[0]["shape"] = [8, 8]  # wrong shape
        with self.assertRaises(ValueError) as ctx:
            _validate_sharding_compatibility(state, sharding_info, mesh)
        self.assertIn("shape mismatch", str(ctx.exception))

    def test_dtype_mismatch_raises(self):
        mesh = _make_cpu_mesh()
        model = DummyModel(hidden=4)
        _, state = nnx.split(model)
        sharding_info = _extract_sharding_info(state, mesh)
        sharding_info[0]["dtype"] = "float16"  # wrong dtype
        with self.assertRaises(ValueError) as ctx:
            _validate_sharding_compatibility(state, sharding_info, mesh)
        self.assertIn("dtype mismatch", str(ctx.exception))


class _FakeFusionMethod(QuantizeMethodBase):
    """Mock quant_method that mimics FP8 MoE fusion: kernel_a + kernel_b -> kernel_ab."""

    def apply_jax(self, layer, *args, **kwargs):
        raise NotImplementedError("Not used in tests")

    def process_weights_after_loading(self, layer) -> bool:
        a = layer.kernel_a.value
        b = layer.kernel_b.value
        fused = jnp.concatenate([a, b], axis=-1)
        del layer.kernel_a
        del layer.kernel_b
        layer.kernel_ab = nnx.Param(fused)
        return True


class _FakeFusionLayer(JaxModule):
    """Layer with two kernels that get fused by _FakeFusionMethod."""

    def __init__(self, dim_in: int, dim_out: int, rngs: nnx.Rngs):
        self.kernel_a = nnx.Param(
            jax.random.normal(rngs.params(), (dim_in, dim_out)))
        self.kernel_b = nnx.Param(
            jax.random.normal(rngs.params(), (dim_in, dim_out)))
        self.quant_method = _FakeFusionMethod()


class _FakeFusionModel(JaxModule):
    """JaxModule-based model holding one fake-fusion layer.

    Accepts the (vllm_config, rng, mesh) signature expected by
    _build_post_load_abstract_model.
    """

    def __init__(self, vllm_config=None, rng=None, mesh=None, dim: int = 8):
        rngs = nnx.Rngs(0)
        self.fusion = _FakeFusionLayer(dim, dim, rngs)


class _NoQuantModel(JaxModule):
    """JaxModule-based model with no quant_method anywhere."""

    def __init__(self, vllm_config=None, rng=None, mesh=None, dim: int = 8):
        rngs = nnx.Rngs(0)
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (dim, dim)))


class TestBuildPostLoadAbstractModel(unittest.TestCase):
    """Test that _build_post_load_abstract_model applies post-load structural
    mutations from process_weights_after_loading under abstract tracing."""

    def test_fusion_structure_applied(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.PRNGKey(0)

        abstract = _build_post_load_abstract_model(
            _FakeFusionModel, vllm_config, mesh, rng)
        _, state = nnx.split(abstract)
        leaves = jax.tree_util.tree_leaves(state)

        # Pre-fusion: 2 params (kernel_a, kernel_b).
        # Post-fusion: 1 param (kernel_ab).
        self.assertEqual(len(leaves), 1)
        self.assertTrue(hasattr(abstract.fusion, 'kernel_ab'))
        self.assertFalse(hasattr(abstract.fusion, 'kernel_a'))
        self.assertFalse(hasattr(abstract.fusion, 'kernel_b'))

    def test_fused_shape_correct(self):
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.PRNGKey(0)

        abstract = _build_post_load_abstract_model(
            _FakeFusionModel, vllm_config, mesh, rng, )
        # kernel_a and kernel_b were each (8, 8); fused along axis=-1 -> (8, 16)
        self.assertEqual(abstract.fusion.kernel_ab.value.shape, (8, 16))

    def test_no_quant_state_unchanged(self):
        """Regression: with no quant_method, new path matches the old one."""
        mesh = _make_cpu_mesh()
        vllm_config = _make_mock_vllm_config()
        rng = jax.random.PRNGKey(0)

        with jax.set_mesh(mesh):
            old_abstract = nnx.eval_shape(
                lambda: _NoQuantModel(vllm_config, rng, mesh))
        _, old_state = nnx.split(old_abstract)

        new_abstract = _build_post_load_abstract_model(
            _NoQuantModel, vllm_config, mesh, rng)
        _, new_state = nnx.split(new_abstract)

        old_leaves = jax.tree_util.tree_leaves(old_state)
        new_leaves = jax.tree_util.tree_leaves(new_state)
        self.assertEqual(len(old_leaves), len(new_leaves))
        for old, new in zip(old_leaves, new_leaves):
            self.assertEqual(old.shape, new.shape)
            self.assertEqual(old.dtype, new.dtype)


class TestCpuMeshContextNoopUnderTrace(unittest.TestCase):
    """cpu_mesh_context must yield without calling jax.set_mesh inside any
    tracing context where set_mesh is forbidden (jit / eval_shape / vmap /
    grad), otherwise process_weights_after_loading cannot run abstractly."""

    def test_noop_under_eval_shape(self):
        def fn():
            with cpu_mesh_context():
                return jnp.zeros((4, 4))

        out = jax.eval_shape(fn)
        self.assertEqual(out.shape, (4, 4))

    def test_noop_under_jit(self):
        @jax.jit
        def fn(x):
            with cpu_mesh_context():
                return x + 1

        # Crashes if cpu_mesh_context tries to set_mesh inside jit.
        out = fn(jnp.array(1.0))
        self.assertEqual(float(out), 2.0)

    def test_noop_under_vmap(self):
        # Detection must cover non-DynamicJaxpr traces too. Under vmap,
        # jax.core.unsafe_am_i_under_a_jit_DO_NOT_USE returns False but
        # set_mesh still raises — the gating must match set_mesh's own check.
        def fn(x):
            with cpu_mesh_context():
                return x * 2

        out = jax.vmap(fn)(jnp.ones((3,)))
        self.assertEqual(out.shape, (3,))

    def test_noop_under_grad(self):
        def fn(x):
            with cpu_mesh_context():
                return (x * x).sum()

        g = jax.grad(fn)(jnp.ones((3,)))
        self.assertEqual(g.shape, (3,))

    def test_real_context_outside_trace_sets_cpu_mesh(self):
        # Outside any trace, the context should actually set the CPU mesh —
        # not silently take the no-op path.
        cpu = jax.devices("cpu")[0]
        with cpu_mesh_context():
            arr = jnp.zeros((2,))
            # The active mesh becomes a CPU mesh, so allocations land on CPU.
            self.assertIn(cpu, list(arr.sharding.device_set))


if __name__ == "__main__":
    unittest.main()
