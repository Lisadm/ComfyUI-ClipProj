import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'owned_clipproj_projection', Path(__file__).resolve().parents[1] / 'clipproj_projection.py')
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


@pytest.mark.parametrize('shape', [(1, 256), (257, 256), (1, 1025, 256), (2, 65, 256)])
def test_fp16_native_parallel_is_bit_exact_and_restores_threads(shape):
    previous = torch.get_num_threads()
    torch.manual_seed(1003)
    reference = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(),
                                    torch.nn.Linear(1024, 512)).half()
    actual = projection.CPUParallelResidual(*reference.children())
    try:
        torch.set_num_threads(1)
        with torch.inference_mode():
            x = torch.randn(shape).half()
            expected = reference(x)
            result = actual(x)
        assert torch.equal(result.view(torch.uint16), expected.view(torch.uint16))
        assert torch.get_num_threads() == 1
        assert result.shape == (*shape[:-1], 512)
    finally:
        torch.set_num_threads(previous)


def test_restores_threads_after_failure():
    previous = torch.get_num_threads()
    model = projection.CPUParallelResidual(torch.nn.Linear(3, 2).half())
    try:
        torch.set_num_threads(1)
        with pytest.raises(RuntimeError):
            model(torch.ones(4, 5, dtype=torch.float16))
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_float32_keeps_cpu_policy():
    model = projection.CPUParallelResidual(torch.nn.Linear(3, 2))
    with patch.object(torch, 'set_num_threads') as setter:
        model(torch.ones(4, 3))
    setter.assert_not_called()


def test_existing_higher_thread_budget_is_preserved():
    model = projection.CPUParallelResidual(torch.nn.Identity())
    with patch.object(torch, 'get_num_threads', return_value=32), \
            patch.object(torch, 'set_num_threads') as setter:
        model(torch.ones(4, 3, dtype=torch.float16))
    setter.assert_not_called()


def test_build_residual_preserves_weights_dtype_and_state_keys():
    checkpoint = {'mlp.0.weight': torch.randn(64, 32).half(),
                  'mlp.0.bias': torch.randn(64).half(),
                  'mlp.2.weight': torch.randn(16, 64).half(),
                  'mlp.2.bias': torch.randn(16).half()}
    model = projection.build_residual(checkpoint, 'cpu')
    assert isinstance(model, projection.CPUParallelResidual)
    assert set(model.state_dict()) == {key.removeprefix('mlp.') for key in checkpoint}
    for key, value in model.state_dict().items():
        assert value.dtype == torch.float16
        assert torch.equal(value.view(torch.uint16), checkpoint['mlp.' + key].view(torch.uint16))


def _load_clipproj_nodes_module():
    import sys
    import types
    pkg_dir = Path(__file__).resolve().parents[1]
    pkg_name = "owned_clipproj_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pkg_dir)]
    sys.modules[pkg_name] = pkg
    for sub in ("clipproj_pinning", "clipproj_projection", "clipproj_nodes"):
        s = importlib.util.spec_from_file_location(f"{pkg_name}.{sub}", pkg_dir / f"{sub}.py")
        m = importlib.util.module_from_spec(s)
        m.__package__ = pkg_name
        sys.modules[f"{pkg_name}.{sub}"] = m
        s.loader.exec_module(m)
    return sys.modules[f"{pkg_name}.clipproj_nodes"]


def test_projected_clip_truncates_layers_at_tap_and_strips_vision_tokens(monkeypatch):
    from types import SimpleNamespace
    nodes_mod = _load_clipproj_nodes_module()

    executed_layers = []
    norm_called = []

    class DummyLayer(torch.nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx

        def forward(self, x):
            executed_layers.append(self.idx)
            return x + 1.0

    class DummyNorm(torch.nn.Module):
        def forward(self, x):
            norm_called.append(True)
            return x * 2.0

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DummyLayer(i) for i in range(36)])
            self.norm = DummyNorm()

    class DummyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyModel()

        def preprocess_embed(self, embed, device):
            return torch.zeros(4, 16), {"grid": torch.tensor([[1, 2, 2]]), "deepstack": []}

        def forward(self, *args, **kwargs):
            x = torch.zeros(1, 10, 16)
            for layer in self.model.layers:
                x = layer(x)
            if self.model.norm is not None:
                x = self.model.norm(x)
            return x, x.unsqueeze(1)

    tr = DummyTransformer()

    class DummySM:
        def __init__(self):
            self.transformer = tr
            self.layer = "last"
            self.layer_idx = None
            self.layer_norm_hidden_state = False
            self.execution_device = None

        def encode_token_weights(self, pairs):
            embeds_info = [{"type": "image", "index": 2, "size": 4}]
            _, inter = self.transformer(pairs, embeds_info=embeds_info)
            return inter, None

    sm = DummySM()
    base = SimpleNamespace(
        cond_stage_model=SimpleNamespace(clip="qwen3vl_4b", qwen3vl_4b=sm),
        tokenizer=SimpleNamespace(clip_name="qwen3vl_4b", qwen3vl_4b=SimpleNamespace(
            tokenizer=lambda s, add_special_tokens=False: {"input_ids": [10, 11, 12]})),
        patcher=SimpleNamespace(load_device=torch.device("cpu")),
        add_hooks_to_dict=lambda d: None,
    )
    fake_proj = {
        "tap": 24,
        "mean_in": torch.zeros(16),
        "std_in": torch.ones(16),
        "mean_out": torch.zeros(32),
        "std_out": torch.ones(32),
        "W": torch.ones(16, 32),
        "sink_out": torch.full((32,), 7.0),
    }
    monkeypatch.setattr(nodes_mod, "load_projection", lambda name: fake_proj)
    monkeypatch.setattr(nodes_mod.mm, "load_models_gpu", lambda models: None)
    monkeypatch.setattr(nodes_mod.mm, "intermediate_device", lambda: torch.device("cpu"))

    pclip = nodes_mod.ProjectedCLIP(base, "dummy.safetensors", vision_tokens="text_only (fast)")
    tokens = pclip.tokenize("hello", images=[torch.zeros(1, 64, 64, 3)])
    vision_entry = [e[0] for e in tokens["qwen3vl_4b"][0] if isinstance(e[0], dict)][0]
    assert vision_entry["clipproj_max_pixels"] == nodes_mod.DEFAULT_VISION_MAX_PIXELS

    cond, tags = pclip._encode(tokens)
    # Only layers 0..23 ran (24 layers instead of 36), and final norm was skipped
    assert executed_layers == list(range(24))
    assert not norm_called
    # Original 36 layers and norm are restored after _encode
    assert len(tr.model.layers) == 36
    assert tr.model.norm is not None
    # Vision span (indices 1..6 out of 10) stripped -> 4 text tokens remain, all tagged 1
    assert cond.shape == (1, 4, 32)
    assert torch.equal(tags, torch.ones(4, dtype=torch.long))
    assert torch.allclose(cond[0, 0], torch.full((32,), 7.0))

    # In 'full' mode, all 10 tokens (including 6 vision-tagged tokens) are preserved
    executed_layers.clear()
    pclip_full = nodes_mod.ProjectedCLIP(base, "dummy.safetensors", vision_tokens="full")
    cond_full, tags_full = pclip_full._encode(pclip_full.tokenize("hello", images=[torch.zeros(1, 64, 64, 3)]))
    assert executed_layers == list(range(24))
    assert cond_full.shape == (1, 10, 32)
    assert int((tags_full == 0).sum()) == 6
    assert int((tags_full == 1).sum()) == 4


def test_cuda_projector_preserves_reference_fp16_linear_and_gelu():
    import sys
    if (sys.platform != "win32" or torch.version.git_version != "cf30153c4c131c8164ee7798e5022d810682e2cb"
            or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)):
        pytest.skip("Requires the verified Windows CUDA projector build")
    pytest.importorskip("triton")
    source = Path(__file__).resolve().parents[1] / "clipproj_legacy_cuda.py"
    spec = importlib.util.spec_from_file_location("owned_clipproj_legacy_cuda_test", source)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernel
    spec.loader.exec_module(kernel)
    rng = torch.Generator().manual_seed(1003)
    half_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    lookup = torch.nn.functional.gelu(half_bits).cuda()
    # The real GELU width is vector-aligned. Exercise masked columns separately.
    for features, outputs, table in ((2560, 64, lookup), (32768, 67, None)):
        x = torch.randn(19, features, generator=rng).half()
        weight = (torch.randn(outputs, features, generator=rng) * 0.05).half()
        bias = torch.randn(outputs, generator=rng).half()
        expected = torch.nn.functional.linear(x, weight, bias)
        if table is not None:
            expected = torch.nn.functional.gelu(expected)
        actual = kernel._linear(x.cuda(), weight.cuda(), bias.cuda(), table).cpu()
        assert torch.equal(expected.view(torch.uint16), actual.view(torch.uint16))
