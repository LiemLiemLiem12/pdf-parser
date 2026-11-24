from typing import Any

import pytest

from app.core.config import Settings
from app.llm.model_manager import ModelLoadError, ModelManager, ModelManagerState


class FakeCuda:
    def __init__(self, *, available: bool = True, bf16: bool = False) -> None:
        self.available = available
        self.bf16 = bf16
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    float16 = "float16-object"
    bfloat16 = "bfloat16-object"

    def __init__(self, *, available: bool = True, bf16: bool = False) -> None:
        self.cuda = FakeCuda(available=available, bf16=bf16)


class FakeQuantization:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs


class FakeTokenizerClass:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
        cls.calls.append((model_id, kwargs))
        return object()


class FakeModel:
    def __init__(self, device_map: dict[str, object]) -> None:
        self.hf_device_map = device_map
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class FakeModelClass:
    calls: list[tuple[str, dict[str, Any]]] = []
    device_map: dict[str, object] = {"": 0}
    instance: FakeModel | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
        cls.calls.append((model_id, kwargs))
        cls.instance = FakeModel(cls.device_map)
        return cls.instance


def real_settings() -> Settings:
    return Settings(
        _env_file=None,
        inference_mode="transformers",
        model_alias="qwen14b",
        model_revision="test-revision",
        quantization="nf4-4bit",
    )


def dependencies(torch: FakeTorch):  # type: ignore[no-untyped-def]
    return lambda: (torch, FakeModelClass, FakeTokenizerClass, FakeQuantization)


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    FakeModelClass.calls = []
    FakeModelClass.device_map = {"": 0}
    FakeModelClass.instance = None
    FakeTokenizerClass.calls = []
    FakeQuantization.last_kwargs = None


def test_loads_once_and_captures_reproducibility_metadata() -> None:
    torch = FakeTorch(bf16=False)
    manager = ModelManager(real_settings(), dependency_loader=dependencies(torch))

    manager.load()
    manager.load()

    assert manager.state is ModelManagerState.READY
    assert len(FakeModelClass.calls) == 1
    assert len(FakeTokenizerClass.calls) == 1
    assert FakeModelClass.instance is not None
    assert FakeModelClass.instance.eval_calls == 1
    assert FakeQuantization.last_kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16-object",
    }
    assert manager.metadata == {
        "alias": "qwen14b",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "test-revision",
        "quantization": "nf4-4bit",
        "compute_dtype": "float16",
        "device_map": {"": "0"},
    }


def test_uses_bfloat16_only_when_cuda_reports_support() -> None:
    manager = ModelManager(
        real_settings(), dependency_loader=dependencies(FakeTorch(bf16=True))
    )

    manager.load()

    assert manager.metadata is not None
    assert manager.metadata["compute_dtype"] == "bfloat16"


def test_rejects_cpu_or_disk_offload_and_does_not_retry() -> None:
    FakeModelClass.device_map = {"model.embed_tokens": 0, "lm_head": "cpu"}
    manager = ModelManager(
        real_settings(), dependency_loader=dependencies(FakeTorch())
    )

    with pytest.raises(ModelLoadError, match="GPU-only baseline rejected"):
        manager.load()

    assert manager.state is ModelManagerState.FAILED
    assert manager.ready is False
    with pytest.raises(ModelLoadError, match="cannot run"):
        manager.load()
    assert len(FakeModelClass.calls) == 1


def test_cuda_is_required_before_checkpoint_download() -> None:
    manager = ModelManager(
        real_settings(), dependency_loader=dependencies(FakeTorch(available=False))
    )

    with pytest.raises(ModelLoadError, match="CUDA is unavailable"):
        manager.load()

    assert FakeTokenizerClass.calls == []
    assert FakeModelClass.calls == []


def test_close_releases_references_and_gpu_cache() -> None:
    torch = FakeTorch()
    manager = ModelManager(real_settings(), dependency_loader=dependencies(torch))
    manager.load()

    manager.close()

    assert manager.state is ModelManagerState.CLOSED
    assert torch.cuda.empty_cache_calls == 1
    with pytest.raises(ModelLoadError, match="not ready"):
        _ = manager.model
