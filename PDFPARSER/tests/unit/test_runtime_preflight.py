import pytest

from app.llm.runtime_preflight import evaluate_runtime


def snapshot(**overrides):  # type: ignore[no-untyped-def]
    value = {
        "python": "3.12.0",
        "platform": "Linux",
        "packages": {
            "torch": "2.6.0",
            "transformers": "4.50.0",
            "accelerate": "1.4.0",
            "bitsandbytes": "0.46.0",
        },
        "system_ram_gib": 50.0,
        "disk_free_gib": 100.0,
        "nvidia_smi": {"available": True, "output": "T4", "error": None},
        "gpu": {
            "cuda_available": True,
            "cuda_version": "12.4",
            "device_name": "Tesla T4",
            "compute_capability": [7, 5],
            "vram_gib": 15.0,
            "bf16_supported": False,
            "nf4_smoke_ok": True,
            "nf4_smoke_error": None,
        },
    }
    value.update(overrides)
    return value


def test_qwen14b_runtime_is_eligible_without_automatic_fallback() -> None:
    report = evaluate_runtime("qwen14b", snapshot())

    assert report["decision"] == "supported_for_load_test"
    assert report["model_alias"] == "qwen14b"
    assert report["compute_dtype"] == "float16"
    assert report["reasons"] == []


def test_missing_cuda_blocks_preflight_with_explicit_reasons() -> None:
    runtime = snapshot()
    runtime["gpu"] = {
        "cuda_available": False,
        "compute_capability": None,
        "vram_gib": None,
        "bf16_supported": False,
        "nf4_smoke_ok": False,
    }

    report = evaluate_runtime("qwen14b", runtime)

    assert report["decision"] == "blocked"
    assert "CUDA is not available" in report["reasons"]
    assert "bitsandbytes NF4 smoke test failed" in report["reasons"]
    assert report["model_alias"] == "qwen14b"


def test_insufficient_vram_does_not_silently_select_qwen7b() -> None:
    runtime = snapshot()
    runtime["gpu"] = {**runtime["gpu"], "vram_gib": 10.0}

    report = evaluate_runtime("qwen14b", runtime)

    assert report["decision"] == "blocked"
    assert report["model_alias"] == "qwen14b"
    assert "VRAM is below 14 GiB" in report["reasons"]


def test_bf16_is_selected_only_when_runtime_reports_support() -> None:
    runtime = snapshot()
    runtime["gpu"] = {**runtime["gpu"], "bf16_supported": True}

    assert evaluate_runtime("qwen14b", runtime)["compute_dtype"] == "bfloat16"


def test_unknown_model_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model alias"):
        evaluate_runtime("arbitrary-model", snapshot())
