"""Google Colab GPU preflight without loading model checkpoints."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


# Conservative gates for deciding whether a model load test may be attempted.
# Passing these gates is not evidence that the complete checkpoint will load.
MODEL_RESOURCE_GATES: dict[str, dict[str, float]] = {
    "qwen7b": {"min_vram_gib": 8.0, "min_ram_gib": 8.0, "min_disk_gib": 20.0},
    "qwen14b": {"min_vram_gib": 14.0, "min_ram_gib": 12.0, "min_disk_gib": 35.0},
    "qwen72b": {"min_vram_gib": 48.0, "min_ram_gib": 32.0, "min_disk_gib": 150.0},
}

_PACKAGE_NAMES = ("torch", "transformers", "accelerate", "bitsandbytes")
_GIB = 1024**3


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _system_ram_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) * 1024 / _GIB, 2)
    return None


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "output": None, "error": "nvidia-smi not found"}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {"available": True, "output": completed.stdout.strip(), "error": None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "output": None, "error": type(exc).__name__}


def _torch_gpu_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "compute_capability": None,
        "vram_gib": None,
        "bf16_supported": False,
        "nf4_smoke_ok": False,
        "nf4_smoke_error": None,
    }
    try:
        import torch
    except (ImportError, OSError) as exc:
        result["nf4_smoke_error"] = f"torch unavailable: {type(exc).__name__}"
        return result

    result["cuda_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        result["nf4_smoke_error"] = "CUDA is not available"
        return result

    properties = torch.cuda.get_device_properties(0)
    result.update(
        {
            "cuda_available": True,
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "vram_gib": round(properties.total_memory / _GIB, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    )

    compute_dtype = torch.bfloat16 if result["bf16_supported"] else torch.float16
    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(
            64,
            64,
            bias=False,
            compute_dtype=compute_dtype,
            quant_type="nf4",
            compress_statistics=True,
        ).to("cuda")
        sample = torch.zeros((1, 64), dtype=compute_dtype, device="cuda")
        with torch.inference_mode():
            output = layer(sample)
        torch.cuda.synchronize()
        result["nf4_smoke_ok"] = output.shape == (1, 64)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        result["nf4_smoke_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    finally:
        torch.cuda.empty_cache()
    return result


def collect_runtime_snapshot(*, disk_path: Path | str = "/content") -> dict[str, Any]:
    """Collect non-secret runtime facts without downloading a model."""

    path = Path(disk_path)
    if not path.exists():
        path = Path.cwd()
    disk = shutil.disk_usage(path)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "system_ram_gib": _system_ram_gib(),
        "disk_free_gib": round(disk.free / _GIB, 2),
        "nvidia_smi": _nvidia_smi(),
        "gpu": _torch_gpu_probe(),
    }


def evaluate_runtime(model_alias: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit load-test eligibility decision for one allowlisted alias."""

    if model_alias not in MODEL_RESOURCE_GATES:
        allowed = ", ".join(sorted(MODEL_RESOURCE_GATES))
        raise ValueError(f"Unknown model alias {model_alias!r}; expected one of: {allowed}")

    gates = MODEL_RESOURCE_GATES[model_alias]
    gpu = snapshot.get("gpu") or {}
    packages = snapshot.get("packages") or {}
    reasons: list[str] = []

    if not gpu.get("cuda_available"):
        reasons.append("CUDA is not available")
    capability = gpu.get("compute_capability")
    if capability is None or tuple(capability) < (6, 0):
        reasons.append("GPU compute capability is below the NF4 minimum of 6.0")
    if not gpu.get("nf4_smoke_ok"):
        reasons.append("bitsandbytes NF4 smoke test failed")

    vram_gib = gpu.get("vram_gib")
    if vram_gib is None or vram_gib < gates["min_vram_gib"]:
        reasons.append(f"VRAM is below {gates['min_vram_gib']:.0f} GiB")
    ram_gib = snapshot.get("system_ram_gib")
    if ram_gib is None or ram_gib < gates["min_ram_gib"]:
        reasons.append(f"system RAM is below {gates['min_ram_gib']:.0f} GiB")
    disk_gib = snapshot.get("disk_free_gib")
    if disk_gib is None or disk_gib < gates["min_disk_gib"]:
        reasons.append(f"free disk is below {gates['min_disk_gib']:.0f} GiB")

    for package_name in _PACKAGE_NAMES:
        if packages.get(package_name) is None:
            reasons.append(f"required package is missing: {package_name}")

    return {
        "model_alias": model_alias,
        "decision": "supported_for_load_test" if not reasons else "blocked",
        "compute_dtype": "bfloat16" if gpu.get("bf16_supported") else "float16",
        "resource_gates": gates,
        "reasons": reasons,
        "snapshot": snapshot,
        "notice": (
            "Passing preflight permits RP-011 model-load testing; it does not prove that the "
            "checkpoint will fit. No fallback model is selected automatically."
        ),
    }


def run_preflight(
    model_alias: str,
    *,
    output_path: Path | str | None = None,
    disk_path: Path | str = "/content",
) -> dict[str, Any]:
    report = evaluate_runtime(model_alias, collect_runtime_snapshot(disk_path=disk_path))
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report
