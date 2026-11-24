"""Process liveness and model readiness endpoints."""

from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.core.dependencies import RuntimeState, get_runtime_state, get_settings
from app.core.exceptions import ModelNotReadyError


router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready")
def ready(
    settings: Settings = Depends(get_settings),
    runtime: RuntimeState = Depends(get_runtime_state),
) -> dict[str, str]:
    if not runtime.ready:
        raise ModelNotReadyError(runtime.reason)
    return {
        "status": "ready",
        "model": runtime.model_id or settings.resolved_model_id,
        "inference_mode": settings.inference_mode.value,
    }
