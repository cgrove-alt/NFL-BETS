"""Health check endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """
    Health check endpoint for Railway deployment.

    Returns status of all system components.
    """
    app_state = request.app.state.app_state

    # Get component health
    components = app_state.get_health_status()

    # Determine overall health
    is_healthy = components.get("initialized", False)

    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": components,
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> dict[str, Any]:
    """
    Kubernetes-style readiness probe.

    Returns 200 if the service is ready to accept traffic.
    """
    app_state = request.app.state.app_state

    return {
        "ready": app_state.is_initialized,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """
    Kubernetes-style liveness probe.

    Returns 200 if the service is alive (even if not fully ready).
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat(),
    }
