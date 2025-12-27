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


@router.get("/health/status")
async def detailed_status(request: Request) -> dict[str, Any]:
    """
    Detailed health status endpoint for frontend live connection indicator.

    Returns:
        - status: "online" | "degraded" | "offline"
        - bets_in_memory: Number of value bets currently detected
        - last_update: ISO timestamp of last data refresh
        - scheduler_running: Boolean
        - models_loaded: Boolean
    """
    app_state = request.app.state.app_state

    # Get full health status
    components = app_state.get_health_status()

    # Determine status level
    if not components.get("initialized", False):
        status = "offline"
    elif components.get("scheduler_running", False) and components.get("startup_refresh_complete", False):
        status = "online"
    else:
        status = "degraded"

    # Count loaded models
    models_loaded = False
    if app_state.model_manager:
        try:
            # Check if at least spread model is loaded
            models_loaded = app_state.value_detector is not None and \
                           app_state.value_detector.spread_model is not None
        except Exception:
            pass

    return {
        "status": status,
        "bets_in_memory": components.get("bets_in_memory", 0),
        "last_update": components.get("last_data_refresh"),
        "scheduler_running": components.get("scheduler_running", False),
        "models_loaded": models_loaded,
        "startup_complete": components.get("startup_refresh_complete", False),
        "timestamp": datetime.now().isoformat(),
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


@router.get("/debug/settings")
async def debug_settings(request: Request) -> dict[str, Any]:
    """
    Debug endpoint to verify settings are loaded correctly.

    Shows masked API key info to diagnose configuration issues.
    """
    import os

    app_state = request.app.state.app_state
    settings = app_state.settings

    # Get API key from settings
    api_key = ""
    if settings and hasattr(settings, "odds_api"):
        api_key = settings.odds_api.api_key or ""

    # Also check raw environment variables
    env_odds_api_key = os.environ.get("ODDS_API_KEY", "")
    env_odds_api__api_key = os.environ.get("ODDS_API__API_KEY", "")

    return {
        "settings_loaded": settings is not None,
        "odds_api_key_from_settings": {
            "set": bool(api_key),
            "length": len(api_key),
            "preview": api_key[:4] + "..." if api_key and len(api_key) > 4 else "(empty)",
        },
        "env_vars": {
            "ODDS_API_KEY": {
                "set": bool(env_odds_api_key),
                "length": len(env_odds_api_key),
                "preview": env_odds_api_key[:4] + "..." if env_odds_api_key and len(env_odds_api_key) > 4 else "(empty)",
            },
            "ODDS_API__API_KEY": {
                "set": bool(env_odds_api__api_key),
                "length": len(env_odds_api__api_key),
                "preview": env_odds_api__api_key[:4] + "..." if env_odds_api__api_key and len(env_odds_api__api_key) > 4 else "(empty)",
            },
        },
        "timestamp": datetime.now().isoformat(),
    }
