"""Model status endpoints."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information response."""

    model_type: str
    model_version: Optional[str] = None
    data_cutoff_date: Optional[str] = None
    training_date: Optional[str] = None
    is_stale: bool
    error: Optional[str] = None
    metrics: Optional[dict] = None


class ModelsStatusResponse(BaseModel):
    """Response for all models status."""

    all_fresh: bool
    stale_models: list[str]
    models: dict[str, ModelInfo]
    checked_at: str


@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status(request: Request) -> dict[str, Any]:
    """
    Get status of all ML models.

    Returns freshness status and metadata for each model.
    """
    app_state = request.app.state.app_state

    if not app_state.model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    model_manager = app_state.model_manager

    # Check all models
    all_fresh, stale_models = model_manager.check_all_models_fresh()

    # Get detailed info for each model
    model_types = ["spread", "passing_yards", "rushing_yards", "receiving_yards"]
    models_info = {}

    for model_type in model_types:
        try:
            info = model_manager.get_model_info(model_type)

            models_info[model_type] = ModelInfo(
                model_type=model_type,
                model_version=info.get("model_version"),
                data_cutoff_date=info.get("data_cutoff_date"),
                training_date=info.get("training_date"),
                is_stale=info.get("is_stale", True),
                error=info.get("error"),
                metrics=info.get("metrics"),
            )
        except Exception as e:
            models_info[model_type] = ModelInfo(
                model_type=model_type,
                is_stale=True,
                error=str(e),
            )

    return {
        "all_fresh": all_fresh,
        "stale_models": stale_models,
        "models": models_info,
        "checked_at": datetime.now().isoformat(),
    }


@router.get("/models/{model_type}")
async def get_model_info(request: Request, model_type: str) -> dict[str, Any]:
    """
    Get detailed information for a specific model.

    Args:
        model_type: Type of model (spread, passing_yards, etc.)
    """
    app_state = request.app.state.app_state

    if not app_state.model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    valid_types = ["spread", "passing_yards", "rushing_yards", "receiving_yards"]
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {valid_types}",
        )

    try:
        info = app_state.model_manager.get_model_info(model_type)
        return {
            "model_type": model_type,
            **info,
            "checked_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/retrain")
async def trigger_retrain(request: Request) -> dict[str, Any]:
    """
    Trigger model retraining.

    This runs the training script in the background.
    Use with caution as training can take several minutes.
    """
    app_state = request.app.state.app_state

    if not app_state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    # Trigger the retrain job
    from nfl_bets.scheduler.jobs import trigger_model_retrain

    try:
        # Run async
        import asyncio
        result = await trigger_model_retrain()

        return {
            "success": result.get("status") == "success",
            "message": result.get("message", "Retraining triggered"),
            "status": result.get("status"),
            "error": result.get("error"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
