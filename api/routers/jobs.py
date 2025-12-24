"""Scheduler job control endpoints."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class JobStatus(BaseModel):
    """Status of a scheduled job."""

    job_id: str
    name: str
    last_status: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    error: Optional[str] = None


class JobsStatusResponse(BaseModel):
    """Response for all jobs status."""

    scheduler_running: bool
    jobs: list[JobStatus]


@router.get("/jobs/status", response_model=JobsStatusResponse)
async def get_jobs_status(request: Request) -> dict[str, Any]:
    """
    Get status of all scheduled jobs.

    Returns last run time, next run time, and status for each job.
    """
    app_state = request.app.state.app_state

    if not app_state.scheduler:
        return {
            "scheduler_running": False,
            "jobs": [],
        }

    scheduler = app_state.scheduler
    job_status = scheduler.get_job_status()

    jobs = []
    for job_id, status in job_status.items():
        jobs.append(
            JobStatus(
                job_id=job_id,
                name=status.get("name", job_id),
                last_status=status.get("last_status", "pending"),
                last_run=status.get("last_run").isoformat() if status.get("last_run") else None,
                next_run=status.get("next_run").isoformat() if status.get("next_run") else None,
                error=status.get("error"),
            )
        )

    return {
        "scheduler_running": scheduler.is_running,
        "jobs": jobs,
    }


@router.post("/jobs/{job_id}/trigger")
async def trigger_job(request: Request, job_id: str) -> dict[str, Any]:
    """
    Manually trigger a job to run immediately.

    Args:
        job_id: ID of the job to trigger (poll_odds, model_refresh, etc.)
    """
    app_state = request.app.state.app_state

    if not app_state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    scheduler = app_state.scheduler

    # Valid job IDs
    valid_jobs = ["poll_odds", "model_refresh", "nflverse_sync", "health_check"]
    if job_id not in valid_jobs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job ID. Must be one of: {valid_jobs}",
        )

    try:
        scheduler.trigger_job(job_id)
        return {
            "success": True,
            "job_id": job_id,
            "message": f"Job '{job_id}' triggered successfully",
            "triggered_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/pause")
async def pause_scheduler(request: Request) -> dict[str, Any]:
    """
    Pause all scheduled jobs.

    Jobs will not run until resumed.
    """
    app_state = request.app.state.app_state

    if not app_state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        app_state.scheduler.pause()
        return {
            "success": True,
            "message": "Scheduler paused",
            "paused_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/resume")
async def resume_scheduler(request: Request) -> dict[str, Any]:
    """
    Resume scheduled jobs after pausing.
    """
    app_state = request.app.state.app_state

    if not app_state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    try:
        app_state.scheduler.resume()
        return {
            "success": True,
            "message": "Scheduler resumed",
            "resumed_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
