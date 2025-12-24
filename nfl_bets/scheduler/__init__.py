"""
Job scheduling module.

Provides APScheduler-based background job orchestration for:
- Odds polling at configurable intervals
- Value detection after odds updates
- Model refresh checks
- Nightly nflverse data sync
- Health monitoring

Example:
    >>> from nfl_bets.scheduler import SchedulerOrchestrator
    >>>
    >>> scheduler = SchedulerOrchestrator(settings, pipeline, ...)
    >>> scheduler.start()
    >>>
    >>> # Check job status
    >>> status = scheduler.get_job_status()
    >>> print(status)
    >>>
    >>> # Manual trigger
    >>> scheduler.trigger_job("poll_odds")
    >>>
    >>> # Graceful shutdown
    >>> scheduler.stop()
"""

from .orchestrator import SchedulerOrchestrator
from .jobs import (
    poll_odds,
    check_model_refresh,
    sync_nflverse,
    health_check,
    trigger_model_retrain,
)

__all__ = [
    "SchedulerOrchestrator",
    "poll_odds",
    "check_model_refresh",
    "sync_nflverse",
    "health_check",
    "trigger_model_retrain",
]
