"""
APScheduler orchestrator for managing background jobs.

Handles:
- Odds polling at configurable intervals
- Value detection after odds updates
- Model refresh checks
- Nightly nflverse data sync
- Health monitoring
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

logger = logging.getLogger(__name__)


class SchedulerOrchestrator:
    """
    Manages APScheduler lifecycle and job registration.

    Coordinates all background polling and maintenance jobs:
    - Odds polling: Every N minutes during active hours
    - Value detection: Triggered after odds updates
    - Model refresh: Daily at configured hour
    - Data sync: Nightly nflverse refresh
    - Health check: Every 5 minutes

    Example:
        >>> scheduler = SchedulerOrchestrator(settings, pipeline, ...)
        >>> scheduler.start()
        >>> # ... application runs ...
        >>> scheduler.stop()
    """

    def __init__(
        self,
        settings: Any,
        pipeline: Any,
        feature_pipeline: Any,
        model_manager: Any,
        value_detector: Any,
        bankroll_manager: Any,
    ):
        """
        Initialize the scheduler.

        Args:
            settings: Application settings
            pipeline: DataPipeline instance
            feature_pipeline: FeaturePipeline instance
            model_manager: ModelManager instance
            value_detector: ValueDetector instance
            bankroll_manager: BankrollManager instance
        """
        self.settings = settings
        self.pipeline = pipeline
        self.feature_pipeline = feature_pipeline
        self.model_manager = model_manager
        self.value_detector = value_detector
        self.bankroll_manager = bankroll_manager

        # APScheduler instance
        self.scheduler = AsyncIOScheduler(
            timezone="America/New_York",  # NFL games in Eastern time
            job_defaults={
                "coalesce": True,  # Combine missed runs into one
                "max_instances": 1,  # Only one instance of each job
                "misfire_grace_time": 60,  # 1 minute grace period
            },
        )

        # Job state tracking
        self._job_status: dict[str, dict] = {}
        self._last_value_bets: list = []
        self._is_running = False

        # Register event listeners
        self.scheduler.add_listener(self._on_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._on_job_error, EVENT_JOB_ERROR)

    def start(self) -> None:
        """Start the scheduler with all registered jobs."""
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self._register_jobs()
        self.scheduler.start()
        self._is_running = True

        logger.info("Scheduler started with jobs:")
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            next_str = next_run.strftime("%H:%M:%S") if next_run else "paused"
            logger.info(f"  - {job.id}: next run at {next_str}")

    def stop(self) -> None:
        """Gracefully shutdown the scheduler."""
        if not self._is_running:
            return

        logger.info("Stopping scheduler...")
        self.scheduler.shutdown(wait=True)
        self._is_running = False
        logger.info("Scheduler stopped")

    def _register_jobs(self) -> None:
        """Register all polling and maintenance jobs."""
        sched_settings = self.settings.scheduler

        # Odds polling job - every N minutes during active hours
        self.scheduler.add_job(
            self._poll_odds_job,
            trigger=IntervalTrigger(minutes=sched_settings.odds_poll_interval_minutes),
            id="poll_odds",
            name="Poll Odds API",
            replace_existing=True,
        )

        # Model refresh check - daily at configured hour
        self.scheduler.add_job(
            self._check_model_refresh_job,
            trigger=CronTrigger(hour=sched_settings.model_refresh_hour, minute=0),
            id="model_refresh_check",
            name="Check Model Freshness",
            replace_existing=True,
        )

        # Nflverse data sync - nightly at configured hour
        self.scheduler.add_job(
            self._sync_nflverse_job,
            trigger=CronTrigger(hour=sched_settings.nflfastr_sync_hour, minute=0),
            id="nflverse_sync",
            name="Sync nflverse Data",
            replace_existing=True,
        )

        # Health check - every 5 minutes
        self.scheduler.add_job(
            self._health_check_job,
            trigger=IntervalTrigger(minutes=5),
            id="health_check",
            name="Health Check",
            replace_existing=True,
        )

        # Auto-retrain job - runs 30 min after model refresh check
        if sched_settings.auto_retrain_on_stale:
            self.scheduler.add_job(
                self._auto_retrain_job,
                trigger=CronTrigger(hour=sched_settings.model_refresh_hour, minute=30),
                id="auto_retrain",
                name="Auto-Retrain Stale Models",
                replace_existing=True,
            )

        # Prop polling job - hourly (separate from spreads due to API costs)
        self.scheduler.add_job(
            self._poll_props_job,
            trigger=IntervalTrigger(hours=1),
            id="poll_props",
            name="Poll Player Props",
            replace_existing=True,
        )

        # Initialize job status
        for job in self.scheduler.get_jobs():
            self._job_status[job.id] = {
                "last_run": None,
                "last_status": "pending",
                "last_error": None,
                "run_count": 0,
            }

    async def _poll_odds_job(self) -> None:
        """
        Poll Odds API for latest lines and run value detection.

        This is the main polling job that:
        1. Checks if we're in active hours
        2. Fetches latest odds from The Odds API
        3. Gets upcoming games
        4. Builds features for predictions
        5. Runs value detection
        6. Stores any value bets found
        """
        from nfl_bets.scheduler.jobs import poll_odds

        # Check if in active hours
        now = datetime.now()
        sched = self.settings.scheduler
        if not (sched.active_hours_start <= now.hour < sched.active_hours_end):
            logger.debug(f"Outside active hours ({sched.active_hours_start}-{sched.active_hours_end})")
            return

        # Run the polling job
        value_bets = await poll_odds(
            pipeline=self.pipeline,
            feature_pipeline=self.feature_pipeline,
            value_detector=self.value_detector,
            bankroll_manager=self.bankroll_manager,
        )

        # Store results
        self._last_value_bets = value_bets

        if value_bets:
            logger.info(f"Found {len(value_bets)} value bet(s)")
            for bet in value_bets:
                logger.info(
                    f"  {bet.description}: edge={bet.edge:.1%}, "
                    f"stake=${bet.recommended_stake:.2f}"
                )

    async def _poll_props_job(self) -> None:
        """
        Poll player props hourly (separate from spreads due to API costs).

        Props use ~5 credits per game vs 1 for spreads, so we poll less frequently.
        """
        from nfl_bets.scheduler.jobs import poll_props

        # Check if in active hours
        now = datetime.now()
        sched = self.settings.scheduler
        if not (sched.active_hours_start <= now.hour < sched.active_hours_end):
            logger.debug(f"Outside active hours for props ({sched.active_hours_start}-{sched.active_hours_end})")
            return

        # Run the polling job
        prop_bets = await poll_props(
            pipeline=self.pipeline,
            feature_pipeline=self.feature_pipeline,
            value_detector=self.value_detector,
            bankroll_manager=self.bankroll_manager,
        )

        # Merge with existing value bets (don't overwrite spread bets)
        if prop_bets:
            self._last_value_bets.extend(prop_bets)
            logger.info(f"Found {len(prop_bets)} prop value bet(s)")
            for bet in prop_bets:
                logger.info(
                    f"  {bet.description}: edge={bet.edge:.1%}, "
                    f"stake=${bet.recommended_stake:.2f}"
                )

    async def _check_model_refresh_job(self) -> None:
        """Check model staleness and log warnings if stale."""
        from nfl_bets.scheduler.jobs import check_model_refresh

        await check_model_refresh(model_manager=self.model_manager)

    async def _sync_nflverse_job(self) -> None:
        """Nightly sync of nflverse play-by-play data."""
        from nfl_bets.scheduler.jobs import sync_nflverse

        await sync_nflverse(pipeline=self.pipeline)

    async def _health_check_job(self) -> None:
        """Periodic health monitoring of all data sources."""
        from nfl_bets.scheduler.jobs import health_check

        health = await health_check(pipeline=self.pipeline)

        # Log any degraded sources
        for source, healthy in health.source_status.items():
            if not healthy:
                logger.warning(f"Data source degraded: {source}")

    async def _auto_retrain_job(self) -> None:
        """Auto-retrain models if any are stale."""
        from nfl_bets.scheduler.jobs import trigger_model_retrain

        # Check if any models are stale
        all_fresh, stale_models = self.model_manager.check_all_models_fresh()

        if all_fresh:
            logger.info("All models are fresh, no retraining needed")
            return

        logger.info(f"Stale models detected: {stale_models}")
        logger.info("Starting automatic model retraining...")

        result = await trigger_model_retrain()

        if result.get("status") == "success":
            logger.info("Auto-retrain completed successfully")
            # Reload models after retraining
            try:
                spread_model = self.model_manager.load_spread_model()
                self.value_detector.spread_model = spread_model
                logger.info("Spread model reloaded")
            except Exception as e:
                logger.error(f"Failed to reload model after retrain: {e}")
        else:
            logger.error(f"Auto-retrain failed: {result.get('error')}")

    def _on_job_executed(self, event: JobExecutionEvent) -> None:
        """Handle successful job execution."""
        job_id = event.job_id
        if job_id in self._job_status:
            self._job_status[job_id]["last_run"] = datetime.now()
            self._job_status[job_id]["last_status"] = "success"
            self._job_status[job_id]["run_count"] += 1

    def _on_job_error(self, event: JobExecutionEvent) -> None:
        """Handle job execution error."""
        job_id = event.job_id
        if job_id in self._job_status:
            self._job_status[job_id]["last_run"] = datetime.now()
            self._job_status[job_id]["last_status"] = "error"
            self._job_status[job_id]["last_error"] = str(event.exception)
            self._job_status[job_id]["run_count"] += 1

        logger.error(f"Job {job_id} failed: {event.exception}")

    def get_job_status(self) -> dict[str, dict]:
        """Get status of all jobs."""
        status = {}
        for job in self.scheduler.get_jobs():
            job_info = self._job_status.get(job.id, {})
            status[job.id] = {
                "name": job.name,
                "next_run": job.next_run_time,
                "last_run": job_info.get("last_run"),
                "last_status": job_info.get("last_status", "pending"),
                "run_count": job_info.get("run_count", 0),
            }
        return status

    def get_last_value_bets(self) -> list:
        """Get value bets from most recent poll."""
        return self._last_value_bets

    def trigger_job(self, job_id: str) -> bool:
        """
        Manually trigger a job to run immediately.

        Args:
            job_id: ID of job to trigger

        Returns:
            True if job was triggered
        """
        job = self.scheduler.get_job(job_id)
        if job:
            self.scheduler.modify_job(job_id, next_run_time=datetime.now())
            logger.info(f"Triggered job: {job_id}")
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self.scheduler.get_job(job_id)
        if job:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self.scheduler.get_job(job_id)
        if job:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        return False

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running
