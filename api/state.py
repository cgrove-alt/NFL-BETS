"""
Application state management for FastAPI.

Holds shared state across the application:
- Data pipeline
- Model manager
- Value detector
- Bankroll manager
- Scheduler orchestrator
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AppState:
    """
    Centralized application state.

    Initializes and manages lifecycle of all major components.
    """

    def __init__(self):
        self.settings = None
        self.pipeline = None
        self.feature_pipeline = None
        self.model_manager = None
        self.value_detector = None
        self.bankroll_manager = None
        self.scheduler = None
        self._initialized = False
        self._last_value_bets: list = []
        self._init_error: str = None  # Track initialization error

    async def initialize(self) -> None:
        """Initialize all application components."""
        if self._initialized:
            return

        try:
            # Import here to avoid circular imports
            from nfl_bets.config.settings import get_settings
            from nfl_bets.data.pipeline import DataPipeline
            from nfl_bets.features.feature_pipeline import FeaturePipeline
            from nfl_bets.models.model_manager import ModelManager
            from nfl_bets.betting.value_detector import ValueDetector
            from nfl_bets.tracking.bankroll_manager import BankrollManager
            from nfl_bets.scheduler.orchestrator import SchedulerOrchestrator

            # Load settings
            self.settings = get_settings()
            logger.info("Settings loaded")

            # Initialize data pipeline
            self.pipeline = DataPipeline.from_settings(self.settings)
            await self.pipeline.initialize()
            logger.info("Data pipeline initialized")

            # Initialize feature pipeline
            self.feature_pipeline = FeaturePipeline(self.pipeline)
            logger.info("Feature pipeline initialized")

            # Initialize model manager
            self.model_manager = ModelManager()
            logger.info("Model manager initialized")

            # Load spread model for predictions
            try:
                spread_model = self.model_manager.load_spread_model()
                logger.info("Spread model loaded successfully")
            except FileNotFoundError as e:
                logger.warning(f"Spread model not found: {e}")
                spread_model = None

            # Initialize value detector with spread model
            self.value_detector = ValueDetector(
                spread_model=spread_model,
                min_edge=float(self.settings.value_detection.min_edge_threshold),
            )
            logger.info("Value detector initialized")

            # Initialize bankroll manager
            self.bankroll_manager = BankrollManager(
                initial_bankroll=float(self.settings.initial_bankroll),
            )
            logger.info("Bankroll manager initialized")

            # Initialize scheduler (but don't start it yet)
            self.scheduler = SchedulerOrchestrator(
                settings=self.settings,
                pipeline=self.pipeline,
                feature_pipeline=self.feature_pipeline,
                model_manager=self.model_manager,
                value_detector=self.value_detector,
                bankroll_manager=self.bankroll_manager,
            )
            logger.info("Scheduler orchestrator initialized")

            # Start scheduler
            self.scheduler.start()
            logger.info("Scheduler started")

            self._initialized = True
            logger.info("All components initialized successfully")

        except Exception as e:
            import traceback
            self._init_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Failed to initialize components: {self._init_error}")
            # Don't raise - allow API to start even if some components fail
            self._initialized = False

    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        if self.scheduler:
            try:
                self.scheduler.stop()
                logger.info("Scheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if all components are initialized."""
        return self._initialized

    @property
    def last_value_bets(self) -> list:
        """Get last detected value bets from scheduler."""
        if self.scheduler:
            return self.scheduler.get_last_value_bets()
        return self._last_value_bets

    def get_health_status(self) -> dict:
        """Get health status of all components."""
        status = {
            "initialized": self._initialized,
            "settings": self.settings is not None,
            "pipeline": self.pipeline is not None,
            "feature_pipeline": self.feature_pipeline is not None,
            "model_manager": self.model_manager is not None,
            "value_detector": self.value_detector is not None,
            "bankroll_manager": self.bankroll_manager is not None,
            "scheduler": self.scheduler is not None,
            "scheduler_running": self.scheduler.is_running if self.scheduler else False,
        }
        if self._init_error:
            status["init_error"] = self._init_error
        return status
