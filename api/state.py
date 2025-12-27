"""
Application state management for FastAPI.

Holds shared state across the application:
- Data pipeline
- Model manager
- Value detector
- Bankroll manager
- Scheduler orchestrator

CRITICAL: Uses ONLY REAL DATA from:
- nflverse (schedules, play-by-play, rosters)
- The Odds API (live betting lines)
- ESPN (injuries, depth charts)

NO FAKE DATA: All mock/demo/fallback data has been removed.
If no value bets are detected, the system returns empty - this is valid
and means the betting market is efficient (no exploitable edges found).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache file path for instant load from stale data
CACHE_DIR = Path(__file__).parent.parent / "cache"
LAST_RUN_CACHE = CACHE_DIR / "last_run.json"


class AppState:
    """
    Centralized application state.

    Initializes and manages lifecycle of all major components.
    Uses ONLY REAL DATA - no mock/demo/fake data.
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
        self._last_data_refresh: datetime = None  # Track when data was last refreshed
        self._startup_refresh_complete: bool = False  # Track if instant start completed
        self._is_initializing: bool = True  # Track if we're still in cold start
        self._using_fallback: bool = False  # Legacy flag - always False now

    def _load_cached_data(self) -> bool:
        """
        Load cached data from last run for instant display.

        Returns:
            True if cache loaded successfully, False otherwise
        """
        try:
            if not LAST_RUN_CACHE.exists():
                logger.info("No cache file found - will wait for fresh data")
                return False

            with open(LAST_RUN_CACHE, "r") as f:
                cache_data = json.load(f)

            # Check cache age (stale after 6 hours)
            cached_at = datetime.fromisoformat(cache_data.get("cached_at", "2000-01-01"))
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600

            if age_hours > 6:
                logger.info(f"Cache is {age_hours:.1f}h old - too stale, skipping")
                return False

            # Load cached bets (minimal format)
            cached_bets = cache_data.get("value_bets", [])
            if cached_bets:
                logger.info(f"📦 Loaded {len(cached_bets)} bets from cache ({age_hours:.1f}h old)")
                # Note: These are dicts, not ValueBet objects - router handles conversion
                self._cached_bets_raw = cached_bets
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save current value bets to cache for future instant loads."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Convert value bets to serializable format
            bets_data = []
            for bet in self._last_value_bets:
                bets_data.append({
                    "game_id": getattr(bet, "game_id", ""),
                    "bet_type": getattr(bet, "bet_type", ""),
                    "description": getattr(bet, "description", ""),
                    "edge": getattr(bet, "edge", 0),
                    "model_probability": getattr(bet, "model_probability", 0),
                    "bookmaker": getattr(bet, "bookmaker", ""),
                    "odds": getattr(bet, "odds", 0),
                    "line": getattr(bet, "line", 0),
                })

            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "value_bets": bets_data,
            }

            with open(LAST_RUN_CACHE, "w") as f:
                json.dump(cache_data, f)

            logger.info(f"💾 Saved {len(bets_data)} bets to cache")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    async def initialize(self) -> None:
        """Initialize all application components with REAL DATA only."""
        if self._initialized:
            return

        # Try to load cached data for instant display
        cache_loaded = self._load_cached_data()
        if cache_loaded:
            logger.info("📦 Using cached data while fresh data loads...")

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

            # Initialize feature pipeline with ESPN client for injury integration
            espn_client = getattr(self.pipeline, 'espn', None)
            self.feature_pipeline = FeaturePipeline(
                data_pipeline=self.pipeline,
                espn_client=espn_client,
            )
            logger.info(f"Feature pipeline initialized (ESPN client: {espn_client is not None})")

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

            # Load prop models for player prop predictions
            prop_models = {}
            prop_types = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
            for prop_type in prop_types:
                try:
                    model = self.model_manager.load_prop_model(prop_type)
                    prop_models[prop_type] = model
                    logger.info(f"Prop model loaded: {prop_type}")
                except FileNotFoundError:
                    logger.warning(f"Prop model not found: {prop_type}")

            # Initialize value detector with spread and prop models
            self.value_detector = ValueDetector(
                spread_model=spread_model,
                prop_models=prop_models,
                min_edge=float(self.settings.value_detection.min_edge_threshold),
            )
            logger.info(f"Value detector initialized with {len(prop_models)} prop models")

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

            # INSTANT START: Trigger immediate data refresh
            # Don't wait for scheduler - users need data NOW
            logger.info("🚀 Startup Data Refresh Initiated")
            asyncio.create_task(self._perform_startup_refresh())

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

    def get_fallback_data(self) -> list:
        """
        Fallback value bets - returns empty list.

        Previously returned hardcoded fake bets with fake game matchups
        (BAL @ KC, DAL @ SF, GB @ DET) that don't exist in the real schedule.

        Now returns empty list - no fake data. The scheduler should produce
        real value bets from live odds data.
        """
        logger.warning("get_fallback_data called - returning empty list (no fake bets)")
        self._using_fallback = False  # Not using fake fallback
        return []

    def get_fallback_games(self) -> list:
        """
        Fallback games - returns empty list.

        Previously returned hardcoded fake games, but this caused
        confusion when fake matchups (BAL @ KC) were shown that
        don't exist in the real schedule.

        Now returns empty list - let the UI handle "no games" gracefully.
        The pipeline should be fetching real games from nflverse.
        """
        logger.warning("get_fallback_games called - returning empty list (no fake games)")
        return []

    @property
    def last_value_bets(self) -> list:
        """
        Get last detected value bets from scheduler.

        Returns real value bets only - no fake fallback data.
        If no value bets are found, returns empty list (this is valid -
        it means the market is efficient or no edges were detected).
        """
        self._using_fallback = False  # Reset flag

        # Try to get live bets from scheduler
        if self.scheduler:
            live_bets = self.scheduler.get_last_value_bets()
            if live_bets and len(live_bets) > 0:
                return live_bets

        # Try instance-level cached bets
        if self._last_value_bets and len(self._last_value_bets) > 0:
            return self._last_value_bets

        # No bets found - return empty list (NOT fake fallback data)
        # This is valid: it means no value edges were detected
        logger.info("No value bets in memory - market may be efficient or poll hasn't run yet")
        return []

    async def _perform_startup_refresh(self) -> None:
        """
        Perform immediate data refresh on startup.

        This is the "Instant Start" feature - loads live odds and detects
        value bets immediately, so users see data on first page load.
        """
        try:
            if self.scheduler:
                logger.info("🔄 Running startup value bet poll...")
                # Use the scheduler's poll method which handles everything
                await self.scheduler._poll_odds_job()
                self._last_data_refresh = datetime.now()
                self._startup_refresh_complete = True
                self._is_initializing = False
                logger.info(f"✅ Startup Data Refresh Complete - {len(self.last_value_bets)} bets found")

                # Save to cache for future instant loads
                if len(self.last_value_bets) > 0:
                    self._save_to_cache()
            else:
                logger.warning("⚠️ Scheduler not available for startup refresh")
                self._is_initializing = False
        except Exception as e:
            logger.error(f"❌ Startup Data Refresh Failed: {e}")
            self._is_initializing = False
            # Don't raise - this is a background task

    def get_health_status(self) -> dict:
        """Get health status of all components."""
        # Count value bets in memory
        bets_count = len(self.last_value_bets)

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
            "bets_in_memory": bets_count,
            "last_data_refresh": self._last_data_refresh.isoformat() if self._last_data_refresh else None,
            "startup_refresh_complete": self._startup_refresh_complete,
            "demo_mode": False,  # Demo mode removed - always False
            "is_initializing": self._is_initializing,
            "using_fallback": False,  # Fallback data removed - always False
        }
        if self._init_error:
            status["init_error"] = self._init_error
        return status
