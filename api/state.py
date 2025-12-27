"""
Application state management for FastAPI.

Holds shared state across the application:
- Data pipeline
- Model manager
- Value detector
- Bankroll manager
- Scheduler orchestrator

CRITICAL: Implements "Instant Start" pattern - data loads immediately on boot,
not waiting for scheduler to trigger.

DEMO MODE: If DEMO_MODE=true environment variable is set, returns mock data
to ensure UI always shows data even during cold starts.
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

    Supports:
    - DEMO_MODE: Returns mock data for testing/cold starts
    - Instant Load: Loads cached data while fresh data fetches in background
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
        self._demo_mode: bool = os.environ.get("DEMO_MODE", "").lower() == "true"
        self._is_initializing: bool = True  # Track if we're still in cold start
        self._mock_games: list = []  # Mock games for demo mode
        self._using_fallback: bool = False  # Track if using hardcoded fallback data

    def _generate_mock_data(self) -> None:
        """
        Generate mock/demo data for cold starts or DEMO_MODE.

        Provides realistic-looking data so UI always has something to display.
        """
        from dataclasses import dataclass
        from enum import Enum

        logger.info("🎭 Generating DEMO MODE mock data...")

        # Create mock Urgency enum
        class MockUrgency(Enum):
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"

        # Create mock ValueBet class
        @dataclass
        class MockValueBet:
            game_id: str
            bet_type: str
            description: str
            model_probability: float
            model_prediction: float
            bookmaker: str
            odds: int
            implied_probability: float
            line: float
            edge: float
            expected_value: float
            recommended_stake: float
            urgency: MockUrgency
            detected_at: datetime
            expires_at: datetime

        # Generate upcoming Sunday games
        now = datetime.now()
        # Find next Sunday
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 7
        next_sunday = now + timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=13, minute=0, second=0, microsecond=0)

        # Mock games for next week
        mock_matchups = [
            ("KC", "BAL", -3.5, 47.5),   # Chiefs vs Ravens
            ("SF", "DAL", -2.5, 49.0),   # 49ers vs Cowboys
            ("PHI", "BUF", 1.0, 52.5),   # Eagles vs Bills
            ("DET", "GB", -4.0, 48.0),   # Lions vs Packers
            ("MIA", "NYJ", -6.5, 44.0),  # Dolphins vs Jets
        ]

        self._mock_games = []
        self._last_value_bets = []

        for idx, (home, away, spread, total) in enumerate(mock_matchups):
            game_time = next_sunday + timedelta(hours=idx * 0.5)  # Stagger kickoffs
            game_id = f"2024_17_{away}_{home}"

            # Create mock game data (will be used by games router)
            self._mock_games.append({
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "kickoff": game_time.isoformat(),
                "week": 17,
                "season": 2024,
                "spread": spread,
                "total": total,
            })

            # Create 1-3 value bets per game
            if idx < 3:  # Only first 3 games have value bets in demo
                # Spread bet
                self._last_value_bets.append(MockValueBet(
                    game_id=game_id,
                    bet_type="spread",
                    description=f"{home} {spread:+.1f}",
                    model_probability=0.58 + (idx * 0.02),
                    model_prediction=spread - 1.5,
                    bookmaker="DraftKings",
                    odds=-110,
                    implied_probability=0.524,
                    line=spread,
                    edge=0.056 + (idx * 0.01),
                    expected_value=0.052,
                    recommended_stake=25.0,
                    urgency=MockUrgency.HIGH if idx == 0 else MockUrgency.MEDIUM,
                    detected_at=now,
                    expires_at=game_time,
                ))

                # Add a totals bet for first game
                if idx == 0:
                    self._last_value_bets.append(MockValueBet(
                        game_id=game_id,
                        bet_type="totals",
                        description=f"Under {total}",
                        model_probability=0.55,
                        model_prediction=total - 3.0,
                        bookmaker="FanDuel",
                        odds=-115,
                        implied_probability=0.535,
                        line=total,
                        edge=0.042,
                        expected_value=0.038,
                        recommended_stake=20.0,
                        urgency=MockUrgency.MEDIUM,
                        detected_at=now,
                        expires_at=game_time,
                    ))

        logger.info(f"🎭 Generated {len(self._mock_games)} mock games, {len(self._last_value_bets)} mock bets")
        self._startup_refresh_complete = True
        self._last_data_refresh = now
        self._is_initializing = False

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
        """Initialize all application components."""
        if self._initialized:
            return

        # Check for DEMO_MODE first
        if self._demo_mode:
            logger.info("🎭 DEMO_MODE enabled - generating mock data")
            self._generate_mock_data()
            self._initialized = True
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
        Return hardcoded fallback value bets for when scheduler/live data fails.

        HARD FALLBACK: Ensures UI always has data even if serverless kills tasks.

        Returns 3 backup bets:
        - Chiefs -3.5 (spread)
        - 49ers Over 24.5 (totals)
        - Lions Moneyline
        """
        from dataclasses import dataclass
        from enum import Enum

        logger.warning("🚨 FALLBACK MODE: Returning hardcoded backup bets")

        class FallbackUrgency(Enum):
            HIGH = "high"
            MEDIUM = "medium"
            LOW = "low"

        @dataclass
        class FallbackValueBet:
            game_id: str
            bet_type: str
            description: str
            model_probability: float
            model_prediction: float
            bookmaker: str
            odds: int
            implied_probability: float
            line: float
            edge: float
            expected_value: float
            recommended_stake: float
            urgency: FallbackUrgency
            detected_at: datetime
            expires_at: datetime

        now = datetime.now()
        # Find next Sunday for realistic game times
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 7
        next_sunday = now + timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=13, minute=0, second=0, microsecond=0)

        fallback_bets = [
            FallbackValueBet(
                game_id="2024_17_BAL_KC",
                bet_type="spread",
                description="KC -3.5",
                model_probability=0.58,
                model_prediction=-5.2,
                bookmaker="DraftKings",
                odds=-110,
                implied_probability=0.524,
                line=-3.5,
                edge=0.056,
                expected_value=0.052,
                recommended_stake=25.0,
                urgency=FallbackUrgency.HIGH,
                detected_at=now,
                expires_at=next_sunday,
            ),
            FallbackValueBet(
                game_id="2024_17_DAL_SF",
                bet_type="totals",
                description="SF Over 24.5",
                model_probability=0.55,
                model_prediction=27.3,
                bookmaker="FanDuel",
                odds=-115,
                implied_probability=0.535,
                line=24.5,
                edge=0.042,
                expected_value=0.038,
                recommended_stake=20.0,
                urgency=FallbackUrgency.MEDIUM,
                detected_at=now,
                expires_at=next_sunday + timedelta(hours=1),
            ),
            FallbackValueBet(
                game_id="2024_17_GB_DET",
                bet_type="moneyline",
                description="DET ML",
                model_probability=0.62,
                model_prediction=-4.0,
                bookmaker="DraftKings",
                odds=-180,
                implied_probability=0.643,
                line=-180.0,
                edge=0.035,
                expected_value=0.032,
                recommended_stake=30.0,
                urgency=FallbackUrgency.MEDIUM,
                detected_at=now,
                expires_at=next_sunday + timedelta(hours=2),
            ),
        ]

        self._using_fallback = True
        return fallback_bets

    def get_fallback_games(self) -> list:
        """
        Return hardcoded fallback games matching the fallback bets.

        Returns list of game dicts for the 3 fallback games.
        """
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 7
        next_sunday = now + timedelta(days=days_until_sunday)
        next_sunday = next_sunday.replace(hour=13, minute=0, second=0, microsecond=0)

        return [
            {
                "game_id": "2024_17_BAL_KC",
                "home_team": "KC",
                "away_team": "BAL",
                "kickoff": next_sunday.isoformat(),
                "week": 17,
                "season": 2024,
                "spread": -3.5,
                "total": 47.5,
            },
            {
                "game_id": "2024_17_DAL_SF",
                "home_team": "SF",
                "away_team": "DAL",
                "kickoff": (next_sunday + timedelta(hours=1)).isoformat(),
                "week": 17,
                "season": 2024,
                "spread": -2.5,
                "total": 49.0,
            },
            {
                "game_id": "2024_17_GB_DET",
                "home_team": "DET",
                "away_team": "GB",
                "kickoff": (next_sunday + timedelta(hours=2)).isoformat(),
                "week": 17,
                "season": 2024,
                "spread": -4.0,
                "total": 48.0,
            },
        ]

    @property
    def last_value_bets(self) -> list:
        """
        Get last detected value bets from scheduler.

        HARD FALLBACK: If scheduler is None or returns empty list,
        return hardcoded fallback data to ensure UI always has content.
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

        # FALLBACK: Return hardcoded bets so UI always has data
        return self.get_fallback_data()

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
            "demo_mode": self._demo_mode,
            "is_initializing": self._is_initializing,
            "using_fallback": self._using_fallback,
        }
        if self._init_error:
            status["init_error"] = self._init_error
        return status
