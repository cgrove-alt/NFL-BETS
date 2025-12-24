#!/usr/bin/env python3
"""
NFL Bets - Main Application Entry Point.

Real-time NFL betting value detection system that:
1. Polls odds from multiple bookmakers
2. Runs ML predictions against current lines
3. Identifies value betting opportunities
4. Provides real-time dashboards and alerts

Usage:
    nfl-bets                    # Start with default settings
    nfl-bets --mode terminal    # Terminal dashboard only
    nfl-bets --mode web         # Web dashboard only
    nfl-bets --no-scheduler     # Dashboard only, no background polling
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NFLBetsApp:
    """
    Main application orchestrator.

    Initializes all components and manages the application lifecycle:
    - Data pipeline for fetching odds and game data
    - ML models for predictions
    - Value detector for identifying opportunities
    - Scheduler for background polling
    - Dashboard for real-time display
    """

    def __init__(
        self,
        dashboard_mode: Optional[str] = None,
        enable_scheduler: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the NFL Bets application.

        Args:
            dashboard_mode: Override dashboard mode (terminal, web, both)
            enable_scheduler: Whether to start background polling jobs
            debug: Enable debug logging
        """
        self.dashboard_mode = dashboard_mode
        self.enable_scheduler = enable_scheduler
        self.debug = debug

        # Components (initialized in setup)
        self.settings = None
        self.pipeline = None
        self.feature_pipeline = None
        self.model_manager = None
        self.spread_model = None
        self.value_detector = None
        self.kelly_calculator = None
        self.bankroll_manager = None
        self.scheduler = None
        self.dashboard = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def setup(self) -> None:
        """Initialize all application components."""
        from nfl_bets.config.settings import get_settings

        logger.info("=" * 60)
        logger.info("NFL BETS - Real-Time Value Detection System")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load settings
        self.settings = get_settings()
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Override dashboard mode if specified
        if self.dashboard_mode:
            self.settings.dashboard.dashboard_mode = self.dashboard_mode

        logger.info(f"Dashboard mode: {self.settings.dashboard.dashboard_mode}")

        # Initialize data pipeline
        logger.info("Initializing data pipeline...")
        await self._init_data_pipeline()

        # Initialize models
        logger.info("Loading ML models...")
        await self._init_models()

        # Initialize betting components
        logger.info("Initializing betting components...")
        self._init_betting()

        # Initialize scheduler if enabled
        if self.enable_scheduler:
            logger.info("Initializing scheduler...")
            await self._init_scheduler()

        logger.info("=" * 60)
        logger.info("Initialization complete!")
        logger.info("=" * 60)

    async def _init_data_pipeline(self) -> None:
        """Initialize data pipeline and feature pipeline."""
        from nfl_bets.data.pipeline import DataPipeline
        from nfl_bets.features.feature_pipeline import FeaturePipeline

        self.pipeline = DataPipeline.from_settings(self.settings)

        # Run health check
        health = await self.pipeline.health_check()
        logger.info(f"Pipeline health: {health.status}")

        for source, status in health.source_status.items():
            emoji = "✓" if status else "✗"
            logger.info(f"  {emoji} {source}: {'OK' if status else 'UNAVAILABLE'}")

        # Initialize feature pipeline
        self.feature_pipeline = FeaturePipeline(self.pipeline)

    async def _init_models(self) -> None:
        """Load ML models with staleness checking."""
        from nfl_bets.models.model_manager import ModelManager

        self.model_manager = ModelManager(
            model_dir=Path("models/trained"),
            auto_check_staleness=True,
        )

        # Try to load spread model
        try:
            self.spread_model = self.model_manager.load_spread_model()
            logger.info("✓ Spread model loaded")

            if self.spread_model.metadata:
                cutoff = self.spread_model.metadata.data_cutoff_date
                if cutoff:
                    logger.info(f"  Data cutoff: {cutoff.strftime('%Y-%m-%d')}")
        except FileNotFoundError:
            logger.warning(
                "⚠ Spread model not found. Run 'python -m nfl_bets.scripts.train_models' first."
            )
            self.spread_model = None

        # Check model freshness
        if self.spread_model:
            all_fresh, stale = self.model_manager.check_all_models_fresh()
            if not all_fresh:
                logger.warning(f"⚠ Stale models detected: {stale}")
                logger.warning("  Consider running: python -m nfl_bets.scripts.train_models")

    def _init_betting(self) -> None:
        """Initialize betting components."""
        from nfl_bets.betting.value_detector import ValueDetector
        from nfl_bets.betting.kelly_calculator import KellyCalculator
        from nfl_bets.tracking.bankroll_manager import BankrollManager

        # Value detector
        self.value_detector = ValueDetector(
            spread_model=self.spread_model,
            min_edge=float(self.settings.value_detection.min_edge_threshold),
            min_ev=0.02,  # 2% minimum EV
        )

        # Kelly calculator
        self.kelly_calculator = KellyCalculator(
            kelly_fraction=float(self.settings.value_detection.kelly_multiplier),
            max_bet_fraction=float(self.settings.value_detection.max_stake_percent),
        )

        # Bankroll manager
        self.bankroll_manager = BankrollManager(
            initial_bankroll=float(self.settings.initial_bankroll),
        )

        logger.info(f"✓ Bankroll: ${self.bankroll_manager.current_bankroll:,.2f}")

    async def _init_scheduler(self) -> None:
        """Initialize APScheduler with background jobs."""
        from nfl_bets.scheduler.orchestrator import SchedulerOrchestrator

        self.scheduler = SchedulerOrchestrator(
            settings=self.settings,
            pipeline=self.pipeline,
            feature_pipeline=self.feature_pipeline,
            model_manager=self.model_manager,
            value_detector=self.value_detector,
            bankroll_manager=self.bankroll_manager,
        )

        self.scheduler.start()
        logger.info("✓ Scheduler started")

    async def run(self) -> None:
        """Run the main application loop."""
        self._running = True

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        mode = self.settings.dashboard.dashboard_mode

        try:
            if mode in ("terminal", "both"):
                await self._run_terminal_dashboard()
            elif mode == "web":
                await self._run_web_dashboard()
            else:
                # No dashboard, just keep running
                logger.info("Running in headless mode (no dashboard)")
                await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
        finally:
            await self.shutdown()

    async def _run_terminal_dashboard(self) -> None:
        """Run the Rich terminal dashboard."""
        from nfl_bets.dashboard.terminal import TerminalDashboard

        self.dashboard = TerminalDashboard(
            pipeline=self.pipeline,
            model_manager=self.model_manager,
            value_detector=self.value_detector,
            bankroll_manager=self.bankroll_manager,
            scheduler=self.scheduler,
        )

        # Run dashboard (blocking)
        await self.dashboard.run(shutdown_event=self._shutdown_event)

    async def _run_web_dashboard(self) -> None:
        """Run the Streamlit web dashboard."""
        import subprocess

        port = self.settings.dashboard.web_dashboard_port
        dashboard_path = Path(__file__).parent / "dashboard" / "web.py"

        logger.info(f"Starting web dashboard on port {port}...")

        # Launch Streamlit in subprocess
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port",
                str(port),
                "--server.headless",
                "true",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info(f"Web dashboard available at: http://localhost:{port}")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Terminate Streamlit
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received...")
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down...")

        if self.scheduler:
            self.scheduler.stop()
            logger.info("✓ Scheduler stopped")

        if self.dashboard:
            await self.dashboard.stop()
            logger.info("✓ Dashboard stopped")

        self._running = False
        logger.info("Shutdown complete")


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    app = NFLBetsApp(
        dashboard_mode=args.mode,
        enable_scheduler=not args.no_scheduler,
        debug=args.debug,
    )

    try:
        await app.setup()
        await app.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="NFL Bets - Real-time betting value detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    nfl-bets                    Start with default settings
    nfl-bets --mode terminal    Terminal dashboard only
    nfl-bets --mode web         Web dashboard only
    nfl-bets --no-scheduler     Dashboard only, no polling
    nfl-bets --debug            Enable debug logging
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["terminal", "web", "both"],
        default=None,
        help="Dashboard mode (overrides config)",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable background polling jobs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Run async main
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
