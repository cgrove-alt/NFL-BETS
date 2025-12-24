"""
Rich-based terminal dashboard for real-time value bet monitoring.

Provides a live-updating terminal UI with:
- Value bets table with edge and stake info
- Bankroll summary panel
- Job status monitoring
- Health indicators
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style

logger = logging.getLogger(__name__)


class TerminalDashboard:
    """
    Rich-based real-time terminal UI.

    Displays a live dashboard with panels for:
    - Header: Status bar with time and health indicators
    - Main: Value bets table with edge, EV, and stake
    - Sidebar: Bankroll summary and active bets
    - Footer: Job status and recent alerts

    Example:
        >>> dashboard = TerminalDashboard(pipeline, model_manager, ...)
        >>> await dashboard.run(shutdown_event)
    """

    def __init__(
        self,
        pipeline: Any,
        model_manager: Any,
        value_detector: Any,
        bankroll_manager: Any,
        scheduler: Optional[Any] = None,
    ):
        """
        Initialize the terminal dashboard.

        Args:
            pipeline: DataPipeline instance
            model_manager: ModelManager instance
            value_detector: ValueDetector instance
            bankroll_manager: BankrollManager instance
            scheduler: Optional SchedulerOrchestrator instance
        """
        self.pipeline = pipeline
        self.model_manager = model_manager
        self.value_detector = value_detector
        self.bankroll_manager = bankroll_manager
        self.scheduler = scheduler

        self.console = Console()
        self._running = False

    def _build_layout(self) -> Layout:
        """Build the dashboard layout structure."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=5),
        )

        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", ratio=1),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render the header panel with status info."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build status indicators
        status_parts = [
            f"[bold white]NFL BETS[/bold white]",
            f"[dim]|[/dim]",
            f"[cyan]{now}[/cyan]",
        ]

        # Add scheduler status
        if self.scheduler:
            if self.scheduler.is_running:
                status_parts.extend([
                    f"[dim]|[/dim]",
                    f"[green]Scheduler: Running[/green]",
                ])
            else:
                status_parts.extend([
                    f"[dim]|[/dim]",
                    f"[red]Scheduler: Stopped[/red]",
                ])

        # Add model status
        try:
            all_fresh, stale = self.model_manager.check_all_models_fresh()
            if all_fresh:
                status_parts.extend([
                    f"[dim]|[/dim]",
                    f"[green]Models: Fresh[/green]",
                ])
            else:
                status_parts.extend([
                    f"[dim]|[/dim]",
                    f"[yellow]Models: Stale ({len(stale)})[/yellow]",
                ])
        except Exception:
            status_parts.extend([
                f"[dim]|[/dim]",
                f"[red]Models: Error[/red]",
            ])

        status_text = " ".join(status_parts)

        return Panel(
            Text.from_markup(status_text),
            style="bold",
            border_style="blue",
        )

    def _render_value_bets_table(self) -> Panel:
        """Render the value bets table."""
        table = Table(
            title="Value Bets",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=True,
        )

        table.add_column("Game", style="white", no_wrap=True)
        table.add_column("Bet", style="white")
        table.add_column("Book", style="dim")
        table.add_column("Odds", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("EV", justify="right")
        table.add_column("Stake", justify="right", style="green")
        table.add_column("Urgency", justify="center")

        # Get value bets from scheduler
        value_bets = []
        if self.scheduler:
            value_bets = self.scheduler.get_last_value_bets()

        if not value_bets:
            table.add_row(
                "[dim]No value bets found[/dim]",
                "", "", "", "", "", "", "",
            )
        else:
            for bet in value_bets:
                # Format urgency with color
                urgency = bet.urgency.value if hasattr(bet.urgency, 'value') else str(bet.urgency)
                urgency_style = self._get_urgency_style(urgency)

                table.add_row(
                    bet.game_id[:20] if bet.game_id else "",
                    bet.description[:25] if bet.description else "",
                    bet.bookmaker[:12] if bet.bookmaker else "",
                    f"{bet.odds:+d}" if bet.odds else "",
                    f"{bet.edge:.1%}" if bet.edge else "",
                    f"{bet.expected_value:.1%}" if bet.expected_value else "",
                    f"${bet.recommended_stake:.0f}" if bet.recommended_stake else "",
                    f"[{urgency_style}]{urgency}[/{urgency_style}]",
                )

        return Panel(table, border_style="green")

    def _render_bankroll_panel(self) -> Panel:
        """Render the bankroll summary panel."""
        bankroll = self.bankroll_manager

        # Build bankroll info
        lines = []

        # Current balance
        balance = bankroll.current_bankroll
        lines.append(f"[bold]Balance:[/bold] [green]${balance:,.2f}[/green]")

        # Pending exposure
        pending = bankroll.pending_exposure
        if pending > 0:
            lines.append(f"[bold]Pending:[/bold] [yellow]${pending:,.2f}[/yellow]")

        # Available
        available = bankroll.available_bankroll
        lines.append(f"[bold]Available:[/bold] [cyan]${available:,.2f}[/cyan]")

        # Divider
        lines.append("")

        # Performance summary
        try:
            summary = bankroll.get_performance_summary()
            if summary:
                lines.append("[dim]─── Performance ───[/dim]")

                total_bets = summary.get("total_bets", 0)
                wins = summary.get("wins", 0)
                losses = summary.get("losses", 0)
                roi = summary.get("roi", 0)

                lines.append(f"Bets: {total_bets} ({wins}W-{losses}L)")

                roi_style = "green" if roi >= 0 else "red"
                lines.append(f"ROI: [{roi_style}]{roi:+.1%}[/{roi_style}]")
        except Exception:
            pass

        content = "\n".join(lines)

        return Panel(
            Text.from_markup(content),
            title="Bankroll",
            border_style="cyan",
        )

    def _render_footer(self) -> Panel:
        """Render the footer with job status."""
        if not self.scheduler:
            return Panel(
                "[dim]Scheduler not running[/dim]",
                title="Jobs",
                border_style="dim",
            )

        # Build job status table
        table = Table(
            show_header=True,
            header_style="dim",
            border_style="dim",
            expand=True,
            box=None,
        )

        table.add_column("Job", style="white")
        table.add_column("Status", justify="center")
        table.add_column("Last Run", justify="right")
        table.add_column("Next Run", justify="right")

        job_status = self.scheduler.get_job_status()

        for job_id, status in job_status.items():
            # Status indicator
            last_status = status.get("last_status", "pending")
            if last_status == "success":
                status_text = "[green]OK[/green]"
            elif last_status == "error":
                status_text = "[red]ERR[/red]"
            else:
                status_text = "[dim]--[/dim]"

            # Last run time
            last_run = status.get("last_run")
            if last_run:
                last_str = last_run.strftime("%H:%M:%S")
            else:
                last_str = "[dim]never[/dim]"

            # Next run time
            next_run = status.get("next_run")
            if next_run:
                next_str = next_run.strftime("%H:%M:%S")
            else:
                next_str = "[dim]paused[/dim]"

            table.add_row(
                status.get("name", job_id)[:20],
                status_text,
                last_str,
                next_str,
            )

        return Panel(table, title="Scheduled Jobs", border_style="blue")

    def _get_urgency_style(self, urgency: str) -> str:
        """Get style for urgency level."""
        urgency_upper = urgency.upper()
        if urgency_upper == "CRITICAL":
            return "bold red"
        elif urgency_upper == "HIGH":
            return "red"
        elif urgency_upper == "MEDIUM":
            return "yellow"
        else:
            return "dim"

    def _update_layout(self, layout: Layout) -> None:
        """Update all layout panels with current data."""
        layout["header"].update(self._render_header())
        layout["main"].update(self._render_value_bets_table())
        layout["sidebar"].update(self._render_bankroll_panel())
        layout["footer"].update(self._render_footer())

    async def run(self, shutdown_event: asyncio.Event) -> None:
        """
        Start the live dashboard with auto-refresh.

        Args:
            shutdown_event: Event to signal shutdown
        """
        self._running = True
        layout = self._build_layout()

        logger.info("Starting terminal dashboard...")

        with Live(
            layout,
            console=self.console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while not shutdown_event.is_set():
                try:
                    self._update_layout(layout)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")
                    await asyncio.sleep(5)

        self._running = False
        logger.info("Terminal dashboard stopped")

    async def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._running
