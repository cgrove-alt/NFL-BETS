"""
Dashboard interfaces for NFL Bets.

Provides two dashboard options:
- Terminal: Rich-based live terminal UI
- Web: Streamlit web application

Example:
    # Terminal dashboard
    >>> from nfl_bets.dashboard import TerminalDashboard
    >>> dashboard = TerminalDashboard(pipeline, model_manager, ...)
    >>> await dashboard.run(shutdown_event)

    # Web dashboard
    # Run with: streamlit run nfl_bets/dashboard/web.py
"""

from .terminal import TerminalDashboard

__all__ = [
    "TerminalDashboard",
]
