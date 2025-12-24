"""
Streamlit web dashboard for NFL Bets.

Provides a web-based interface for:
- Real-time value bet monitoring
- Bankroll and performance analytics
- Historical bet tracking
- Model performance visualization

Run with:
    streamlit run nfl_bets/dashboard/web.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.settings = None
        st.session_state.bankroll_manager = None
        st.session_state.model_manager = None


def load_components():
    """Load application components."""
    if st.session_state.initialized:
        return

    try:
        from nfl_bets.config.settings import get_settings
        from nfl_bets.tracking.bankroll_manager import BankrollManager
        from nfl_bets.models.model_manager import ModelManager

        st.session_state.settings = get_settings()
        st.session_state.bankroll_manager = BankrollManager(
            initial_bankroll=float(st.session_state.settings.initial_bankroll)
        )
        st.session_state.model_manager = ModelManager()
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("NFL Bets")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Value Bets", "Analytics", "Models", "Settings"],
    )

    st.sidebar.markdown("---")

    # Quick stats in sidebar
    if st.session_state.initialized and st.session_state.bankroll_manager:
        bankroll = st.session_state.bankroll_manager
        st.sidebar.metric(
            "Bankroll",
            f"${bankroll.current_bankroll:,.2f}",
        )

    return page


def render_dashboard():
    """Render the main dashboard page."""
    st.title("Dashboard")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    if st.session_state.bankroll_manager:
        bankroll = st.session_state.bankroll_manager
        summary = bankroll.get_performance_summary()

        with col1:
            st.metric(
                "Current Bankroll",
                f"${bankroll.current_bankroll:,.2f}",
            )

        with col2:
            roi = summary.get("roi", 0)
            st.metric(
                "ROI",
                f"{roi:+.1%}",
                delta=f"{roi:+.1%}" if roi != 0 else None,
            )

        with col3:
            total_bets = summary.get("total_bets", 0)
            wins = summary.get("wins", 0)
            st.metric(
                "Win Rate",
                f"{wins}/{total_bets}" if total_bets > 0 else "N/A",
            )

        with col4:
            pending = bankroll.pending_exposure
            st.metric(
                "Pending Bets",
                f"${pending:,.2f}",
            )

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Active Value Bets")
        # Placeholder for value bets table
        st.info("No active value bets. Run the scheduler to detect opportunities.")

    with col2:
        st.subheader("Model Status")
        if st.session_state.model_manager:
            try:
                all_fresh, stale = st.session_state.model_manager.check_all_models_fresh()
                if all_fresh:
                    st.success("All models are fresh")
                else:
                    st.warning(f"Stale models: {', '.join(stale)}")
            except Exception as e:
                st.error(f"Model check failed: {e}")


def render_value_bets():
    """Render the value bets page."""
    st.title("Value Bets")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_edge = st.slider(
            "Minimum Edge",
            min_value=0.0,
            max_value=0.20,
            value=0.03,
            step=0.01,
            format="%.0f%%",
        )

    with col2:
        bet_types = st.multiselect(
            "Bet Types",
            ["Spread", "Moneyline", "Total", "Player Props"],
            default=["Spread"],
        )

    with col3:
        urgency = st.multiselect(
            "Urgency",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )

    st.markdown("---")

    # Value bets table
    st.subheader("Detected Opportunities")

    # Placeholder - in production this would query the database
    st.info(
        "Start the main application with `nfl-bets` to begin detecting value bets.\n\n"
        "Value bets will appear here when the scheduler polls the Odds API."
    )


def render_analytics():
    """Render the analytics page."""
    st.title("Analytics")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
        )

    st.markdown("---")

    # Performance charts
    tab1, tab2, tab3 = st.tabs(["Bankroll", "Win Rate", "By Bet Type"])

    with tab1:
        st.subheader("Bankroll Over Time")
        # Placeholder chart
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        bankroll_data = pd.DataFrame({
            "date": dates,
            "balance": [1000 + i * 10 for i in range(len(dates))],
        })
        fig = px.line(
            bankroll_data,
            x="date",
            y="balance",
            title="Bankroll History",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Win Rate by Week")
        # Placeholder chart
        weeks = [f"Week {i+1}" for i in range(17)]
        win_rates = [0.55 + (i % 3) * 0.05 for i in range(17)]
        fig = px.bar(
            x=weeks,
            y=win_rates,
            title="Weekly Win Rate",
            labels={"x": "Week", "y": "Win Rate"},
        )
        fig.add_hline(y=0.524, line_dash="dash", annotation_text="Breakeven (52.4%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Performance by Bet Type")
        # Placeholder chart
        bet_types = ["Spread", "Moneyline", "Total", "Props"]
        roi_values = [0.08, 0.05, 0.03, 0.12]
        fig = px.bar(
            x=bet_types,
            y=roi_values,
            title="ROI by Bet Type",
            labels={"x": "Bet Type", "y": "ROI"},
            color=roi_values,
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_models():
    """Render the models page."""
    st.title("Model Performance")

    if not st.session_state.model_manager:
        st.error("Model manager not initialized")
        return

    # Model overview
    st.subheader("Model Status")

    model_types = ["spread", "passing_yards", "rushing_yards", "receiving_yards"]

    for model_type in model_types:
        try:
            info = st.session_state.model_manager.get_model_info(model_type)

            with st.expander(f"{model_type.replace('_', ' ').title()} Model"):
                if "error" in info:
                    st.warning(f"Model not found: {info.get('error')}")
                else:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Version", info.get("model_version", "N/A"))

                    with col2:
                        cutoff = info.get("data_cutoff_date")
                        if cutoff:
                            st.metric("Data Cutoff", cutoff[:10])
                        else:
                            st.metric("Data Cutoff", "Unknown")

                    with col3:
                        is_stale = info.get("is_stale", True)
                        if is_stale:
                            st.error("STALE")
                        else:
                            st.success("FRESH")

                    # Metrics if available
                    metrics = info.get("metrics", {})
                    if metrics:
                        st.markdown("**Validation Metrics:**")
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                        with mcol2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                        with mcol3:
                            st.metric("R²", f"{metrics.get('r2', 0):.3f}")

        except Exception as e:
            st.error(f"Error loading {model_type}: {e}")

    st.markdown("---")

    # Retrain button
    st.subheader("Model Training")
    st.warning(
        "Model training can take several minutes. "
        "Make sure you have the latest nflverse data."
    )

    if st.button("Retrain All Models", type="primary"):
        st.info("Training would be triggered here. Run from CLI: `python -m nfl_bets.scripts.train_models`")


def render_settings():
    """Render the settings page."""
    st.title("Settings")

    if not st.session_state.settings:
        st.error("Settings not loaded")
        return

    settings = st.session_state.settings

    # Betting settings
    st.subheader("Betting Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Initial Bankroll",
            value=float(settings.initial_bankroll),
            min_value=0.0,
            step=100.0,
            disabled=True,
            help="Set in .env file",
        )

        st.number_input(
            "Minimum Edge Threshold",
            value=float(settings.value_detection.min_edge_threshold) * 100,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            format="%.1f%%",
            disabled=True,
            help="Set in .env file",
        )

    with col2:
        st.number_input(
            "Kelly Multiplier",
            value=float(settings.value_detection.kelly_multiplier),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            disabled=True,
            help="Set in .env file",
        )

        st.number_input(
            "Max Stake %",
            value=float(settings.value_detection.max_stake_percent) * 100,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.0f%%",
            disabled=True,
            help="Set in .env file",
        )

    st.markdown("---")

    # Scheduler settings
    st.subheader("Scheduler Configuration")

    sched = settings.scheduler

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Odds Poll Interval (minutes)",
            value=sched.odds_poll_interval_minutes,
            disabled=True,
        )

        st.text_input(
            "Active Hours",
            value=f"{sched.active_hours_start}:00 - {sched.active_hours_end}:00",
            disabled=True,
        )

    with col2:
        st.number_input(
            "Model Refresh Hour",
            value=sched.model_refresh_hour,
            disabled=True,
        )

        st.number_input(
            "nflverse Sync Hour",
            value=sched.nflfastr_sync_hour,
            disabled=True,
        )

    st.markdown("---")

    st.info(
        "Settings are configured via environment variables in the `.env` file. "
        "See `.env.example` for available options."
    )


def main():
    """Main entry point for Streamlit app."""
    st.set_page_config(
        page_title="NFL Bets Dashboard",
        page_icon="🏈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize
    init_session_state()
    load_components()

    # Render navigation and get current page
    page = render_sidebar()

    # Render selected page
    if page == "Dashboard":
        render_dashboard()
    elif page == "Value Bets":
        render_value_bets()
    elif page == "Analytics":
        render_analytics()
    elif page == "Models":
        render_models()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
