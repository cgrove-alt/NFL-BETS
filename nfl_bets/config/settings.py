"""
Configuration management using pydantic-settings.

Loads configuration from environment variables and .env file.
"""
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ValueDetectionSettings(BaseSettings):
    """Settings for value bet detection."""

    model_config = SettingsConfigDict(env_prefix="")

    min_edge_threshold: Decimal = Field(
        default=Decimal("0.03"),
        description="Minimum edge (model prob - implied prob) to flag as value bet",
    )
    min_confidence_threshold: Decimal = Field(
        default=Decimal("0.55"),
        description="Minimum model confidence to consider bet",
    )
    kelly_multiplier: Decimal = Field(
        default=Decimal("0.25"),
        description="Fraction of Kelly Criterion to use (0.25 = quarter Kelly)",
    )
    max_stake_percent: Decimal = Field(
        default=Decimal("0.05"),
        description="Maximum stake as percentage of bankroll",
    )


class ArbitrageSettings(BaseSettings):
    """Settings for arbitrage detection."""

    model_config = SettingsConfigDict(env_prefix="ARB_")

    min_profit_percent: Decimal = Field(
        default=Decimal("0.005"),
        description="Minimum profit percentage to flag arbitrage opportunity",
    )
    default_stake: Decimal = Field(
        default=Decimal("100"),
        description="Default total stake for arbitrage calculations",
    )
    excluded_bookmakers: list[str] = Field(
        default_factory=list,
        description="Bookmakers to exclude from arbitrage scanning",
    )


class NotificationSettings(BaseSettings):
    """Settings for notification channels."""

    model_config = SettingsConfigDict(env_prefix="NOTIFICATIONS_")

    # Console (Rich terminal output)
    console_enabled: bool = Field(default=True)

    # Desktop notifications (plyer)
    desktop_enabled: bool = Field(default=True)
    desktop_min_priority: str = Field(default="high")

    # Discord webhook
    discord_enabled: bool = Field(default=False)
    discord_webhook_url: Optional[str] = Field(default=None)

    # Telegram bot
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)

    # SMS via Twilio
    sms_enabled: bool = Field(default=False)
    twilio_account_sid: Optional[str] = Field(default=None)
    twilio_auth_token: Optional[str] = Field(default=None)
    twilio_from_number: Optional[str] = Field(default=None)
    twilio_to_number: Optional[str] = Field(default=None)

    # Alert management
    dedup_window_minutes: int = Field(
        default=15,
        description="Minutes to wait before re-alerting on same opportunity",
    )
    rate_limit_per_minute: int = Field(
        default=10,
        description="Maximum alerts per minute",
    )


class SchedulerSettings(BaseSettings):
    """Settings for job scheduler."""

    model_config = SettingsConfigDict(env_prefix="")

    odds_poll_interval_minutes: int = Field(
        default=2,
        description="Minutes between odds API polls",
    )
    active_hours_start: int = Field(
        default=8,
        description="Hour to start polling (24-hour format)",
    )
    active_hours_end: int = Field(
        default=23,
        description="Hour to stop polling (24-hour format)",
    )
    model_refresh_hour: int = Field(
        default=6,
        description="Hour to refresh ML models (24-hour format)",
    )
    nflfastr_sync_hour: int = Field(
        default=3,
        description="Hour to sync nflfastR data (24-hour format)",
    )


class OddsAPISettings(BaseSettings):
    """Settings for The Odds API."""

    model_config = SettingsConfigDict(env_prefix="ODDS_")

    api_key: str = Field(
        default="",
        description="API key from the-odds-api.com",
    )
    base_url: str = Field(
        default="https://api.the-odds-api.com/v4",
        description="Base URL for the API",
    )
    regions: list[str] = Field(
        default=["us"],
        description="Regions to fetch odds from",
    )
    default_bookmakers: list[str] = Field(
        default=[
            "draftkings",
            "fanduel",
            "betmgm",
            "caesars",
            "pointsbetus",
        ],
        description="Default bookmakers to fetch",
    )


class PremiumDataSettings(BaseSettings):
    """Settings for premium data sources."""

    model_config = SettingsConfigDict(env_prefix="")

    # PFF
    pff_enabled: bool = Field(default=False)
    pff_api_key: Optional[str] = Field(default=None)

    # FTN / DVOA
    ftn_enabled: bool = Field(default=False)
    ftn_api_key: Optional[str] = Field(default=None)
    ftn_csv_path: str = Field(default="data/external/ftn/")

    # SIC Score
    sic_score_enabled: bool = Field(default=False)
    sic_api_key: Optional[str] = Field(default=None)


class DashboardSettings(BaseSettings):
    """Settings for dashboard interfaces."""

    model_config = SettingsConfigDict(env_prefix="")

    dashboard_mode: str = Field(
        default="terminal",
        description="Dashboard mode: terminal, web, or both",
    )
    web_dashboard_port: int = Field(
        default=8501,
        description="Port for Streamlit web dashboard",
    )

    @field_validator("dashboard_mode")
    @classmethod
    def validate_dashboard_mode(cls, v: str) -> str:
        allowed = ["terminal", "web", "both"]
        if v not in allowed:
            raise ValueError(f"dashboard_mode must be one of {allowed}")
        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///nfl_bets.db",
        description="Database connection URL",
    )

    # Redis (optional)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching",
    )

    # Bankroll
    initial_bankroll: Decimal = Field(
        default=Decimal("1000.00"),
        description="Starting bankroll amount",
    )

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/nfl_bets.log")
    debug: bool = Field(default=False)

    # Sub-settings
    value_detection: ValueDetectionSettings = Field(
        default_factory=ValueDetectionSettings
    )
    arbitrage: ArbitrageSettings = Field(default_factory=ArbitrageSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    odds_api: OddsAPISettings = Field(default_factory=OddsAPISettings)
    premium_data: PremiumDataSettings = Field(default_factory=PremiumDataSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        data_path = self.project_root / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        return data_path

    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        logs_path = self.project_root / "logs"
        logs_path.mkdir(parents=True, exist_ok=True)
        return logs_path


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
