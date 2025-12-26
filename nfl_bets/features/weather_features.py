"""
Weather feature engineering for NFL betting.

Fetches and processes weather data for outdoor games.
Weather significantly impacts scoring, especially:
- Wind: Reduces passing efficiency
- Cold: Favors run-heavy teams
- Rain/Snow: Reduces scoring

Uses OpenWeatherMap API (free tier: 1000 calls/day).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx
from loguru import logger

from .base import BaseFeatureBuilder, FeatureSet
from .game_features import INDOOR_STADIUMS, TEAM_LOCATIONS


@dataclass
class WeatherData:
    """Weather conditions for a game."""

    temperature: float  # Fahrenheit
    feels_like: float  # Wind chill / heat index
    wind_speed: float  # mph
    wind_gust: Optional[float]  # mph
    humidity: float  # percentage
    precipitation_prob: float  # 0-1
    precipitation_type: Optional[str]  # rain, snow, none
    visibility: float  # miles
    description: str  # e.g., "light rain", "clear sky"

    @property
    def is_adverse(self) -> bool:
        """Check if weather is adverse for passing."""
        return (
            self.wind_speed > 15 or
            self.temperature < 35 or
            self.precipitation_prob > 0.3
        )

    @property
    def weather_impact_score(self) -> float:
        """
        Calculate weather impact score (0-1).

        0 = perfect conditions
        1 = severe weather impact
        """
        score = 0.0

        # Wind impact (biggest factor for passing)
        if self.wind_speed > 20:
            score += 0.3
        elif self.wind_speed > 15:
            score += 0.2
        elif self.wind_speed > 10:
            score += 0.1

        # Temperature impact
        if self.temperature < 20:
            score += 0.25
        elif self.temperature < 35:
            score += 0.15
        elif self.temperature > 90:
            score += 0.1

        # Precipitation impact
        if self.precipitation_prob > 0.7:
            score += 0.25
        elif self.precipitation_prob > 0.4:
            score += 0.15
        elif self.precipitation_prob > 0.2:
            score += 0.05

        # Snow is worse than rain
        if self.precipitation_type == "snow":
            score += 0.1

        return min(1.0, score)


class WeatherAPIClient:
    """Client for OpenWeatherMap API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather client.

        Args:
            api_key: OpenWeatherMap API key. If not provided,
                     uses OPENWEATHERMAP_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.client = httpx.AsyncClient(timeout=10.0)

        if not self.api_key:
            logger.warning(
                "No OpenWeatherMap API key configured. "
                "Set OPENWEATHERMAP_API_KEY environment variable."
            )

    async def get_current_weather(
        self,
        lat: float,
        lon: float,
    ) -> Optional[WeatherData]:
        """
        Get current weather for a location.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            WeatherData or None if fetch fails
        """
        if not self.api_key:
            return None

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "imperial",  # Fahrenheit
                },
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_weather_response(data)

        except Exception as e:
            logger.warning(f"Weather API error: {e}")
            return None

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        game_time: datetime,
    ) -> Optional[WeatherData]:
        """
        Get weather forecast for a specific time.

        Uses 3-hour forecast API for future games.

        Args:
            lat: Latitude
            lon: Longitude
            game_time: When the game starts

        Returns:
            WeatherData or None if fetch fails
        """
        if not self.api_key:
            return None

        try:
            response = await self.client.get(
                f"{self.BASE_URL}/forecast",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self.api_key,
                    "units": "imperial",
                },
            )
            response.raise_for_status()
            data = response.json()

            # Find closest forecast to game time
            closest_forecast = None
            min_diff = float("inf")

            for item in data.get("list", []):
                forecast_time = datetime.fromtimestamp(item["dt"])
                diff = abs((forecast_time - game_time).total_seconds())

                if diff < min_diff:
                    min_diff = diff
                    closest_forecast = item

            if closest_forecast:
                return self._parse_weather_response(closest_forecast)

            return None

        except Exception as e:
            logger.warning(f"Weather forecast API error: {e}")
            return None

    def _parse_weather_response(self, data: dict) -> WeatherData:
        """Parse OpenWeatherMap response into WeatherData."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]

        # Determine precipitation type
        precip_type = None
        weather_id = weather.get("id", 800)
        if 200 <= weather_id < 600:  # Rain/thunderstorm
            precip_type = "rain"
        elif 600 <= weather_id < 700:  # Snow
            precip_type = "snow"

        # Estimate precipitation probability from weather code
        # OpenWeatherMap doesn't give probability in basic API
        precip_prob = 0.0
        if weather_id < 800:  # Not clear
            if weather_id < 300:  # Thunderstorm
                precip_prob = 0.9
            elif weather_id < 400:  # Drizzle
                precip_prob = 0.6
            elif weather_id < 600:  # Rain
                precip_prob = 0.8
            elif weather_id < 700:  # Snow
                precip_prob = 0.8
            else:  # Atmosphere (fog, etc.)
                precip_prob = 0.2

        return WeatherData(
            temperature=main.get("temp", 65),
            feels_like=main.get("feels_like", 65),
            wind_speed=wind.get("speed", 5),
            wind_gust=wind.get("gust"),
            humidity=main.get("humidity", 50),
            precipitation_prob=precip_prob,
            precipitation_type=precip_type,
            visibility=data.get("visibility", 10000) / 1609.34,  # m to miles
            description=weather.get("description", "clear"),
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class WeatherFeatureBuilder(BaseFeatureBuilder):
    """
    Builds weather-related features for NFL games.

    Weather significantly impacts scoring:
    - High wind reduces passing efficiency by 10-20%
    - Cold weather (<35°F) reduces scoring by ~3 points
    - Rain/snow reduces both passing and scoring

    Only applies to outdoor stadiums.
    """

    WEATHER_FEATURES = [
        "temperature",
        "feels_like",
        "wind_speed",
        "wind_gust",
        "humidity",
        "precipitation_prob",
        "weather_impact_score",
        "is_cold_game",
        "is_windy",
        "is_precipitation",
        "is_adverse_weather",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache=None,
        cache_ttl_seconds: int = 3600,  # 1 hour
    ):
        super().__init__(cache=cache, cache_ttl_seconds=cache_ttl_seconds)
        self.weather_client = WeatherAPIClient(api_key)

    def get_feature_names(self) -> list[str]:
        """Get all feature names this builder produces."""
        return self.WEATHER_FEATURES

    async def build_features(
        self,
        home_team: str,
        game_time: Optional[datetime] = None,
        weather_data: Optional[WeatherData] = None,
    ) -> FeatureSet:
        """
        Build weather features for a game.

        Args:
            home_team: Home team abbreviation
            game_time: Game start time (for forecast)
            weather_data: Pre-fetched weather data (optional)

        Returns:
            FeatureSet with weather features
        """
        # Indoor stadiums get default perfect conditions
        if home_team in INDOOR_STADIUMS:
            return self._get_indoor_features(home_team)

        # Check cache
        cache_key = self._get_cache_key("weather", home_team, game_time)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Fetch weather if not provided
        if weather_data is None:
            weather_data = await self._fetch_weather(home_team, game_time)

        if weather_data is None:
            # Use default outdoor conditions
            return self._get_default_outdoor_features(home_team)

        # Build features from weather data
        features = {
            "temperature": weather_data.temperature,
            "feels_like": weather_data.feels_like,
            "wind_speed": weather_data.wind_speed,
            "wind_gust": weather_data.wind_gust or weather_data.wind_speed,
            "humidity": weather_data.humidity,
            "precipitation_prob": weather_data.precipitation_prob,
            "weather_impact_score": weather_data.weather_impact_score,
            "is_cold_game": 1.0 if weather_data.temperature < 40 else 0.0,
            "is_windy": 1.0 if weather_data.wind_speed > 15 else 0.0,
            "is_precipitation": 1.0 if weather_data.precipitation_prob > 0.3 else 0.0,
            "is_adverse_weather": 1.0 if weather_data.is_adverse else 0.0,
        }

        feature_set = FeatureSet(features=features)
        await self._set_cached(cache_key, feature_set)

        return feature_set

    async def _fetch_weather(
        self,
        home_team: str,
        game_time: Optional[datetime] = None,
    ) -> Optional[WeatherData]:
        """Fetch weather for a stadium location."""
        location = TEAM_LOCATIONS.get(home_team)
        if not location:
            return None

        lat, lon = location

        if game_time and game_time > datetime.now():
            # Future game - use forecast
            return await self.weather_client.get_forecast(lat, lon, game_time)
        else:
            # Current/past - use current weather (best approximation)
            return await self.weather_client.get_current_weather(lat, lon)

    def _get_indoor_features(self, home_team: str) -> FeatureSet:
        """Get perfect indoor conditions."""
        return FeatureSet(
            features={
                "temperature": 72.0,
                "feels_like": 72.0,
                "wind_speed": 0.0,
                "wind_gust": 0.0,
                "humidity": 50.0,
                "precipitation_prob": 0.0,
                "weather_impact_score": 0.0,
                "is_cold_game": 0.0,
                "is_windy": 0.0,
                "is_precipitation": 0.0,
                "is_adverse_weather": 0.0,
            }
        )

    def _get_default_outdoor_features(self, home_team: str) -> FeatureSet:
        """Get default outdoor conditions when weather unavailable."""
        return FeatureSet(
            features={
                "temperature": 55.0,
                "feels_like": 52.0,
                "wind_speed": 8.0,
                "wind_gust": 12.0,
                "humidity": 60.0,
                "precipitation_prob": 0.1,
                "weather_impact_score": 0.1,
                "is_cold_game": 0.0,
                "is_windy": 0.0,
                "is_precipitation": 0.0,
                "is_adverse_weather": 0.0,
            }
        )


# Convenience function for quick weather fetch
async def get_game_weather(
    home_team: str,
    game_time: Optional[datetime] = None,
    api_key: Optional[str] = None,
) -> dict[str, float]:
    """
    Quick function to get weather features for a game.

    Args:
        home_team: Home team abbreviation
        game_time: Game start time
        api_key: OpenWeatherMap API key (optional)

    Returns:
        Dictionary of weather features
    """
    builder = WeatherFeatureBuilder(api_key=api_key)
    features = await builder.build_features(home_team, game_time)
    return features.features
