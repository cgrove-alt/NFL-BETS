"""
Model lifecycle management.

Provides centralized model loading, versioning, and staleness detection
to ensure predictions use appropriately trained models.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

from .base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path("models/trained")


class ModelManager:
    """
    Manages model lifecycle including loading, versioning, and staleness checks.

    Ensures models are properly trained on latest data before making predictions.

    Example:
        >>> manager = ModelManager()
        >>> spread_model = manager.load_spread_model()
        >>>
        >>> # Check if model needs retraining
        >>> if manager.is_stale("spread"):
        ...     logger.warning("Spread model is stale, consider retraining")
        >>>
        >>> # Get model info
        >>> info = manager.get_model_info("spread")
        >>> print(f"Trained on: {info['data_cutoff_date']}")
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        auto_check_staleness: bool = True,
    ):
        """
        Initialize the model manager.

        Args:
            model_dir: Directory containing trained models
            auto_check_staleness: If True, warn when loading stale models
        """
        self.model_dir = Path(model_dir)
        self.auto_check_staleness = auto_check_staleness
        self._loaded_models: dict[str, BaseModel] = {}
        self._latest_game_date: Optional[datetime] = None

    def set_latest_game_date(self, date: datetime) -> None:
        """
        Set the date of the most recent completed game.

        This is used for staleness checking. Should be updated
        when new game data becomes available.

        Args:
            date: Date of latest completed game
        """
        self._latest_game_date = date
        logger.debug(f"Latest game date set to {date.strftime('%Y-%m-%d')}")

    def load_spread_model(
        self,
        version: str = "latest",
        check_staleness: Optional[bool] = None,
    ) -> "BaseModel":
        """
        Load the spread prediction model.

        Args:
            version: Model version or "latest" for most recent
            check_staleness: Override auto_check_staleness setting

        Returns:
            Loaded SpreadModel
        """
        from .spread_model import SpreadModel

        return self._load_model(
            model_class=SpreadModel,
            model_name="spread_model",
            version=version,
            check_staleness=check_staleness,
        )

    def load_moneyline_model(
        self,
        version: str = "latest",
        check_staleness: Optional[bool] = None,
    ) -> "BaseModel":
        """
        Load the moneyline prediction model.

        Args:
            version: Model version or "latest" for most recent
            check_staleness: Override auto_check_staleness setting

        Returns:
            Loaded MoneylineModel
        """
        from .moneyline_model import MoneylineModel

        return self._load_model(
            model_class=MoneylineModel,
            model_name="moneyline_model",
            version=version,
            check_staleness=check_staleness,
        )

    def load_totals_model(
        self,
        version: str = "latest",
        check_staleness: Optional[bool] = None,
    ) -> "BaseModel":
        """
        Load the totals (over/under) prediction model.

        Args:
            version: Model version or "latest" for most recent
            check_staleness: Override auto_check_staleness setting

        Returns:
            Loaded TotalsModel
        """
        from .totals_model import TotalsModel

        return self._load_model(
            model_class=TotalsModel,
            model_name="totals_model",
            version=version,
            check_staleness=check_staleness,
        )

    def load_prop_model(
        self,
        prop_type: str,
        version: str = "latest",
        check_staleness: Optional[bool] = None,
    ) -> "BaseModel":
        """
        Load a player prop prediction model.

        Args:
            prop_type: Type of prop (passing_yards, rushing_yards, etc.)
            version: Model version or "latest"
            check_staleness: Override auto_check_staleness setting

        Returns:
            Loaded prop model
        """
        # Import the appropriate model class
        if prop_type == "passing_yards":
            from .player_props import PassingYardsModel
            model_class = PassingYardsModel
        elif prop_type == "rushing_yards":
            from .player_props import RushingYardsModel
            model_class = RushingYardsModel
        elif prop_type == "receiving_yards":
            from .player_props import ReceivingYardsModel
            model_class = ReceivingYardsModel
        elif prop_type == "receptions":
            from .player_props import ReceptionsModel
            model_class = ReceptionsModel
        else:
            raise ValueError(f"Unknown prop type: {prop_type}")

        return self._load_model(
            model_class=model_class,
            model_name=f"{prop_type}_model",
            version=version,
            check_staleness=check_staleness,
        )

    def _load_model(
        self,
        model_class: Type[BaseModel],
        model_name: str,
        version: str = "latest",
        check_staleness: Optional[bool] = None,
    ) -> BaseModel:
        """
        Load a model from disk.

        Args:
            model_class: Class of model to instantiate
            model_name: Base name of model file
            version: Version string or "latest"
            check_staleness: Whether to check for staleness

        Returns:
            Loaded model instance
        """
        # Build file path
        if version == "latest":
            model_path = self.model_dir / f"{model_name}_latest.joblib"
        else:
            model_path = self.model_dir / f"{model_name}_v{version}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Run 'python -m nfl_bets.scripts.train_models' to train models."
            )

        # Load model
        model = model_class()
        model.load(model_path)

        # Check staleness
        should_check = check_staleness if check_staleness is not None else self.auto_check_staleness
        if should_check and self._latest_game_date:
            if self._is_model_stale(model):
                self._warn_stale_model(model, model_name)

        # Cache loaded model
        cache_key = f"{model_name}_{version}"
        self._loaded_models[cache_key] = model

        return model

    def _is_model_stale(self, model: BaseModel) -> bool:
        """Check if a model is stale (trained before latest game data)."""
        if self._latest_game_date is None:
            return False  # Can't check without latest date

        if model.metadata is None:
            return True  # No metadata, assume stale

        if model.metadata.data_cutoff_date is None:
            return True  # No cutoff date, assume stale

        return model.metadata.data_cutoff_date < self._latest_game_date

    def _warn_stale_model(self, model: BaseModel, model_name: str) -> None:
        """Log a warning about stale model."""
        if model.metadata and model.metadata.data_cutoff_date:
            cutoff = model.metadata.data_cutoff_date.strftime("%Y-%m-%d")
            latest = self._latest_game_date.strftime("%Y-%m-%d") if self._latest_game_date else "unknown"
            logger.warning(
                f"⚠️  Model '{model_name}' is STALE! "
                f"Trained on data through {cutoff}, but games through {latest} are available. "
                f"Run 'python -m nfl_bets.scripts.train_models' to retrain."
            )
        else:
            logger.warning(
                f"⚠️  Model '{model_name}' may be stale (no data cutoff date in metadata). "
                f"Run 'python -m nfl_bets.scripts.train_models' to retrain."
            )

    def is_stale(self, model_type: str) -> bool:
        """
        Check if a model type is stale.

        Args:
            model_type: "spread", "moneyline", "totals", or prop type name

        Returns:
            True if model is stale
        """
        try:
            if model_type == "spread":
                model = self.load_spread_model(check_staleness=False)
            elif model_type == "moneyline":
                model = self.load_moneyline_model(check_staleness=False)
            elif model_type == "totals":
                model = self.load_totals_model(check_staleness=False)
            else:
                model = self.load_prop_model(model_type, check_staleness=False)

            return self._is_model_stale(model)
        except FileNotFoundError:
            return True  # No model = stale

    def get_model_info(self, model_type: str) -> dict:
        """
        Get information about a trained model.

        Args:
            model_type: "spread", "moneyline", "totals", or prop type name

        Returns:
            Dictionary with model metadata
        """
        try:
            if model_type == "spread":
                model = self.load_spread_model(check_staleness=False)
            elif model_type == "moneyline":
                model = self.load_moneyline_model(check_staleness=False)
            elif model_type == "totals":
                model = self.load_totals_model(check_staleness=False)
            else:
                model = self.load_prop_model(model_type, check_staleness=False)

            if model.metadata:
                info = model.metadata.to_dict()
                info["is_stale"] = self._is_model_stale(model)
                return info
            else:
                return {
                    "model_type": model.MODEL_TYPE,
                    "version": model.VERSION,
                    "is_stale": True,
                    "error": "No metadata available",
                }
        except FileNotFoundError:
            return {
                "model_type": model_type,
                "error": "Model not found",
                "is_stale": True,
            }

    def list_available_models(self) -> list[dict]:
        """
        List all available trained models.

        Returns:
            List of model info dictionaries
        """
        models = []

        if not self.model_dir.exists():
            return models

        for model_file in self.model_dir.glob("*_latest.joblib"):
            model_name = model_file.stem.replace("_latest", "")
            model_type = model_name.replace("_model", "")

            try:
                info = self.get_model_info(model_type)
                info["file"] = str(model_file)
                models.append(info)
            except Exception as e:
                models.append({
                    "model_type": model_type,
                    "file": str(model_file),
                    "error": str(e),
                })

        return models

    def check_all_models_fresh(self) -> tuple[bool, list[str]]:
        """
        Check if all models are trained on latest data.

        Returns:
            Tuple of (all_fresh, list_of_stale_models)
        """
        stale_models = []

        model_types = [
            "spread", "moneyline", "totals",
            "passing_yards", "rushing_yards", "receiving_yards", "receptions"
        ]

        for model_type in model_types:
            if self.is_stale(model_type):
                stale_models.append(model_type)

        return len(stale_models) == 0, stale_models

    def clear_cache(self) -> None:
        """Clear the loaded models cache."""
        self._loaded_models.clear()
        logger.debug("Model cache cleared")


async def get_model_manager_with_latest_date() -> ModelManager:
    """
    Create a ModelManager with the latest game date set.

    This is a convenience function that automatically fetches
    the latest game date from nflverse.

    Returns:
        ModelManager with latest_game_date configured
    """
    from nfl_bets.data.sources.nflverse import NFLVerseClient
    from nfl_bets.config.settings import get_settings

    settings = get_settings()
    client = NFLVerseClient(settings)

    # Get current season
    current_year = datetime.now().year
    current_month = datetime.now().month
    season = current_year if current_month >= 9 else current_year - 1

    # Load schedules to find latest completed game
    try:
        schedules = await client.load_schedules([season], force_refresh=True)
        completed = schedules.filter(schedules["result"].is_not_null())

        if len(completed) > 0:
            latest_date = completed.select("gameday").max().item()
            if isinstance(latest_date, str):
                latest_date = datetime.fromisoformat(latest_date)
        else:
            latest_date = None
    except Exception as e:
        logger.warning(f"Could not fetch latest game date: {e}")
        latest_date = None

    manager = ModelManager()
    if latest_date:
        manager.set_latest_game_date(latest_date)

    return manager
