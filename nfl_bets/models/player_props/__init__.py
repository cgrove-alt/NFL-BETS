"""
Player prop prediction models.

Provides quantile regression models for predicting player props:
- Passing yards (QB)
- Rushing yards (RB)
- Receiving yards (WR/TE)
- Receptions (WR/TE)

Each model uses quantile regression to provide:
- Point prediction (median)
- Full distribution estimate
- Over/under probabilities for any line
- Calibrated probabilities for bet sizing
"""

from .base_prop import BasePropModel, PropPrediction
from .passing_yards import PassingYardsModel
from .rushing_yards import RushingYardsModel
from .receiving_yards import ReceivingYardsModel, ReceptionsModel

__all__ = [
    # Base class
    "BasePropModel",
    "PropPrediction",
    # Prop models
    "PassingYardsModel",
    "RushingYardsModel",
    "ReceivingYardsModel",
    "ReceptionsModel",
]
