"""
Depth chart analysis for injury replacement predictions.

Analyzes team depth charts to:
- Identify backup players when starters are injured (Out/Doubtful)
- Calculate usage redistribution to teammates
- Provide replacement player expectations
"""
from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl
from loguru import logger


@dataclass
class InjuredPlayer:
    """Information about an injured starter."""

    player_id: str
    player_name: str
    position: str
    team: str
    injury_status: str  # "Out" or "Doubtful"
    baseline_usage: dict[str, float] = field(default_factory=dict)
    # baseline_usage example: {"targets": 8.5, "target_share": 0.22, "carries": 15.0}


@dataclass
class BackupActivation:
    """Information about a backup player stepping into a starter role."""

    backup_player_id: str
    backup_player_name: str
    position: str
    team: str
    depth_order: int
    replacing_player_id: str
    replacing_player_name: str
    expected_usage_boost: dict[str, float] = field(default_factory=dict)
    # expected_usage_boost: multipliers like {"targets": 1.8, "carries": 2.0}


@dataclass
class UsageRedistribution:
    """Usage redistribution to a beneficiary player."""

    beneficiary_player_id: str
    beneficiary_name: str
    position: str
    team: str
    usage_boosts: dict[str, float] = field(default_factory=dict)
    # usage_boosts: multipliers like {"target_share": 1.15, "carries": 1.25}
    reason: str = ""  # e.g., "RB1_out" or "WR1_doubtful"


@dataclass
class InjuryImpactAnalysis:
    """Complete analysis of injury impact on a team."""

    team: str
    injured_starters: list[InjuredPlayer] = field(default_factory=list)
    backup_activations: list[BackupActivation] = field(default_factory=list)
    usage_redistributions: list[UsageRedistribution] = field(default_factory=list)


# Redistribution percentages when a starter is out
# Format: {injured_position: {beneficiary_position: boost_multiplier}}
REDISTRIBUTION_RULES = {
    "RB": {
        # RB1 out: RB2 gets most carries, passing game gets boost
        "RB": {"carries": 1.70, "rushing_yards": 1.70, "targets": 1.40},  # RB2
        "WR": {"targets": 1.12, "target_share": 1.10},  # WRs get more targets
        "TE": {"targets": 1.15, "target_share": 1.12},  # TEs get more targets
    },
    "WR": {
        # WR1 out: Other WRs and TE get redistributed targets
        "WR": {"targets": 1.25, "target_share": 1.20, "air_yards": 1.20},  # Other WRs
        "TE": {"targets": 1.20, "target_share": 1.15},  # TE gets boost
        "RB": {"targets": 1.08, "target_share": 1.05},  # RB gets slight boost
    },
    "TE": {
        # TE1 out: TE2 and WRs get targets
        "TE": {"targets": 1.60, "target_share": 1.50},  # TE2
        "WR": {"targets": 1.15, "target_share": 1.12},  # WRs
    },
    "QB": {
        # QB out: Backup takes over, no redistribution needed
        # Just apply uncertainty penalty to backup
    },
}

# Backup activation boost (when backup becomes starter)
BACKUP_STARTER_BOOST = {
    "QB": {"pass_attempts": 1.0, "passing_yards": 0.85},  # Backup QB less efficient
    "RB": {"carries": 2.0, "rushing_yards": 1.8, "targets": 1.5},
    "WR": {"targets": 1.8, "target_share": 1.6, "receiving_yards": 1.7},
    "TE": {"targets": 1.7, "target_share": 1.5, "receiving_yards": 1.6},
}


class DepthChartAnalyzer:
    """
    Analyzes depth charts to identify backup players and usage redistribution.

    When a starter is Out or Doubtful:
    1. Identifies the next healthy player on the depth chart
    2. Calculates expected usage boost for the backup
    3. Calculates usage redistribution to other position groups
    """

    # Injury statuses that trigger backup activation
    INJURED_STATUSES = {"out", "doubtful", "ir", "pup", "nfi"}

    # Skill positions we care about for props
    SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

    def __init__(self, espn_client):
        """
        Initialize analyzer with ESPN client.

        Args:
            espn_client: ESPNClient instance for fetching depth charts and injuries
        """
        self.espn = espn_client
        self.logger = logger.bind(component="DepthChartAnalyzer")

    async def get_backup_for_position(
        self,
        team: str,
        position: str,
        injured_player_id: str,
        season: int = 2024,
    ) -> Optional[dict[str, Any]]:
        """
        Find the next healthy player on the depth chart for a position.

        Args:
            team: Team abbreviation (e.g., "KC")
            position: Position to find backup for (e.g., "RB")
            injured_player_id: ESPN player ID of injured starter
            season: NFL season year

        Returns:
            Dict with backup player info, or None if no healthy backup:
            {
                "player_id": str,
                "player_name": str,
                "depth_order": int,
                "injury_status": str,  # "ACTIVE" or status
            }
        """
        # Get depth chart
        depth_chart = await self.espn.get_depth_chart(team, season)
        position_players = depth_chart.get(position.upper(), [])

        if not position_players:
            self.logger.debug(f"No depth chart for {team} {position}")
            return None

        # Get current injuries
        injuries = await self.espn.get_team_injuries(team)
        injury_map = {}
        if len(injuries) > 0:
            for row in injuries.iter_rows(named=True):
                injury_map[row.get("player_id", "")] = row.get("status", "").lower()

        # Find injured player's position in depth chart
        injured_idx = None
        for i, player in enumerate(position_players):
            if player.get("player_id") == injured_player_id:
                injured_idx = i
                break

        if injured_idx is None:
            # Player not in depth chart, find first healthy player
            injured_idx = -1

        # Find next healthy player after the injured one
        for player in position_players[injured_idx + 1:]:
            player_id = player.get("player_id", "")
            player_status = injury_map.get(player_id, "active")

            if player_status not in self.INJURED_STATUSES:
                return {
                    "player_id": player_id,
                    "player_name": player.get("name", ""),
                    "depth_order": player.get("depth", 2),
                    "injury_status": player_status.upper() if player_status != "active" else "ACTIVE",
                }

        self.logger.warning(f"No healthy backup found for {team} {position}")
        return None

    async def get_position_depth(
        self,
        team: str,
        position: str,
        season: int = 2024,
        max_depth: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Get all players at a position with their injury status.

        Args:
            team: Team abbreviation
            position: Position to get depth for
            season: NFL season year
            max_depth: Maximum depth to return

        Returns:
            List of players with injury status, sorted by depth
        """
        depth_chart = await self.espn.get_depth_chart(team, season)
        position_players = depth_chart.get(position.upper(), [])[:max_depth]

        if not position_players:
            return []

        # Get injuries
        injuries = await self.espn.get_team_injuries(team)
        injury_map = {}
        if len(injuries) > 0:
            for row in injuries.iter_rows(named=True):
                injury_map[row.get("player_id", "")] = {
                    "status": row.get("status", ""),
                    "injury_type": row.get("injury_type", ""),
                }

        result = []
        for player in position_players:
            player_id = player.get("player_id", "")
            injury_info = injury_map.get(player_id, {})

            result.append({
                "player_id": player_id,
                "player_name": player.get("name", ""),
                "depth_order": player.get("depth", 1),
                "injury_status": injury_info.get("status", "Active").upper(),
                "injury_type": injury_info.get("injury_type", ""),
            })

        return result

    async def analyze_injury_impact(
        self,
        team: str,
        season: int = 2024,
    ) -> InjuryImpactAnalysis:
        """
        Analyze the full injury impact for a team.

        Identifies:
        - Which starters are Out/Doubtful
        - Who will replace them
        - How usage redistributes to other players

        Args:
            team: Team abbreviation
            season: NFL season year

        Returns:
            InjuryImpactAnalysis with all injury impact data
        """
        analysis = InjuryImpactAnalysis(team=team)

        # Get injuries
        injuries = await self.espn.get_team_injuries(team)
        if len(injuries) == 0:
            return analysis

        # Get depth chart
        depth_chart = await self.espn.get_depth_chart(team, season)

        # Find injured starters (depth 1 players who are Out/Doubtful)
        for row in injuries.iter_rows(named=True):
            status = row.get("status", "").lower()
            if status not in self.INJURED_STATUSES:
                continue

            position = row.get("position", "").upper()
            if position not in self.SKILL_POSITIONS:
                continue

            player_id = row.get("player_id", "")
            player_name = row.get("player_name", "")

            # Check if this player is a starter (depth 1)
            position_players = depth_chart.get(position, [])
            is_starter = False
            for p in position_players:
                if p.get("player_id") == player_id and p.get("depth", 99) == 1:
                    is_starter = True
                    break

            if not is_starter:
                continue

            # This is an injured starter
            injured_player = InjuredPlayer(
                player_id=player_id,
                player_name=player_name,
                position=position,
                team=team,
                injury_status=status.upper(),
            )
            analysis.injured_starters.append(injured_player)

            # Find backup
            backup = await self.get_backup_for_position(team, position, player_id, season)
            if backup:
                boost = BACKUP_STARTER_BOOST.get(position, {})
                activation = BackupActivation(
                    backup_player_id=backup["player_id"],
                    backup_player_name=backup["player_name"],
                    position=position,
                    team=team,
                    depth_order=backup["depth_order"],
                    replacing_player_id=player_id,
                    replacing_player_name=player_name,
                    expected_usage_boost=boost.copy(),
                )
                analysis.backup_activations.append(activation)

            # Calculate redistributions to other positions
            redistributions = self._calculate_redistributions(
                team=team,
                injured_position=position,
                injured_player_name=player_name,
                depth_chart=depth_chart,
            )
            analysis.usage_redistributions.extend(redistributions)

        return analysis

    def _calculate_redistributions(
        self,
        team: str,
        injured_position: str,
        injured_player_name: str,
        depth_chart: dict[str, list[dict]],
    ) -> list[UsageRedistribution]:
        """
        Calculate usage redistributions when a position has an injury.

        Args:
            team: Team abbreviation
            injured_position: Position of injured starter
            injured_player_name: Name of injured player
            depth_chart: Full depth chart dict

        Returns:
            List of UsageRedistribution for beneficiary players
        """
        redistributions = []
        rules = REDISTRIBUTION_RULES.get(injured_position, {})

        for beneficiary_position, boosts in rules.items():
            # Get players at beneficiary position
            players = depth_chart.get(beneficiary_position, [])

            for player in players[:2]:  # Top 2 at each position benefit
                # Skip if same position as injured (handled by backup activation)
                if beneficiary_position == injured_position:
                    continue

                redistribution = UsageRedistribution(
                    beneficiary_player_id=player.get("player_id", ""),
                    beneficiary_name=player.get("name", ""),
                    position=beneficiary_position,
                    team=team,
                    usage_boosts=boosts.copy(),
                    reason=f"{injured_position}1_{injured_player_name}_out",
                )
                redistributions.append(redistribution)

        return redistributions

    async def get_player_injury_context(
        self,
        player_id: str,
        team: str,
        position: str,
        season: int = 2024,
    ) -> dict[str, Any]:
        """
        Get injury context for a specific player.

        Returns info about:
        - Player's own injury status
        - Whether they're a backup stepping into starter role
        - Usage boosts from injured teammates

        Args:
            player_id: ESPN player ID
            team: Team abbreviation
            position: Player's position
            season: NFL season year

        Returns:
            Dict with:
            - injury_status: str
            - is_backup_starter: bool (True if stepping into starter role)
            - replacing_player: Optional[str] (name of injured starter)
            - teammate_injury_boosts: dict[str, float] (boosts from injured teammates)
            - uncertainty_multiplier: float
        """
        # Get team injury analysis
        analysis = await self.analyze_injury_impact(team, season)

        # Check if player is an injured starter
        own_status = "ACTIVE"
        for injured in analysis.injured_starters:
            if injured.player_id == player_id:
                own_status = injured.injury_status
                break

        # Check if player is a backup stepping in
        is_backup_starter = False
        replacing_player = None
        backup_boost = {}

        for activation in analysis.backup_activations:
            if activation.backup_player_id == player_id:
                is_backup_starter = True
                replacing_player = activation.replacing_player_name
                backup_boost = activation.expected_usage_boost
                break

        # Check for teammate injury boosts
        teammate_boosts = {}
        for redistribution in analysis.usage_redistributions:
            if redistribution.beneficiary_player_id == player_id:
                # Merge boosts
                for metric, boost in redistribution.usage_boosts.items():
                    existing = teammate_boosts.get(metric, 1.0)
                    teammate_boosts[metric] = existing * boost

        # Combine backup boost and teammate boost
        combined_boosts = backup_boost.copy()
        for metric, boost in teammate_boosts.items():
            existing = combined_boosts.get(metric, 1.0)
            combined_boosts[metric] = existing * boost

        # Calculate uncertainty multiplier
        uncertainty = self._calculate_uncertainty(
            own_status=own_status,
            is_backup_starter=is_backup_starter,
        )

        return {
            "injury_status": own_status,
            "is_backup_starter": is_backup_starter,
            "replacing_player": replacing_player,
            "teammate_injury_boosts": combined_boosts,
            "uncertainty_multiplier": uncertainty,
        }

    def _calculate_uncertainty(
        self,
        own_status: str,
        is_backup_starter: bool,
    ) -> float:
        """
        Calculate uncertainty multiplier based on injury context.

        Args:
            own_status: Player's own injury status
            is_backup_starter: Whether player is backup becoming starter

        Returns:
            Uncertainty multiplier (1.0 = normal, >1.0 = higher uncertainty)
        """
        status_lower = own_status.lower()

        # Base uncertainty from own status
        if status_lower in ["out", "ir"]:
            return 0.0  # Not playing
        elif status_lower == "doubtful":
            return 2.5
        elif status_lower == "questionable":
            return 1.8
        elif status_lower == "probable":
            return 1.2
        else:
            base = 1.0

        # Additional uncertainty for backup stepping in
        if is_backup_starter:
            base *= 1.5  # Role uncertainty

        return base
