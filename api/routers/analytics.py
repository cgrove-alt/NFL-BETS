"""Analytics and performance endpoints."""

from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


class PerformanceMetrics(BaseModel):
    """Performance metrics response."""

    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_wagered: float
    total_profit: float
    roi: float
    average_edge: float
    average_stake: float


class PerformanceByType(BaseModel):
    """Performance breakdown by bet type."""

    bet_type: str
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    roi: float


@router.get("/analytics/performance")
async def get_performance(
    request: Request,
    days: int = Query(30, description="Number of days to include"),
) -> dict[str, Any]:
    """
    Get overall performance analytics.

    Returns win rate, ROI, and other metrics for the specified period.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        return {
            "metrics": PerformanceMetrics(
                total_bets=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0,
                total_wagered=0,
                total_profit=0,
                roi=0,
                average_edge=0,
                average_stake=0,
            ),
            "period_start": (datetime.now() - timedelta(days=days)).isoformat(),
            "period_end": datetime.now().isoformat(),
        }

    summary = app_state.bankroll_manager.get_performance_summary()

    return {
        "metrics": PerformanceMetrics(
            total_bets=summary.get("total_bets", 0),
            wins=summary.get("wins", 0),
            losses=summary.get("losses", 0),
            pushes=summary.get("pushes", 0),
            win_rate=summary.get("win_rate", 0),
            total_wagered=summary.get("total_wagered", 0),
            total_profit=summary.get("total_profit", 0),
            roi=summary.get("roi", 0),
            average_edge=summary.get("average_edge", 0),
            average_stake=summary.get("average_stake", 0),
        ),
        "period_start": (datetime.now() - timedelta(days=days)).isoformat(),
        "period_end": datetime.now().isoformat(),
    }


@router.get("/analytics/by-type")
async def get_performance_by_type(request: Request) -> dict[str, Any]:
    """
    Get performance breakdown by bet type.

    Returns win rate and ROI for each bet type (spread, player props, etc.).
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        return {"by_type": []}

    bankroll = app_state.bankroll_manager
    bets = list(bankroll._bets.values())

    # Group by bet type
    by_type = {}
    for bet in bets:
        if bet.status not in ["WON", "LOST", "PUSH"]:
            continue

        bet_type = bet.bet_type
        if bet_type not in by_type:
            by_type[bet_type] = {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "total_wagered": 0,
                "total_profit": 0,
            }

        by_type[bet_type]["total_bets"] += 1
        by_type[bet_type]["total_wagered"] += bet.stake

        if bet.status == "WON":
            by_type[bet_type]["wins"] += 1
            by_type[bet_type]["total_profit"] += bet.profit or 0
        elif bet.status == "LOST":
            by_type[bet_type]["losses"] += 1
            by_type[bet_type]["total_profit"] -= bet.stake
        elif bet.status == "PUSH":
            by_type[bet_type]["pushes"] += 1

    # Calculate metrics for each type
    result = []
    for bet_type, stats in by_type.items():
        win_loss = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / win_loss if win_loss > 0 else 0
        roi = stats["total_profit"] / stats["total_wagered"] if stats["total_wagered"] > 0 else 0

        result.append(
            PerformanceByType(
                bet_type=bet_type,
                total_bets=stats["total_bets"],
                wins=stats["wins"],
                losses=stats["losses"],
                win_rate=win_rate,
                roi=roi,
            )
        )

    return {"by_type": result}


@router.get("/analytics/bankroll-history")
async def get_bankroll_history(
    request: Request,
    days: int = Query(30, description="Number of days to include"),
) -> dict[str, Any]:
    """
    Get bankroll balance over time.

    Returns daily snapshots of bankroll for charting.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        return {
            "history": [],
            "period_start": (datetime.now() - timedelta(days=days)).isoformat(),
            "period_end": datetime.now().isoformat(),
        }

    bankroll = app_state.bankroll_manager
    initial = float(app_state.settings.initial_bankroll) if app_state.settings else 1000.0

    # Get all settled bets
    bets = list(bankroll._bets.values())
    settled_bets = [b for b in bets if b.settled_at is not None]
    settled_bets.sort(key=lambda b: b.settled_at)

    # Build history
    cutoff = datetime.now() - timedelta(days=days)
    history = []
    running_balance = initial

    # Start with initial balance
    history.append({
        "date": cutoff.date().isoformat(),
        "balance": initial,
    })

    # Add each bet's impact
    for bet in settled_bets:
        if bet.settled_at < cutoff:
            # Update running balance for bets before the period
            if bet.profit:
                running_balance += bet.profit
            continue

        if bet.profit:
            running_balance += bet.profit

        history.append({
            "date": bet.settled_at.date().isoformat(),
            "balance": running_balance,
        })

    # Add current balance
    history.append({
        "date": datetime.now().date().isoformat(),
        "balance": bankroll.current_bankroll,
    })

    return {
        "history": history,
        "period_start": cutoff.isoformat(),
        "period_end": datetime.now().isoformat(),
    }


@router.get("/analytics/model-accuracy")
async def get_model_accuracy(request: Request) -> dict[str, Any]:
    """
    Get model prediction accuracy.

    Compares model predictions to actual outcomes.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        return {
            "accuracy": {},
            "message": "No settled bets available",
        }

    bankroll = app_state.bankroll_manager
    bets = list(bankroll._bets.values())
    settled_bets = [b for b in bets if b.status in ["WON", "LOST"]]

    if not settled_bets:
        return {
            "accuracy": {},
            "message": "No settled bets available",
        }

    # Calculate accuracy by bet type
    accuracy = {}
    for bet in settled_bets:
        if bet.bet_type not in accuracy:
            accuracy[bet.bet_type] = {"correct": 0, "total": 0}

        accuracy[bet.bet_type]["total"] += 1
        if bet.status == "WON":
            accuracy[bet.bet_type]["correct"] += 1

    # Calculate percentages
    for bet_type, stats in accuracy.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "accuracy": accuracy,
        "total_bets": len(settled_bets),
    }
