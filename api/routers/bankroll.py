"""Bankroll management endpoints."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class BankrollSummary(BaseModel):
    """Bankroll summary response."""

    current_bankroll: float
    pending_exposure: float
    available_bankroll: float
    initial_bankroll: float
    total_profit: float
    roi: float


class BetRecordResponse(BaseModel):
    """Bet record response."""

    bet_id: str
    placed_at: str
    game_id: str
    bet_type: str
    description: str
    bookmaker: str
    odds: int
    line: float
    stake: float
    model_probability: float
    edge: float
    status: str
    settled_at: Optional[str] = None
    payout: Optional[float] = None
    profit: Optional[float] = None


class PlaceBetRequest(BaseModel):
    """Request to record a placed bet."""

    game_id: str
    bet_type: str
    description: str
    bookmaker: str
    odds: int
    line: float
    stake: float
    model_probability: float
    edge: float


class SettleBetRequest(BaseModel):
    """Request to settle a bet."""

    result: str  # WON, LOST, PUSH, VOIDED
    actual_outcome: Optional[float] = None


@router.get("/bankroll", response_model=BankrollSummary)
async def get_bankroll(request: Request) -> dict[str, Any]:
    """
    Get current bankroll status.

    Returns balance, pending exposure, and performance metrics.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        raise HTTPException(status_code=503, detail="Bankroll manager not initialized")

    bankroll = app_state.bankroll_manager
    initial = float(app_state.settings.initial_bankroll) if app_state.settings else 1000.0

    return {
        "current_bankroll": bankroll.current_bankroll,
        "pending_exposure": bankroll.pending_exposure,
        "available_bankroll": bankroll.available_bankroll,
        "initial_bankroll": initial,
        "total_profit": bankroll.current_bankroll - initial,
        "roi": (bankroll.current_bankroll - initial) / initial if initial > 0 else 0,
    }


@router.get("/bankroll/performance")
async def get_performance(request: Request) -> dict[str, Any]:
    """
    Get detailed performance summary.

    Returns win/loss record, ROI by bet type, and other metrics.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        raise HTTPException(status_code=503, detail="Bankroll manager not initialized")

    return app_state.bankroll_manager.get_performance_summary()


@router.get("/bankroll/bets")
async def get_bets(
    request: Request,
    status: Optional[str] = None,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Get bet history.

    Returns list of all bets, optionally filtered by status.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        raise HTTPException(status_code=503, detail="Bankroll manager not initialized")

    bankroll = app_state.bankroll_manager

    # Get all bets
    all_bets = list(bankroll._bets.values())

    # Filter by status if provided
    if status:
        all_bets = [b for b in all_bets if b.status.upper() == status.upper()]

    # Sort by placed_at descending
    all_bets.sort(key=lambda b: b.placed_at, reverse=True)

    # Limit results
    all_bets = all_bets[:limit]

    # Convert to response format
    response_bets = []
    for bet in all_bets:
        response_bets.append(
            BetRecordResponse(
                bet_id=bet.bet_id,
                placed_at=bet.placed_at.isoformat(),
                game_id=bet.game_id,
                bet_type=bet.bet_type,
                description=bet.description,
                bookmaker=bet.bookmaker,
                odds=bet.odds,
                line=bet.line,
                stake=bet.stake,
                model_probability=bet.model_probability,
                edge=bet.edge,
                status=bet.status.value if hasattr(bet.status, "value") else bet.status,
                settled_at=bet.settled_at.isoformat() if bet.settled_at else None,
                payout=bet.payout,
                profit=bet.profit,
            )
        )

    return {
        "count": len(response_bets),
        "bets": response_bets,
    }


@router.post("/bankroll/bets")
async def place_bet(request: Request, bet_request: PlaceBetRequest) -> dict[str, Any]:
    """
    Record a placed bet.

    Use this to track bets you've placed manually.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        raise HTTPException(status_code=503, detail="Bankroll manager not initialized")

    bankroll = app_state.bankroll_manager

    # Check if we can place this bet
    if not bankroll.can_place_bet(bet_request.stake):
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient available bankroll. Available: ${bankroll.available_bankroll:.2f}",
        )

    # Create a value bet object to pass to place_bet
    from nfl_bets.betting.value_detector import ValueBet, Urgency

    value_bet = ValueBet(
        bet_type=bet_request.bet_type,
        game_id=bet_request.game_id,
        description=bet_request.description,
        model_probability=bet_request.model_probability,
        model_prediction=0,  # Not needed for recording
        bookmaker=bet_request.bookmaker,
        odds=bet_request.odds,
        implied_probability=0,  # Not needed for recording
        line=bet_request.line,
        edge=bet_request.edge,
        expected_value=0,  # Not needed for recording
        recommended_stake=bet_request.stake,
        urgency=Urgency.MEDIUM,
        detected_at=datetime.now(),
    )

    # Place the bet
    bet_record = bankroll.place_bet(value_bet, bet_request.stake)

    return {
        "success": True,
        "bet_id": bet_record.bet_id,
        "message": f"Bet recorded: ${bet_request.stake:.2f} on {bet_request.description}",
    }


@router.post("/bankroll/bets/{bet_id}/settle")
async def settle_bet(
    request: Request,
    bet_id: str,
    settle_request: SettleBetRequest,
) -> dict[str, Any]:
    """
    Settle a pending bet.

    Marks the bet as won, lost, push, or voided.
    """
    app_state = request.app.state.app_state

    if not app_state.bankroll_manager:
        raise HTTPException(status_code=503, detail="Bankroll manager not initialized")

    bankroll = app_state.bankroll_manager

    try:
        bankroll.settle_bet(
            bet_id=bet_id,
            result=settle_request.result,
            actual_outcome=settle_request.actual_outcome,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "success": True,
        "bet_id": bet_id,
        "result": settle_request.result,
        "message": f"Bet {bet_id} settled as {settle_request.result}",
    }
