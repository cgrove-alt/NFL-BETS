# NFL Bets

Real-time NFL betting prediction system with ML-powered value bet detection.

## Features

- **Spread Predictions**: XGBoost ensemble model for game spread predictions
- **Player Props**: Quantile regression models for passing, rushing, and receiving yards
- **Value Detection**: Identifies +EV betting opportunities by comparing model predictions to market odds
- **Kelly Sizing**: Optimal bet sizing using fractional Kelly Criterion
- **Arbitrage Scanner**: Cross-book arbitrage opportunity detection
- **Real-time Dashboard**: Terminal and web-based dashboards for monitoring

## Architecture

- **Backend**: FastAPI REST API with async data pipeline
- **ML Models**: XGBoost, LightGBM, Ridge ensemble
- **Data Sources**: nflverse (play-by-play), The Odds API (live odds)
- **Scheduling**: APScheduler for automated odds polling and model updates

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
python -m nfl_bets.main
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/value-bets` - Current value bet opportunities
- `GET /api/bankroll` - Bankroll status and bet history
- `GET /api/models/status` - Model freshness and metadata
- `GET /api/analytics` - Performance metrics

## Configuration

Key environment variables:
- `ODDS_API_KEY` - The Odds API key (required)
- `INITIAL_BANKROLL` - Starting bankroll amount
- `MIN_EDGE_THRESHOLD` - Minimum edge for value bets (default: 0.03)

## License

MIT
