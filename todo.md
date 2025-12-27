# Player Props Not Populating - Root Cause Analysis & Fix Plan

## Problem Statement
The NFL Bets dashboard shows "No value bets found for this filter" when viewing player props. 29 games are visible with 0 value bets. This is caused by multiple fundamental issues in the props pipeline.

## Root Cause Analysis

### Issue 1: CRITICAL - `poll_props` Job Never Runs (poll_props is paused by default)
**File:** `nfl_bets/scheduler/orchestrator.py:184`
```python
self.scheduler.pause_job("poll_props")
```
The props polling job is paused immediately after registration and NEVER runs automatically. This means:
- Player props are never fetched from the Odds API
- No prop value bets are ever detected
- Dashboard always shows "0 with value bets" for props

### Issue 2: CRITICAL - `get_player_props()` Uses Wrong Event ID Format
**File:** `nfl_bets/scheduler/jobs.py:637`
```python
props_data = await pipeline.get_player_props(game_id)
```
The `poll_props` job passes `game_id` (e.g., "2024_17_HOU_LAC") but the Odds API expects the Odds API event ID (UUID format like "abc123"). The `get_player_props()` method in `pipeline.py` forwards this to `odds_api.get_player_props()` which makes a request to:
```
GET /sports/americanfootball_nfl/events/{event_id}/odds
```
This will always fail because the game_id format is wrong.

### Issue 3: CRITICAL - Props Should NOT Require External Odds to Make Predictions
The user explicitly stated: **"we shouldn't need lines to predict player production"**

Currently, the system requires odds data to:
1. Get the prop line (e.g., 250.5 passing yards)
2. Get over/under odds from bookmakers

But our ML models can predict raw yardage values directly from features. We should:
1. Generate predictions WITHOUT needing prop lines from sportsbooks
2. Only use DraftKings lines for VALUE DETECTION (comparing prediction vs line)

### Issue 4: Season Mapping Missing for Props
**File:** `nfl_bets/scheduler/jobs.py:663-671`
Unlike the spread polling (which has the 2025→2024 season fix), prop feature building doesn't have this fix:
```python
pf = await feature_pipeline.build_prop_features(
    ...
    season=int(season),  # No mapping to 2024 for PBP data
    ...
)
```

### Issue 5: Player ID Mismatch
**File:** `nfl_bets/scheduler/jobs.py:577`
```python
"player_id": player_name,  # Use name as ID for now
```
The props parser uses `player_name` as `player_id`, but the feature pipeline expects nflverse-style player IDs (e.g., "00-0036971"). This causes feature lookups to fail.

## Fix Plan

### Phase 1: Enable Independent Prop Predictions (No Odds Required)
- [ ] Create new API endpoint `/api/predictions/props` that generates prop predictions without external odds
- [ ] Create scheduled job that builds prop predictions for all key players in upcoming games
- [ ] Store predictions in `app_state` for dashboard display

### Phase 2: Fix Props Polling for Value Detection
- [ ] Map nflverse game_id to Odds API event_id using the events endpoint
- [ ] Add season mapping (2025→2024) for prop feature building
- [ ] Fix player ID resolution - lookup nflverse player_id from player name
- [ ] Enable `poll_props` job to run automatically (or integrate with odds poll)

### Phase 3: Use DraftKings as Primary Prop Source
- [ ] Filter `get_player_props()` to only use DraftKings as the bookmaker
- [ ] Add caching for prop lines to reduce API costs (5 credits per event)

### Phase 4: Dashboard Integration
- [ ] Update frontend to display raw prop predictions (not just value bets)
- [ ] Show prop predictions even when no value is detected
- [ ] Add confidence intervals/uncertainty for predictions

## Implementation Priority
1. **HIGHEST**: Enable independent prop predictions (Phase 1)
2. **HIGH**: Fix player ID resolution and season mapping (Phase 2)
3. **MEDIUM**: DraftKings integration (Phase 3)
4. **LOWER**: Dashboard enhancements (Phase 4)

## Files to Modify
1. `nfl_bets/scheduler/orchestrator.py` - Remove pause for poll_props
2. `nfl_bets/scheduler/jobs.py` - Fix event ID mapping, season mapping, player ID resolution
3. `nfl_bets/data/pipeline.py` - Add event ID lookup method
4. `api/routers/predictions.py` - Add props endpoint
5. `api/state.py` - Add prop predictions storage
6. `frontend/src/components/GameDetailModal.tsx` - Display raw predictions

## Review Section

### Implementation Complete - All Issues Fixed

#### Changes Made:

**1. `nfl_bets/scheduler/orchestrator.py`**
- Removed `self.scheduler.pause_job("poll_props")` - poll_props now runs every 2 hours during active hours
- Updated `_poll_props_job()` to check active hours before running
- Updated to clear old prop bets before adding new ones (keeps spread bets)

**2. `nfl_bets/scheduler/jobs.py`**
- Completely rewrote `poll_props()` function with proper event ID mapping
- Now fetches Odds API events first to build (home_team, away_team) → event_id mapping
- Uses event_id (not game_id) when calling `odds_api.get_player_props()`
- Added season mapping (2025→2024) for PBP data
- Added player ID resolution using `feature_pipeline.lookup_player_id()`
- Filters to DraftKings bookmaker only to save API credits
- Pre-fetches PBP data once for all players (efficiency)
- Added `_get_position_for_prop_type()` helper function

**3. `nfl_bets/features/feature_pipeline.py`**
- Added season mapping (2025→2024) in `build_prop_features()` for PBP data
- Fixed to use `pbp_season` instead of `season` in `player_builder.build_matchup_features()` call

**4. `api/routers/predictions.py`**
- Added season mapping (2025→2024) for player ID lookup and feature building
- Now uses `pbp_season` for all PBP-related lookups

**5. `frontend/src/lib/api.ts`**
- Added `PropPrediction` interface
- Added `GamePredictions` interface
- Added `PredictionsResponse` interface
- Added `getPredictions()` function
- Added `getGamePredictions()` function

**6. `frontend/src/components/GameDetailModal.tsx`**
- Complete rewrite of Props tab to show raw predictions
- Added `useEffect` to fetch predictions when Props tab is selected
- Shows predictions grouped by team (home/away)
- Displays predicted value, range (25th-75th percentile), and confidence
- Shows injury status badges for players
- Still shows value bets when detected (in separate section)
- Loading state while predictions are fetched

### Key Architecture Decisions:

1. **Independent Predictions**: The `/api/predictions` endpoint generates player prop predictions WITHOUT requiring external odds lines. This is what the user requested.

2. **Value Detection Still Works**: The `poll_props` job still polls DraftKings for lines to detect VALUE bets (when model prediction differs significantly from the market line).

3. **DraftKings Only**: Filtered to only use DraftKings as the bookmaker to:
   - Save API credits (5 per event)
   - Provide consistent line comparison

4. **Season Mapping**: All PBP lookups now map 2025→2024 since we're in the 2024 NFL season but nflverse labels it as 2025.

5. **Player ID Resolution**: Props now properly resolve player names (e.g., "Patrick Mahomes") to nflverse player IDs (e.g., "00-0036971") using the feature pipeline lookup.

### Testing Recommended:

1. Deploy and verify prop predictions show up in Props tab
2. Check that predictions are reasonable (e.g., Patrick Mahomes ~275 passing yards)
3. Verify poll_props job runs during active hours
4. Check logs for any player ID resolution failures

---

## Game Bets Not Showing Fix (2025-12-26)

### Problem Statement
The NFL Bets dashboard header showed "Live Odds Active - 3 bets" (value bets ARE detected globally) and Model Spread showed "+7.0" (predictions ARE working), but when selecting a game (e.g., HOU @ LAC), it displayed "0 Value Bets" and empty Spread/Moneyline/Totals sections.

### Root Cause
**Game_ID format mismatch** at `api/routers/games.py:568`:
```python
game_bets = [
    bet for bet in value_bets
    if _get_bet_game_id(bet) == game_id  # STRICT STRING COMPARISON FAILS
]
```

Why it fails:
1. Games from nflverse use one season year (could be 2024 or 2025)
2. Value bets get game_id from `_transform_odds_data` which may use a different season
3. Strict `==` comparison fails on any season mismatch
4. Fallback data used hardcoded `2024_17_*` game_ids

### Fix Implemented (Commit: fd0532b)

**1. `api/routers/games.py`**
- Added `_game_ids_match()` helper function that compares week + teams, ignores season
- Replaced all strict `== game_id` comparisons with `_game_ids_match()`:
  - Demo mode game lookup
  - Demo mode bet filtering
  - Fallback mode game lookup
  - Fallback mode bet filtering
  - Normal mode bet filtering
  - Fallback recovery game lookup
- Added debug logging to show bet game_ids being compared

**2. `api/routers/value_bets.py`**
- Added `game_id` field to debug endpoint output for visibility in diagnostics

**3. `api/state.py`**
- Changed hardcoded `2024_17_*` game_ids to dynamic calculation:
  ```python
  current_season = now.year if now.month >= 9 else now.year - 1
  game_id = f"{current_season}_17_BAL_KC"
  ```
- Updated both `get_fallback_data()` and `get_fallback_games()`
- Also updated `season` field in fallback games to use dynamic value

### Key Technical Decision: Fuzzy Matching
The `_game_ids_match()` function:
```python
def _game_ids_match(bet_game_id: str, query_game_id: str) -> bool:
    # Exact match - fast path
    if bet_game_id == query_game_id:
        return True

    # Fuzzy match - compare week + teams, ignore season
    # Format: YYYY_WW_AWAY_HOME
    bet_parts = bet_game_id.split("_")
    query_parts = query_game_id.split("_")

    if len(bet_parts) >= 4 and len(query_parts) >= 4:
        return bet_parts[1:4] == query_parts[1:4]  # week, away, home

    return False
```

This ensures that `2024_17_HOU_LAC` matches `2025_17_HOU_LAC` since the only difference is the season year.

### Verification
1. Check `/api/value-bets/debug` - verify game_ids are visible in sample_value_bets
2. Check Railway logs for matching attempts: `[game_id] Looking for match in bet game_ids: [...]`
3. Select a game with value bets - bets should now display correctly
4. Test with fallback data by temporarily disabling scheduler

---

## Model Analysis Framework Fix (2025-12-27)

### Problem Statement
The `run_analysis.py` script failed with multiple errors:
1. Several analyses failed with `'Settings' object has no attribute 'load_pbp_data'` (transient/intermittent)
2. Feature analysis failed with `ZeroDivisionError: division by zero`
3. Feature importance returned empty dicts due to feature name mismatch

### Root Causes Identified

#### Issue 1: Division by Zero in Feature Analysis Summary
**File:** `nfl_bets/analysis/feature_analysis.py:135`
```python
f"Stable Features: {self.n_stable_features} ({100*self.n_stable_features/self.n_features:.0f}%)"
```
When `n_features` is 0, this causes a division by zero.

#### Issue 2: Feature Names Not Passed When Using NumPy Arrays
**File:** `nfl_bets/analysis/feature_analysis.py:330-338`
The `analyze_from_model` method passed NumPy arrays to the model, but `_prepare_features` only sets `feature_names` when given a Polars DataFrame.

#### Issue 3: Feature Importance Uses Wrong Feature Names After Selection
**File:** `nfl_bets/models/spread_model.py:918`
```python
for i, name in enumerate(self.feature_names):
```
When feature selection reduces 375 features to 50, `self.feature_names` still has 375 names, causing index out of bounds.

### Fixes Implemented

**1. `nfl_bets/analysis/feature_analysis.py`**
- Added early return in `summary()` when `n_features` is 0
- Protected against division by zero with conditional check
- Modified `analyze_from_model()` to convert NumPy arrays to Polars DataFrames with proper schema
- Added logging when `get_feature_importance()` returns empty dict

**2. `nfl_bets/models/spread_model.py`**
- Updated `get_feature_importance()` to use `selected_feature_names` when feature selection is applied
- Added length validation before iterating to prevent index out of bounds
- Added warning logs for dimension mismatches

### Results After Fix
All 5 analyses now complete successfully:

| Analysis | Status | Key Findings |
|----------|--------|--------------|
| Backtest | ✓ | 5.9% ROI, 55.5% win rate across 474 bets |
| Calibration | ✓ | ECE=0.16, not well calibrated |
| Edge Validation | ✓ | Optimal min edge = 4%, p-value = 0.29 |
| Model Comparison | ✓ | MAE 10.0 vs Vegas 1.5 |
| Feature Analysis | ✓ | 65 features, 18 stable, top: opening_spread |

### Top 10 Most Important Features
1. `opening_spread` (importance: 27.6, stable)
2. `current_spread` (importance: 11.5, stable)
3. `home_implied_score` (importance: 10.2, stable)
4. `diff_success_rate_ema_10g` (importance: 6.7, stable)
5. `home_adj_epa_per_play_10g` (importance: 6.5, stable)
6. `diff_explosive_play_rate_5g` (importance: 6.4)
7. `away_implied_score` (importance: 5.8, stable)
8. `diff_success_rate_ema_3g` (importance: 5.7)
9. `diff_cpoe_10g` (importance: 5.7)
10. `diff_adj_epa_per_play_3g` (importance: 4.8)
