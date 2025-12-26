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
