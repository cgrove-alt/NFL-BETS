# NFL Betting Model MAE Improvement Plan

## Executive Summary

**Current Performance (Out-of-Sample):**
- Model MAE: **10.46 points**
- Vegas MAE: **9.73 points**
- Gap: **+0.73 points** (model underperforms Vegas)
- ATS Win Rate: **50.6%** (break-even)
- R-squared: **0.103** (very weak predictive power)

**Target Performance:**
- Model MAE: **<9.5 points** (beat Vegas)
- ATS Win Rate: **>52.5%** (profitable after vig)

---

## Root Cause Analysis

### 1. Fundamental Issue: Low Signal-to-Noise Ratio

The NFL is inherently unpredictable:
- Spread standard deviation: **14.17 points**
- Best possible MAE (theoretical): ~8-9 points
- Vegas achieves: **9.73 MAE** (very close to theoretical limit)
- Our model: **10.46 MAE**

**Key Insight:** Even perfect feature engineering can only improve MAE by ~1-2 points.

### 2. Critical Data Quality Issues Found

**33 features have ZERO variance** (completely useless):
- Rest/travel features: `home_rest_days`, `away_rest_days`, `rest_advantage`, etc.
- Weather features: `precipitation_prob`, `is_cold_game`, `is_windy`
- Injury features: All 13 injury features have zero variance
- Betting line features: `opening_spread`, `current_spread`, `line_movement`

**Root cause:** These features are being set to default values (0) instead of actual data.

### 3. Feature Analysis Results

**Best Features (highest correlation with spread):**
| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | diff_success_rate_5g | +0.31 |
| 2 | diff_first_down_rate_10g | +0.31 |
| 3 | diff_success_rate_ema_10g | +0.31 |
| 4 | diff_epa_per_play_10g | +0.30 |

**Key insight:** Best features only have ~0.31 correlation. Vegas likely uses:
- Real-time injury data
- Player-level performance (not just team aggregates)
- Weather at game time
- Line movement / sharp money

### 4. Model Architecture Issues

- Using ensemble with fixed weights (not learned)
- No hyperparameter tuning on out-of-sample data
- 298 features but only 511 training samples = overfitting risk
- High multicollinearity between features

---

## Improvement Plan (Prioritized)

### Phase 1: Fix Broken Features (Critical - Expected: -0.5 to -1.0 MAE)

#### 1.1 Fix Weather Feature Integration
**Problem:** Weather features return 0/NaN for all games
**Solution:**
- Fetch historical weather data for game locations/times
- Use OpenWeatherMap historical API or free weather data source
- Cache weather data during feature building

**Files to modify:**
- `nfl_bets/features/weather_features.py`
- `nfl_bets/features/feature_pipeline.py`

#### 1.2 Fix Injury Feature Integration
**Problem:** All 13 injury features have zero variance
**Solution:**
- Use ESPN injury API (already integrated) for historical injury data
- Build proper injury impact scores based on:
  - Player importance (snap %, target share, etc.)
  - Position value (QB >> RB > WR, etc.)
  - Injury severity

**Files to modify:**
- `nfl_bets/features/injury_features.py`

#### 1.3 Fix Rest/Travel Features
**Problem:** `home_rest_days`, `away_rest_days`, etc. all return 0
**Solution:**
- Calculate from schedule data (already available)
- Add proper bye week detection
- Calculate travel distance from team city coordinates

**Files to modify:**
- `nfl_bets/features/game_features.py`

#### 1.4 Add Historical Betting Line Features
**Problem:** `opening_spread`, `line_movement` features are empty
**Solution:**
- Use The Odds API historical data
- Track line movement from open to close
- Calculate "steam moves" (sharp money indicators)

**Files to modify:**
- `nfl_bets/data/sources/odds_api.py`
- `nfl_bets/features/game_features.py`

---

### Phase 2: Feature Engineering Improvements (Expected: -0.3 to -0.5 MAE)

#### 2.1 Add Vegas Line as Feature
**Critical insight:** Vegas lines are the single best predictor of game outcomes.
**Solution:** Use Vegas spread as a feature input to the model.

This is controversial but powerful: instead of trying to beat Vegas from scratch, use Vegas as a baseline and predict *deviations* from the Vegas line.

**Implementation:**
```python
# Instead of: predict(features) -> spread
# Do: predict([features, vegas_line]) -> adjustment_to_vegas_line
# Final: vegas_line + adjustment
```

#### 2.2 Add Player-Level Features
**Problem:** Current features are team aggregates only
**Solution:** Add QB-specific features:
- QB EPA per dropback
- QB pressure rate
- QB injury status
- Backup QB performance delta

**Files to create:**
- `nfl_bets/features/qb_features.py`

#### 2.3 Reduce Feature Dimensionality
**Problem:** 298 features with only 511 training samples
**Solution:**
- Remove zero-variance features (33 features)
- Remove highly correlated pairs (keep one)
- Use PCA or feature selection to reduce to ~50-100 features

**Implementation:**
```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression
selector = SelectKBest(mutual_info_regression, k=100)
```

---

### Phase 3: Model Architecture Changes (Expected: -0.2 to -0.3 MAE)

#### 3.1 Learn Ensemble Weights
**Problem:** Fixed ensemble weights (XGB: 0.4, LGBM: 0.35, Ridge: 0.25)
**Solution:** Use stacking with a meta-learner

```python
from sklearn.ensemble import StackingRegressor
meta_model = Ridge()
ensemble = StackingRegressor(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('ridge', ridge)],
    final_estimator=meta_model
)
```

#### 3.2 Hyperparameter Tuning with Proper CV
**Problem:** No systematic hyperparameter search
**Solution:** Use Optuna with time-series cross-validation

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

#### 3.3 Regularization to Prevent Overfitting
**Problem:** High feature-to-sample ratio causes overfitting
**Solution:** Increase regularization strength

```python
# LightGBM
lgbm_params = {
    'reg_alpha': 1.0,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'min_child_samples': 20,  # Minimum samples per leaf
}
```

---

### Phase 4: Training Data Improvements (Expected: -0.1 to -0.2 MAE)

#### 4.1 Expand Training Data
**Current:** 511 games (2021-2022)
**Target:** 1000+ games (2019-2022)

More data reduces overfitting and allows more features.

#### 4.2 Sample Weighting
**Problem:** Older games may be less relevant
**Solution:** Weight recent games higher

```python
# Exponential decay weighting
weights = np.exp(-0.1 * (max_week - game_week))
```

#### 4.3 Handle Missing Values Better
**Problem:** Missing values filled with 0 (can distort predictions)
**Solution:** Use proper imputation or indicator variables

---

## Implementation Priority

| Phase | Improvement | Expected MAE Reduction | Effort |
|-------|-------------|----------------------|--------|
| 1.1 | Fix weather features | -0.1 to -0.2 | Medium |
| 1.2 | Fix injury features | -0.2 to -0.3 | Medium |
| 1.3 | Fix rest/travel features | -0.1 to -0.2 | Low |
| 1.4 | Add line features | -0.1 to -0.2 | Medium |
| 2.1 | Vegas line as feature | -0.3 to -0.5 | Low |
| 2.2 | Player-level features | -0.1 to -0.2 | High |
| 2.3 | Feature reduction | -0.1 to -0.2 | Medium |
| 3.1 | Learned ensemble weights | -0.1 | Low |
| 3.2 | Hyperparameter tuning | -0.1 to -0.2 | Medium |
| 3.3 | More regularization | -0.05 to -0.1 | Low |

**Total Expected Improvement:** -1.0 to -2.0 MAE (reaching 9.0-9.5)

---

## Quick Wins (Do First)

### 1. Use Vegas Line as Baseline Feature
This alone could reduce MAE by 0.3-0.5 points with minimal code changes.

### 2. Remove Zero-Variance Features
Drop all 33 broken features - they add noise without signal.

### 3. Reduce Feature Count
Use SelectKBest to keep only top 100 features.

### 4. Increase Regularization
Add `reg_alpha=0.5, reg_lambda=0.5` to LGBM params.

---

## Code Changes Summary

### Immediate Changes (No New Data Required)

```python
# 1. In feature_pipeline.py - filter out zero-variance features
zero_variance_features = [
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    'home_is_short_rest', 'away_is_short_rest', 'home_travel_distance',
    'home_timezone_change', 'is_primetime', 'home_coming_off_bye',
    'away_coming_off_bye', 'opening_spread', 'current_spread',
    'line_movement', 'implied_total', 'home_implied_score',
    'away_implied_score', 'precipitation_prob', 'is_cold_game',
    'is_windy', 'home_team_health', 'away_team_health',
    'health_advantage', 'home_offense_health', 'away_offense_health',
    'home_defense_health', 'away_defense_health',
    'offense_vs_defense_health', 'qb_health_diff', 'skill_health_diff',
    'oline_health_diff', 'dline_health_diff', 'linebacker_health_diff',
    'secondary_health_diff'
]

# 2. In spread_model.py - add Vegas line as feature
features['vegas_spread'] = current_line  # Add to feature dict

# 3. In spread_model.py - increase regularization
lgbm_params = {
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'min_child_samples': 30,
    'max_depth': 6,
}
```

---

## Monitoring & Validation

After each change:
1. Run proper out-of-sample test (train 2021-2022, test 2023-2024)
2. Compare MAE to Vegas benchmark (9.73)
3. Check ATS win rate (target >52.5%)
4. Verify feature importance makes sense

---

## Realistic Expectations

**Vegas employs:**
- Full-time analysts
- Real-time data feeds
- Millions in R&D
- Sharp bettors correcting lines

**We can realistically:**
- Match Vegas: Possible with perfect execution
- Beat Vegas by 0.5+ points consistently: Very difficult
- Beat Vegas by 1+ point: Would make us best in the world

**Profitable betting requires:**
- 52.5% win rate to overcome -110 vig
- ~54% win rate for meaningful profit
- This translates to ~0.3-0.5 MAE advantage over Vegas

---

## Next Steps

1. **Today:** Remove zero-variance features and add Vegas line as feature
2. **This week:** Fix weather, injury, and rest features
3. **Next week:** Implement learned ensemble weights and hyperparameter tuning
4. **Ongoing:** Build historical line tracking for CLV analysis
