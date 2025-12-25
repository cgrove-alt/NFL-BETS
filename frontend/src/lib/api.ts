/**
 * API client for NFL Bets backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://nfl-bets-production-a5a6.up.railway.app';

/**
 * Generic fetch wrapper with error handling.
 */
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

// Types
export interface ValueBet {
  bet_type: string;
  game_id: string;
  description: string;
  model_probability: number;
  model_prediction: number;
  bookmaker: string;
  odds: number;
  implied_probability: number;
  line: number;
  edge: number;
  expected_value: number;
  recommended_stake: number | null;
  urgency: string;
  detected_at: string;
  expires_at: string | null;
}

export interface ValueBetsResponse {
  count: number;
  value_bets: ValueBet[];
  last_poll: string | null;
}

export interface BankrollSummary {
  current_bankroll: number;
  pending_exposure: number;
  available_bankroll: number;
  initial_bankroll: number;
  total_profit: number;
  roi: number;
}

export interface ModelInfo {
  model_type: string;
  model_version: string | null;
  data_cutoff_date: string | null;
  training_date: string | null;
  is_stale: boolean;
  error: string | null;
  metrics: Record<string, number> | null;
}

export interface ModelsStatus {
  all_fresh: boolean;
  stale_models: string[];
  models: Record<string, ModelInfo>;
  checked_at: string;
}

export interface JobStatus {
  job_id: string;
  name: string;
  last_status: string;
  last_run: string | null;
  next_run: string | null;
  error: string | null;
}

export interface JobsStatus {
  scheduler_running: boolean;
  jobs: JobStatus[];
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  components: Record<string, boolean>;
}

export interface PerformanceMetrics {
  total_bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  total_wagered: number;
  total_profit: number;
  roi: number;
  average_edge: number;
  average_stake: number;
}

export interface GameInfo {
  game_id: string;
  home_team: string;
  away_team: string;
  kickoff: string;
  week: number;
  season: number;
  value_bet_count: number;
  best_edge: number | null;
  best_bet_description: string | null;
  model_prediction: number | null;
  model_confidence: number | null;
  vegas_line: number | null;
}

export interface GamesResponse {
  count: number;
  games: GameInfo[];
}

export interface GameDetailResponse {
  game_id: string;
  home_team: string;
  away_team: string;
  kickoff: string;
  week: number;
  season: number;
  value_bets: ValueBet[];
  value_bet_count: number;
}

// API Functions

export async function getHealth(): Promise<HealthStatus> {
  return fetchAPI<HealthStatus>('/api/health');
}

export async function getValueBets(params?: {
  min_edge?: number;
  bet_type?: string;
  urgency?: string;
  limit?: number;
}): Promise<ValueBetsResponse> {
  const searchParams = new URLSearchParams();
  if (params?.min_edge !== undefined) searchParams.set('min_edge', params.min_edge.toString());
  if (params?.bet_type) searchParams.set('bet_type', params.bet_type);
  if (params?.urgency) searchParams.set('urgency', params.urgency);
  if (params?.limit) searchParams.set('limit', params.limit.toString());

  const query = searchParams.toString();
  return fetchAPI<ValueBetsResponse>(`/api/value-bets${query ? `?${query}` : ''}`);
}

export async function getValueBetsSummary(): Promise<{
  total_bets: number;
  by_urgency: Record<string, number>;
  by_bet_type: Record<string, number>;
  average_edge: number;
  total_expected_value: number;
}> {
  return fetchAPI('/api/value-bets/summary');
}

export async function getBankroll(): Promise<BankrollSummary> {
  return fetchAPI<BankrollSummary>('/api/bankroll');
}

export async function getBankrollPerformance(): Promise<PerformanceMetrics> {
  return fetchAPI('/api/bankroll/performance');
}

export async function getModelsStatus(): Promise<ModelsStatus> {
  return fetchAPI<ModelsStatus>('/api/models/status');
}

export async function getJobsStatus(): Promise<JobsStatus> {
  return fetchAPI<JobsStatus>('/api/jobs/status');
}

export async function triggerJob(jobId: string): Promise<{ success: boolean; message: string }> {
  return fetchAPI(`/api/jobs/${jobId}/trigger`, { method: 'POST' });
}

export async function getAnalyticsPerformance(days: number = 30): Promise<{
  metrics: PerformanceMetrics;
  period_start: string;
  period_end: string;
}> {
  return fetchAPI(`/api/analytics/performance?days=${days}`);
}

export async function getBankrollHistory(days: number = 30): Promise<{
  history: Array<{ date: string; balance: number }>;
  period_start: string;
  period_end: string;
}> {
  return fetchAPI(`/api/analytics/bankroll-history?days=${days}`);
}

export async function getGames(params?: {
  week?: number;
}): Promise<GamesResponse> {
  const searchParams = new URLSearchParams();
  if (params?.week !== undefined) searchParams.set('week', params.week.toString());

  const query = searchParams.toString();
  return fetchAPI<GamesResponse>(`/api/games${query ? `?${query}` : ''}`);
}

export async function getGameDetail(gameId: string): Promise<GameDetailResponse> {
  return fetchAPI<GameDetailResponse>(`/api/games/${encodeURIComponent(gameId)}`);
}
