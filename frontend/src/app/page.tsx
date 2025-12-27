'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RefreshCw,
  Trophy,
  Target,
  AlertCircle,
  Play,
  Pause,
  Calendar,
  Clock,
  Cpu,
  CheckCircle,
  XCircle,
  Wifi,
  WifiOff,
  Zap,
  TestTube,
  Loader2,
} from 'lucide-react';
import {
  getValueBets,
  getBankroll,
  getModelsStatus,
  getJobsStatus,
  getGames,
  getGameDetail,
  getGamePredictions,
  getDetailedHealth,
  ValueBetsResponse,
  BankrollSummary,
  ModelsStatus,
  JobsStatus,
  GamesResponse,
  GameInfo,
  ValueBet,
  GamePredictions,
  DetailedHealthStatus,
} from '@/lib/api';
import BankrollWidget from '@/components/BankrollWidget';
import ModelStatusBadge from '@/components/ModelStatusBadge';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { BestBets } from '@/components/BestBets';
import { GameSelector } from '@/components/GameSelector';
import { GameBetsDisplay } from '@/components/GameBetsDisplay';

// Live Connection Indicator Component
function LiveConnectionIndicator({
  status,
  betsCount,
  lastUpdate
}: {
  status: 'online' | 'degraded' | 'offline' | 'connecting';
  betsCount: number;
  lastUpdate: string | null;
}) {
  const getStatusConfig = () => {
    switch (status) {
      case 'online':
        return {
          icon: Wifi,
          text: 'Live Odds Active',
          bgColor: 'bg-success/10',
          textColor: 'text-success',
          dotColor: 'bg-success',
          pulse: true,
        };
      case 'degraded':
        return {
          icon: Zap,
          text: 'Connecting...',
          bgColor: 'bg-warning/10',
          textColor: 'text-warning',
          dotColor: 'bg-warning',
          pulse: true,
        };
      case 'connecting':
        return {
          icon: Zap,
          text: 'Connecting...',
          bgColor: 'bg-brand-500/10',
          textColor: 'text-brand-500',
          dotColor: 'bg-brand-500',
          pulse: true,
        };
      case 'offline':
      default:
        return {
          icon: WifiOff,
          text: 'Backend Offline',
          bgColor: 'bg-danger/10',
          textColor: 'text-danger',
          dotColor: 'bg-danger',
          pulse: false,
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bgColor}`}>
      {/* Pulsing dot */}
      <span className="relative flex h-2 w-2">
        {config.pulse && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.dotColor} opacity-75`} />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${config.dotColor}`} />
      </span>

      <Icon className={`w-3.5 h-3.5 ${config.textColor}`} />
      <span className={`text-xs font-medium ${config.textColor}`}>
        {config.text}
      </span>

      {status === 'online' && betsCount > 0 && (
        <span className="text-xs text-text-muted">
          • {betsCount} bets
        </span>
      )}
    </div>
  );
}

// Maximum retry attempts for cold start
const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 3000;

export default function Dashboard() {
  const [valueBets, setValueBets] = useState<ValueBetsResponse | null>(null);
  const [bankroll, setBankroll] = useState<BankrollSummary | null>(null);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null);
  const [jobsStatus, setJobsStatus] = useState<JobsStatus | null>(null);
  const [gamesData, setGamesData] = useState<GamesResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Live connection state
  const [connectionStatus, setConnectionStatus] = useState<'online' | 'degraded' | 'offline' | 'connecting'>('connecting');
  const [healthData, setHealthData] = useState<DetailedHealthStatus | null>(null);

  // Demo mode and cold start state
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [isBackendInitializing, setIsBackendInitializing] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  // Game selection state
  const [selectedGame, setSelectedGame] = useState<GameInfo | null>(null);
  const [selectedGameBets, setSelectedGameBets] = useState<ValueBet[]>([]);
  const [selectedGamePredictions, setSelectedGamePredictions] = useState<GamePredictions | null>(null);
  const [isLoadingGameDetail, setIsLoadingGameDetail] = useState(false);

  // Track if we've auto-selected best game
  const hasAutoSelected = useRef(false);

  // Poll health endpoint for live connection status
  const pollHealth = useCallback(async () => {
    try {
      const health = await getDetailedHealth();
      setHealthData(health);
      setConnectionStatus(health.status);

      // Update demo mode and initializing status from health
      if (health.demo_mode !== undefined) {
        setIsDemoMode(health.demo_mode);
      }
      if (health.is_initializing !== undefined) {
        setIsBackendInitializing(health.is_initializing);
      }
    } catch {
      setConnectionStatus('offline');
      setHealthData(null);
    }
  }, []);

  // Auto-select best game logic
  const autoSelectBestGame = useCallback((games: GameInfo[]) => {
    if (hasAutoSelected.current || games.length === 0) return;

    // Find the game with the highest value_bet_count
    const bestGame = games.reduce((best, game) => {
      if (game.value_bet_count > (best?.value_bet_count || 0)) {
        return game;
      }
      // Tie-breaker: use best_edge
      if (game.value_bet_count === (best?.value_bet_count || 0)) {
        if ((game.best_edge || 0) > (best?.best_edge || 0)) {
          return game;
        }
      }
      return best;
    }, null as GameInfo | null);

    if (bestGame && bestGame.value_bet_count > 0) {
      hasAutoSelected.current = true;
      handleSelectGame(bestGame);
    }
  }, []);

  const fetchData = async (showRefreshing = false, currentRetry = 0) => {
    if (showRefreshing) setIsRefreshing(true);
    try {
      const [bets, bank, models, jobs, games] = await Promise.all([
        getValueBets().catch(() => null),
        getBankroll().catch(() => null),
        getModelsStatus().catch(() => null),
        getJobsStatus().catch(() => null),
        getGames().catch(() => null),
      ]);

      setValueBets(bets);
      setBankroll(bank);
      setModelsStatus(models);
      setJobsStatus(jobs);
      setGamesData(games);
      setError(null);

      // Check for demo mode from games response
      if (games?.is_demo) {
        setIsDemoMode(true);
      }

      // Check for cold start / initializing state with retry logic
      if (games?.is_initializing && games?.count === 0 && games?.retry_after_seconds) {
        setIsBackendInitializing(true);
        setRetryCount(currentRetry);

        // Auto-retry up to MAX_RETRY_ATTEMPTS
        if (currentRetry < MAX_RETRY_ATTEMPTS) {
          const retryDelay = (games.retry_after_seconds || 3) * 1000;
          console.log(`Backend initializing... Retry ${currentRetry + 1}/${MAX_RETRY_ATTEMPTS} in ${retryDelay}ms`);

          setTimeout(() => {
            fetchData(false, currentRetry + 1);
          }, retryDelay);
          return; // Don't finish loading yet
        } else {
          console.log('Max retries reached - showing current state');
        }
      } else {
        setIsBackendInitializing(false);
        setRetryCount(0);
      }

      // Auto-select best game on first load
      if (games?.games && !hasAutoSelected.current) {
        autoSelectBestGame(games.games);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchData();
    pollHealth();

    // Poll health every 5 seconds for live status
    const healthInterval = setInterval(pollHealth, 5000);

    // Refresh data every 30 seconds
    const dataInterval = setInterval(() => fetchData(false), 30000);

    return () => {
      clearInterval(healthInterval);
      clearInterval(dataInterval);
    };
  }, [pollHealth]);

  const handleSelectGame = async (game: GameInfo | null) => {
    setSelectedGame(game);

    if (!game) {
      setSelectedGameBets([]);
      setSelectedGamePredictions(null);
      return;
    }

    setIsLoadingGameDetail(true);

    try {
      // Fetch both game detail and predictions in parallel
      const [detail, predictions] = await Promise.all([
        getGameDetail(game.game_id).catch(() => null),
        getGamePredictions(game.game_id).catch(() => null),
      ]);

      if (detail) {
        setSelectedGameBets(detail.value_bets);
      } else {
        // Fall back to filtering from all value bets
        const gameBets = valueBets?.value_bets.filter(
          bet => bet.game_id === game.game_id
        ) || [];
        setSelectedGameBets(gameBets);
      }

      setSelectedGamePredictions(predictions);
    } catch {
      // Fall back to filtering from all value bets
      const gameBets = valueBets?.value_bets.filter(
        bet => bet.game_id === game.game_id
      ) || [];
      setSelectedGameBets(gameBets);
      setSelectedGamePredictions(null);
    } finally {
      setIsLoadingGameDetail(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-surface-primary border-b border-surface-border sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-brand-600 rounded-lg">
                <Trophy className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-text-primary">NFL Bets</h1>
            </div>

            <div className="flex items-center gap-3">
              {/* Live Connection Indicator */}
              <LiveConnectionIndicator
                status={connectionStatus}
                betsCount={healthData?.bets_in_memory || 0}
                lastUpdate={healthData?.last_update || null}
              />

              {jobsStatus && (
                <Badge
                  variant={jobsStatus.scheduler_running ? 'success' : 'danger'}
                  size="md"
                >
                  {jobsStatus.scheduler_running ? (
                    <Play className="w-3 h-3 mr-1" />
                  ) : (
                    <Pause className="w-3 h-3 mr-1" />
                  )}
                  {jobsStatus.scheduler_running ? 'Scheduler Running' : 'Scheduler Stopped'}
                </Badge>
              )}

              <ThemeToggle />

              <Button
                onClick={() => fetchData(true)}
                variant="primary"
                size="sm"
                loading={isRefreshing}
                icon={<RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />}
              >
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Demo Mode Banner */}
      <AnimatePresence>
        {isDemoMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-brand-500/10 border-b border-brand-500/20"
          >
            <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
              <div className="flex items-center gap-3">
                <TestTube className="w-5 h-5 text-brand-500 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-brand-600 dark:text-brand-400">Demo Mode Active</p>
                  <p className="text-xs text-brand-500/70">
                    Showing sample data for demonstration. Live odds will appear when backend connects to real data sources.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Backend Initializing Banner */}
      <AnimatePresence>
        {isBackendInitializing && !isDemoMode && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-warning/10 border-b border-warning/20"
          >
            <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-warning flex-shrink-0 animate-spin" />
                <div>
                  <p className="text-sm font-medium text-warning">System Initializing...</p>
                  <p className="text-xs text-warning/70">
                    Loading live odds data. {retryCount > 0 ? `Retry ${retryCount}/${MAX_RETRY_ATTEMPTS}` : 'Please wait...'}
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Backend Offline Warning Banner */}
      <AnimatePresence>
        {connectionStatus === 'offline' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-danger/10 border-b border-danger/20"
          >
            <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
              <div className="flex items-center gap-3">
                <WifiOff className="w-5 h-5 text-danger flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-danger">Backend Offline - Check Connection</p>
                  <p className="text-xs text-danger/70">
                    Unable to connect to the NFL Bets API. Data may be stale.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 bg-danger-light dark:bg-danger/10 border border-danger/20 rounded-xl text-danger flex items-center gap-3"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Best Bets & Game Selector */}
          <div className="lg:col-span-2 space-y-6">
            {/* Best Bets Section */}
            <BestBets
              bets={valueBets?.value_bets || []}
              isLoading={isLoading}
            />

            {/* Game Selector Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-5"
            >
              <div className="flex items-center gap-2 mb-4">
                <Calendar className="w-4 h-4 text-brand-500" />
                <h2 className="text-lg font-semibold text-text-primary">Game Bets</h2>
                {gamesData && (
                  <Badge variant="default" size="sm">
                    {gamesData.count} games
                  </Badge>
                )}
                {selectedGame && selectedGame.value_bet_count > 0 && (
                  <Badge variant="success" size="sm">
                    <Zap className="w-3 h-3 mr-1" />
                    {selectedGame.value_bet_count} value bets
                  </Badge>
                )}
              </div>

              {/* Game Selector Dropdown */}
              <GameSelector
                games={gamesData?.games || []}
                selectedGame={selectedGame}
                onSelectGame={handleSelectGame}
                isLoading={isLoading}
              />

              {/* Selected Game Bets Display */}
              <AnimatePresence mode="wait">
                {selectedGame && (
                  <motion.div
                    key={selectedGame.game_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.2 }}
                    className="mt-6"
                  >
                    <GameBetsDisplay
                      game={selectedGame}
                      valueBets={selectedGameBets}
                      predictions={selectedGamePredictions}
                      isLoading={isLoadingGameDetail}
                    />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Empty State when no game selected */}
              {!selectedGame && !isLoading && (
                <div className="text-center py-12 mt-4">
                  <Target className="w-12 h-12 mx-auto text-text-muted mb-3" />
                  <p className="text-lg text-text-secondary">Select a game above</p>
                  <p className="text-sm text-text-muted mt-1">
                    View moneyline, spread, and player prop bets
                  </p>
                </div>
              )}
            </motion.div>
          </div>

          {/* Right column - Widgets */}
          <div className="space-y-6">
            {/* Bankroll Widget */}
            <BankrollWidget bankroll={bankroll} isLoading={isLoading} />

            {/* Model Status */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-4"
            >
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-brand-500" />
                  <h3 className="text-sm font-medium text-text-secondary">Model Status</h3>
                </div>
                {modelsStatus && (
                  <Badge
                    variant={modelsStatus.all_fresh ? 'success' : 'warning'}
                    size="sm"
                  >
                    {modelsStatus.all_fresh ? (
                      <>
                        <CheckCircle className="w-3 h-3 mr-1" />
                        All Fresh
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-3 h-3 mr-1" />
                        {modelsStatus.stale_models.length} Stale
                      </>
                    )}
                  </Badge>
                )}
              </div>

              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-16 bg-surface-secondary rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : modelsStatus ? (
                <div className="space-y-2">
                  {Object.values(modelsStatus.models).map((model) => (
                    <ModelStatusBadge key={model.model_type} model={model} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-4">
                  <XCircle className="w-8 h-8 mx-auto text-text-muted mb-2" />
                  <p className="text-text-muted text-sm">Unable to load model status</p>
                </div>
              )}
            </motion.div>

            {/* Job Status */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-surface-primary rounded-xl shadow-card border border-surface-border p-4"
            >
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-4 h-4 text-brand-500" />
                <h3 className="text-sm font-medium text-text-secondary">Scheduled Jobs</h3>
              </div>

              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-12 bg-surface-secondary rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : jobsStatus ? (
                <div className="space-y-2">
                  {jobsStatus.jobs.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex justify-between items-center py-2.5 px-3 bg-surface-secondary rounded-lg hover:bg-surface-elevated transition-colors"
                    >
                      <div>
                        <p className="font-medium text-sm text-text-primary">{job.name}</p>
                        {job.next_run && (
                          <p className="flex items-center gap-1 text-xs text-text-muted mt-0.5">
                            <Clock className="w-3 h-3" />
                            Next: {new Date(job.next_run).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                      <Badge
                        variant={
                          job.last_status === 'success'
                            ? 'success'
                            : job.last_status === 'error'
                            ? 'danger'
                            : 'default'
                        }
                        size="sm"
                      >
                        {job.last_status}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4">
                  <XCircle className="w-8 h-8 mx-auto text-text-muted mb-2" />
                  <p className="text-text-muted text-sm">Unable to load job status</p>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
}
