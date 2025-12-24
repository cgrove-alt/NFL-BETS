'use client';

import { useEffect, useState } from 'react';
import {
  getValueBets,
  getBankroll,
  getModelsStatus,
  getJobsStatus,
  ValueBetsResponse,
  BankrollSummary,
  ModelsStatus,
  JobsStatus,
} from '@/lib/api';
import ValueBetCard from '@/components/ValueBetCard';
import BankrollWidget from '@/components/BankrollWidget';
import ModelStatusBadge from '@/components/ModelStatusBadge';

export default function Dashboard() {
  const [valueBets, setValueBets] = useState<ValueBetsResponse | null>(null);
  const [bankroll, setBankroll] = useState<BankrollSummary | null>(null);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null);
  const [jobsStatus, setJobsStatus] = useState<JobsStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [bets, bank, models, jobs] = await Promise.all([
        getValueBets().catch(() => null),
        getBankroll().catch(() => null),
        getModelsStatus().catch(() => null),
        getJobsStatus().catch(() => null),
      ]);

      setValueBets(bets);
      setBankroll(bank);
      setModelsStatus(models);
      setJobsStatus(jobs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">NFL Bets Dashboard</h1>
            <div className="flex items-center gap-4">
              {jobsStatus && (
                <span
                  className={`px-3 py-1 rounded-full text-sm ${
                    jobsStatus.scheduler_running
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {jobsStatus.scheduler_running ? 'Scheduler Running' : 'Scheduler Stopped'}
                </span>
              )}
              <button
                onClick={fetchData}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Value Bets */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-4 mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">Value Bets</h2>
                {valueBets && (
                  <span className="text-sm text-gray-500">
                    {valueBets.count} opportunities
                    {valueBets.last_poll && (
                      <> · Last poll: {new Date(valueBets.last_poll).toLocaleTimeString()}</>
                    )}
                  </span>
                )}
              </div>

              {isLoading ? (
                <div className="animate-pulse space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
                  ))}
                </div>
              ) : valueBets && valueBets.value_bets.length > 0 ? (
                <div className="space-y-4">
                  {valueBets.value_bets.map((bet, index) => (
                    <ValueBetCard key={`${bet.game_id}-${index}`} bet={bet} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p className="text-lg">No value bets found</p>
                  <p className="text-sm mt-1">
                    Check back later or wait for the next polling cycle
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right column - Widgets */}
          <div className="space-y-6">
            {/* Bankroll Widget */}
            <BankrollWidget bankroll={bankroll} isLoading={isLoading} />

            {/* Model Status */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-medium text-gray-500">Model Status</h3>
                {modelsStatus && (
                  <span
                    className={`px-2 py-1 rounded text-xs ${
                      modelsStatus.all_fresh
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}
                  >
                    {modelsStatus.all_fresh ? 'All Fresh' : `${modelsStatus.stale_models.length} Stale`}
                  </span>
                )}
              </div>

              {isLoading ? (
                <div className="animate-pulse space-y-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="h-16 bg-gray-200 rounded-lg"></div>
                  ))}
                </div>
              ) : modelsStatus ? (
                <div className="space-y-2">
                  {Object.values(modelsStatus.models).map((model) => (
                    <ModelStatusBadge key={model.model_type} model={model} />
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Unable to load model status</p>
              )}
            </div>

            {/* Job Status */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-sm font-medium text-gray-500 mb-4">Scheduled Jobs</h3>

              {isLoading ? (
                <div className="animate-pulse space-y-2">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-12 bg-gray-200 rounded-lg"></div>
                  ))}
                </div>
              ) : jobsStatus ? (
                <div className="space-y-2">
                  {jobsStatus.jobs.map((job) => (
                    <div
                      key={job.job_id}
                      className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0"
                    >
                      <div>
                        <p className="font-medium text-sm">{job.name}</p>
                        {job.next_run && (
                          <p className="text-xs text-gray-500">
                            Next: {new Date(job.next_run).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          job.last_status === 'success'
                            ? 'bg-green-100 text-green-800'
                            : job.last_status === 'error'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {job.last_status}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Unable to load job status</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
