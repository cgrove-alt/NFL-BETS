'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Calendar, Clock, Check } from 'lucide-react';
import { GameInfo } from '@/lib/api';
import { getTeamColors } from '@/lib/team-colors';

interface GameSelectorProps {
  games: GameInfo[];
  selectedGame: GameInfo | null;
  onSelectGame: (game: GameInfo | null) => void;
  isLoading?: boolean;
}

export function GameSelector({ games, selectedGame, onSelectGame, isLoading }: GameSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsOpen(false);
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const today = new Date();
    const tomorrow = new Date(Date.now() + 86400000);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === tomorrow.toDateString()) return 'Tomorrow';

    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatTime = (dateStr: string) => {
    return new Date(dateStr).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const handleSelect = (game: GameInfo) => {
    onSelectGame(game);
    setIsOpen(false);
  };

  if (isLoading) {
    return (
      <div className="h-12 bg-surface-secondary rounded-lg animate-pulse" />
    );
  }

  return (
    <div ref={dropdownRef} className="relative">
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          w-full flex items-center justify-between px-4 py-3
          bg-surface-primary border rounded-xl
          transition-all duration-200
          focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:outline-none
          ${isOpen
            ? 'border-brand-500 shadow-card-hover'
            : 'border-surface-border hover:border-brand-300 dark:hover:border-brand-700'
          }
        `}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        {selectedGame ? (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className="px-2 py-0.5 rounded text-xs font-bold text-white"
                style={{ backgroundColor: getTeamColors(selectedGame.away_team).primary }}
              >
                {selectedGame.away_team}
              </div>
              <span className="text-text-muted text-sm">@</span>
              <div
                className="px-2 py-0.5 rounded text-xs font-bold text-white"
                style={{ backgroundColor: getTeamColors(selectedGame.home_team).primary }}
              >
                {selectedGame.home_team}
              </div>
            </div>
            <span className="text-sm text-text-muted">
              {formatDate(selectedGame.kickoff)} • {formatTime(selectedGame.kickoff)}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-text-secondary">
            <Calendar className="w-4 h-4" />
            <span>Select a game to view bets</span>
          </div>
        )}
        <ChevronDown
          className={`w-5 h-5 text-text-muted transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-2 bg-surface-primary border border-surface-border rounded-xl shadow-elevated overflow-hidden"
            role="listbox"
          >
            {/* Clear Selection Option */}
            {selectedGame && (
              <button
                onClick={() => {
                  onSelectGame(null);
                  setIsOpen(false);
                }}
                className="w-full px-4 py-3 text-left text-sm text-text-muted hover:bg-surface-secondary transition-colors border-b border-surface-border"
              >
                Clear selection
              </button>
            )}

            {/* Game List */}
            <div className="max-h-80 overflow-y-auto">
              {games.length === 0 ? (
                <div className="px-4 py-6 text-center text-text-muted">
                  <Calendar className="w-8 h-8 mx-auto mb-2" />
                  <p>No games available</p>
                </div>
              ) : (
                games.map((game) => {
                  const isSelected = selectedGame?.game_id === game.game_id;
                  const awayColors = getTeamColors(game.away_team);
                  const homeColors = getTeamColors(game.home_team);

                  return (
                    <button
                      key={game.game_id}
                      onClick={() => handleSelect(game)}
                      className={`
                        w-full px-4 py-3 flex items-center justify-between
                        transition-colors
                        ${isSelected
                          ? 'bg-brand-50 dark:bg-brand-900/20'
                          : 'hover:bg-surface-secondary'
                        }
                      `}
                      role="option"
                      aria-selected={isSelected}
                    >
                      <div className="flex flex-col items-start gap-1">
                        <div className="flex items-center gap-2">
                          <div
                            className="px-2 py-0.5 rounded text-xs font-bold text-white"
                            style={{ backgroundColor: awayColors.primary }}
                          >
                            {game.away_team}
                          </div>
                          <span className="text-text-muted text-sm">@</span>
                          <div
                            className="px-2 py-0.5 rounded text-xs font-bold text-white"
                            style={{ backgroundColor: homeColors.primary }}
                          >
                            {game.home_team}
                          </div>
                          {game.value_bet_count > 0 && (
                            <span className="px-1.5 py-0.5 text-xs font-medium bg-success-light text-success-dark dark:bg-success/20 dark:text-success rounded-full">
                              {game.value_bet_count} bets
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1.5 text-xs text-text-muted">
                          <Clock className="w-3 h-3" />
                          {formatDate(game.kickoff)} • {formatTime(game.kickoff)}
                        </div>
                      </div>

                      {isSelected && (
                        <Check className="w-5 h-5 text-brand-600 dark:text-brand-400" />
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
