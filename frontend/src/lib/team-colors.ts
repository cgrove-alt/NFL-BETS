/**
 * Official NFL team colors for all 32 teams.
 * Colors sourced from official team brand guidelines.
 */

export interface TeamColors {
  primary: string;
  secondary: string;
  name: string;
}

export const NFL_TEAMS: Record<string, TeamColors> = {
  // AFC East
  BUF: { primary: '#00338D', secondary: '#C60C30', name: 'Bills' },
  MIA: { primary: '#008E97', secondary: '#FC4C02', name: 'Dolphins' },
  NE: { primary: '#002244', secondary: '#C60C30', name: 'Patriots' },
  NYJ: { primary: '#125740', secondary: '#000000', name: 'Jets' },

  // AFC North
  BAL: { primary: '#241773', secondary: '#000000', name: 'Ravens' },
  CIN: { primary: '#FB4F14', secondary: '#000000', name: 'Bengals' },
  CLE: { primary: '#311D00', secondary: '#FF3C00', name: 'Browns' },
  PIT: { primary: '#FFB612', secondary: '#101820', name: 'Steelers' },

  // AFC South
  HOU: { primary: '#03202F', secondary: '#A71930', name: 'Texans' },
  IND: { primary: '#002C5F', secondary: '#A2AAAD', name: 'Colts' },
  JAX: { primary: '#006778', secondary: '#D7A22A', name: 'Jaguars' },
  TEN: { primary: '#0C2340', secondary: '#4B92DB', name: 'Titans' },

  // AFC West
  DEN: { primary: '#FB4F14', secondary: '#002244', name: 'Broncos' },
  KC: { primary: '#E31837', secondary: '#FFB81C', name: 'Chiefs' },
  LV: { primary: '#000000', secondary: '#A5ACAF', name: 'Raiders' },
  LAC: { primary: '#0080C6', secondary: '#FFC20E', name: 'Chargers' },

  // NFC East
  DAL: { primary: '#003594', secondary: '#869397', name: 'Cowboys' },
  NYG: { primary: '#0B2265', secondary: '#A71930', name: 'Giants' },
  PHI: { primary: '#004C54', secondary: '#A5ACAF', name: 'Eagles' },
  WAS: { primary: '#5A1414', secondary: '#FFB612', name: 'Commanders' },

  // NFC North
  CHI: { primary: '#0B162A', secondary: '#C83803', name: 'Bears' },
  DET: { primary: '#0076B6', secondary: '#B0B7BC', name: 'Lions' },
  GB: { primary: '#203731', secondary: '#FFB612', name: 'Packers' },
  MIN: { primary: '#4F2683', secondary: '#FFC62F', name: 'Vikings' },

  // NFC South
  ATL: { primary: '#A71930', secondary: '#000000', name: 'Falcons' },
  CAR: { primary: '#0085CA', secondary: '#101820', name: 'Panthers' },
  NO: { primary: '#D3BC8D', secondary: '#101820', name: 'Saints' },
  TB: { primary: '#D50A0A', secondary: '#34302B', name: 'Buccaneers' },

  // NFC West
  ARI: { primary: '#97233F', secondary: '#000000', name: 'Cardinals' },
  LAR: { primary: '#003594', secondary: '#FFA300', name: 'Rams' },
  SF: { primary: '#AA0000', secondary: '#B3995D', name: '49ers' },
  SEA: { primary: '#002244', secondary: '#69BE28', name: 'Seahawks' },
};

/**
 * Get team colors with fallback for unknown teams.
 */
export function getTeamColors(teamAbbr: string): TeamColors {
  return NFL_TEAMS[teamAbbr] || { primary: '#6B7280', secondary: '#9CA3AF', name: teamAbbr };
}

/**
 * Get a CSS gradient for a team.
 */
export function getTeamGradient(teamAbbr: string, direction: string = 'to right'): string {
  const colors = getTeamColors(teamAbbr);
  return `linear-gradient(${direction}, ${colors.primary}, ${colors.secondary})`;
}
