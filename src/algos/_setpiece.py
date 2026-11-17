"""Set-piece / restart features from player layout alone (not an algorithm).

A restart is visible in the tracking even when the ball is not: play stops, then one player stands
still at a characteristic spot while the others arrange themselves. Per frame, for each team:

  throwin   a stationary player (< 0.7 m/s) within 1.5 m of a touchline
  corner    a stationary player within 3 m of a corner flag
  goalkick  the team's deepest player stationary inside its own 6-yard-box zone, and no opponent
            within 16.5 m of that goal line
  kickoff   every player of both teams in his own half, and a player of the team within 3 m of the
            centre spot
  freekick  a stationary player with >= 3 opponents within 9.15-12 m of him and in front of him
            (a wall at the regulation distance)

Each flag is smoothed (rolling share over 1 s) so a one-frame coincidence is weak evidence. They
also say WHO restarts, which is the possession owner after the stoppage.
"""
import warnings

import numpy as np

from src.algos import _features as F

STILL_MPS = 0.7


def setpiece_features(match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _setpiece(match)


def _setpiece(match):
    L, W = match.pitch
    xy = match.players_xy
    x, y = xy[..., 0], xy[..., 1]
    team = match.track_team
    v = F.speed(xy)
    still = v < STILL_MPS
    right = F.home_attacks_right(match)  # home attacks x = L
    out, names = [], []
    for t, tn in ((0, "h"), (1, "a")):
        mine = team[None, :] == t
        opp = team[None, :] == 1 - t
        # own goal line x for team t
        own_goal_x = np.where(right == (t == 0), 0.0, L)[:, None]
        throwin = (still & mine & ((y < 1.5) | (y > W - 1.5)) & (x > 1) & (x < L - 1)).any(1)
        corner = (still & mine & (np.minimum(np.hypot(x, y), np.minimum(np.hypot(x - L, y),
                  np.minimum(np.hypot(x, y - W), np.hypot(x - L, y - W)))) < 3.0)).any(1)
        in6 = mine & (np.abs(x - own_goal_x) < 6.0) & (np.abs(y - W / 2) < 10.0) & still
        opp_far = ~((opp & (np.abs(x - own_goal_x) < 16.5)).any(1))
        goalkick = in6.any(1) & opp_far
        # kick-off: everybody in own half, a team-t player at the centre spot
        home_left = right[:, None]
        own_half = np.where(team[None, :] == 0, np.where(home_left, x <= L / 2 + 1, x >= L / 2 - 1),
                            np.where(home_left, x >= L / 2 - 1, x <= L / 2 + 1))
        everyone_home = (own_half | np.isnan(x)).all(1)
        centre = (mine & (np.hypot(x - L / 2, y - W / 2) < 3.0)).any(1)
        kickoff = everyone_home & centre
        # free kick: still player with a wall of >= 3 opponents at 8.5-12 m
        fk = np.zeros(len(x), bool)
        cand = still & mine
        rows = np.flatnonzero(cand.any(1))
        for i in rows[::5]:  # every 5th candidate frame, then smoothed -- cheap and enough
            js = np.flatnonzero(cand[i])
            ox, oy = x[i, opp[0]], y[i, opp[0]]
            for j in js:
                d = np.hypot(ox - x[i, j], oy - y[i, j])
                if np.sum((d > 8.5) & (d < 12.0)) >= 3:
                    fk[i:i + 5] = True
                    break
        for name, flag in (("throwin", throwin), ("corner", corner), ("goalkick", goalkick),
                           ("kickoff", kickoff), ("freekick", fk)):
            out.append(F.rolling_mean(flag.astype(float), 25))
            names.append("sp_%s_%s" % (name, tn))
    out.append(F.rolling_mean(np.nanmean(np.where(np.isnan(v), np.nan, still), 1), 75))
    names.append("sp_still_share_m75")
    return np.stack(out, 1).astype(np.float32), names
