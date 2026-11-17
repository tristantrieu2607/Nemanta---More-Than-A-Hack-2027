"""Tristan's match-watching observations as features (not an algorithm). Positions only.

1. Convergence: players near a loose ball all close in on it; if both teams do, it is contested.
   -> per team: closing speed of its 2 nearest players to the ball; number of its players within
      15 m moving towards the ball faster than 2 m/s.
2. Control lasts: a team has it when one player stays close for about 2 s.
   -> per team: length of the current spell of close control (nearest player <= 1.5 m and 0.5 m
      closer than any opponent, ball tracked), and the longest such spell in the last / next 3 s.
3. Off-ball reaction: after a win the attackers push up and the nearest players charge; the
   defending team holds or comes short.
   -> per team: mean speed towards the goal it attacks, share of players sprinting forward
      (> 4 m/s), over the last vs the next 1 and 3 s.
4. Hidden ball: the closest players chase the landing spot.
   -> covered by 1, measured against the ball interpolated through the gap.
5. Restart marking: at a throw-in the restarting team's players are marked, the other team's
   players are free.
   -> per team, players within 25 m of the ball: mean distance to the nearest opponent and the
      number with an opponent within 2 m, over the last vs the next 3 s.

Names avoid "_h"/"_a" except as the team suffix so the home/away mirror swaps them correctly.
"""
import warnings

import numpy as np
import pandas as pd

from src.algos import _features as F

CTRL_M, MARGIN_M, CONVERGE_M, APPROACH_MPS, SPRINT_MPS, MARK_ZONE_M, MARK_M = 1.5, 0.5, 15.0, 2.0, 4.0, 25.0, 2.0


def _past(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def _fut(x, w):
    return _past(np.asarray(x)[::-1], w)[::-1]


def _pmax(x, w):
    return pd.Series(x).rolling(w, min_periods=1).max().to_numpy()


def tristan_features(match):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _tristan(match)


def _tristan(match):
    n = match.n_frames
    xy = match.players_xy.astype(np.float32)
    team = match.track_team
    ball = F.interpolate_ball(match.ball_xy, match.ball_seen).astype(np.float32)
    seen = match.ball_seen
    dist = np.linalg.norm(xy - ball[:, None, :], axis=-1)  # (n, P), NaN where missing
    k = 5
    ddist = np.full_like(dist, np.nan)
    ddist[k:-k] = (dist[2 * k:] - dist[:-2 * k]) * (F.FPS / (2 * k))  # m/s, < 0 = closing in
    vel = np.full_like(xy, np.nan)
    vel[2:-2] = (xy[4:] - xy[:-4]) * (F.FPS / 4)
    right = F.home_attacks_right(match)
    out, names = [], []

    def add(x, name):
        out.append(np.asarray(x, np.float32))
        names.append(name)

    near_t = {}
    for t, tn in ((0, "h"), (1, "a")):
        mine = team == t
        D = np.where(mine[None, :] & ~np.isnan(dist), dist, np.inf)
        idx = np.argsort(D, axis=1)[:, :2]
        nearest = np.take_along_axis(D, idx, 1)
        near_t[t] = nearest[:, 0]
        close = np.take_along_axis(np.where(mine[None, :], ddist, np.nan), idx, 1)
        close[~np.isfinite(nearest)] = np.nan
        closing = np.nanmean(close, 1)
        appr = ((dist < CONVERGE_M) & (ddist < -APPROACH_MPS) & mine[None, :]).sum(1).astype(float)
        sign = np.where(right == (t == 0), 1.0, -1.0)
        fwd = np.where(mine[None, :], vel[..., 0], np.nan) * sign[:, None]
        fwd_mean = np.nanmean(fwd, 1)
        sprint = np.nanmean(np.where(np.isnan(fwd), np.nan, fwd > SPRINT_MPS), 1)
        for w in (25, 75):
            add(_past(closing, w), "tr_close_p%d_%s" % (w, tn))
            add(_fut(closing, w), "tr_close_f%d_%s" % (w, tn))
            add(_past(appr, w), "tr_nappr_p%d_%s" % (w, tn))
            add(_fut(appr, w), "tr_nappr_f%d_%s" % (w, tn))
            p, f = _past(fwd_mean, w), _fut(fwd_mean, w)
            add(p, "tr_fwd_p%d_%s" % (w, tn))
            add(f, "tr_fwd_f%d_%s" % (w, tn))
            add(f - p, "tr_fwd_j%d_%s" % (w, tn))
            add(_fut(sprint, w) - _past(sprint, w), "tr_run_j%d_%s" % (w, tn))
    # control spells
    for t, tn in ((0, "h"), (1, "a")):
        other = near_t[1 - t]
        ctl = seen & (near_t[t] <= CTRL_M) & ((other - near_t[t]) >= MARGIN_M)
        spell = np.zeros(n, np.float32)
        since = np.zeros(n, np.float32)
        for s, e in F.runs(ctl):
            spell[s:e] = (e - s) / F.FPS
            since[s:e] = (np.arange(s, e) - s + 1) / F.FPS
        add(since, "tr_ctl_since_%s" % tn)
        add(_pmax(spell, 75), "tr_ctl_pmax_%s" % tn)
        add(_pmax(spell[::-1], 75)[::-1], "tr_ctl_fmax_%s" % tn)
    # marking near the ball, in chunks (pairwise distances)
    mark = {0: np.full(n, np.nan), 1: np.full(n, np.nan)}
    marked = {0: np.zeros(n), 1: np.zeros(n)}
    for c0 in range(0, n, 20000):
        sl = slice(c0, min(n, c0 + 20000))
        for t in (0, 1):
            A = xy[sl][:, team == t]
            B = xy[sl][:, team == 1 - t]
            pair = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)
            nearest_opp = np.nanmin(np.where(np.isnan(pair), np.inf, pair), axis=2)
            nearest_opp[~np.isfinite(nearest_opp)] = np.nan
            zone = dist[sl][:, team == t] < MARK_ZONE_M
            mark[t][sl] = np.nanmean(np.where(zone, nearest_opp, np.nan), 1)
            marked[t][sl] = (zone & (nearest_opp < MARK_M)).sum(1)
    for t, tn in ((0, "h"), (1, "a")):
        add(_past(mark[t], 75), "tr_mark_p75_%s" % tn)
        add(_fut(mark[t], 75), "tr_mark_f75_%s" % tn)
        add(_past(marked[t], 75), "tr_free_p75_%s" % tn)
        add(_fut(marked[t], 75), "tr_free_f75_%s" % tn)
    return np.stack(out, 1).astype(np.float32), names
