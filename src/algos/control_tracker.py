"""Rule-based possession tracker: control radius + hysteresis + loose-ball rule + dead-ball rule.

Possession rule (CHALLENGE.md): it changes only when a player of the other team has the ball under
control. Here "under control" = the ball is tracked, the team's nearest player is within
`radius_m` and clearly closer than any opponent (`margin_m`), for at least `hold_s` -- a single
touch, a deflection or a header is shorter than that. The change is back-dated to the first frame
of that control. With the ball untracked or contested, the team that had it keeps it.

Dead ball: the ball is out of the pitch by more than `out_m`, or the ball is untracked for a gap of
at least `dead_gap_s` while the players are slow (mean speed over 3 s < `dead_speed`). Dead runs
shorter than `min_dead_s` are dropped. After a stoppage the restarting team is the first team
to control the ball.

Every number is physical; they were set from football sense first and then checked (not searched)
on the external benchmark.
"""
import numpy as np

from src.algos import _features as F

PARAMS = {"radius_m": 1.5, "margin_m": 0.5, "hold_s": 0.4, "dead_gap_s": 3.0, "dead_speed": 2.0,
          "out_m": 1.0, "min_dead_s": 2.0}


def control(match, radius_m, margin_m):
    """Per frame: 0 home / 1 away controls the ball, -1 nobody (or ball not tracked)."""
    X, names = F.frame_features(match)
    col = {n: X[:, i] for i, n in enumerate(names)}
    h, a = col["d1_h"], col["d1_a"]
    seen = match.ball_seen
    ctrl = np.full(match.n_frames, -1, np.int8)
    ctrl[seen & (h < radius_m) & ((a - h) > margin_m)] = 0
    ctrl[seen & (a < radius_m) & ((h - a) > margin_m)] = 1
    return ctrl, col


def dead_mask(match, col, dead_gap_s, dead_speed, out_m, min_dead_s):
    slow = np.nan_to_num(F.rolling_mean(col["v_all"], 75), nan=0.0) < dead_speed
    dead = (col["ball_out"] > out_m) | ((col["gap_len"] >= dead_gap_s) & slow)
    dead &= match.graded
    for s, e in F.runs(dead):
        if e - s < min_dead_s * F.FPS:
            dead[s:e] = False
    return dead


def possession(ctrl, dead, hold):
    """Team per frame from control runs with hysteresis; -1 before anybody controls."""
    n = len(ctrl)
    team = np.full(n, -1, np.int8)
    current, start_of_current = -1, 0
    changes = []  # (frame, team)
    for s, e in F.runs(ctrl >= 0):
        # split a control run by team
        seg = ctrl[s:e]
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(seg)) + 1, [e - s]))
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            t = int(seg[b0])
            if t != current and (b1 - b0) >= hold:
                changes.append((s + b0, t))
                current = t
    # restarts: after each dead run, the first controlling team owns the ball from the restart
    for s, e in F.runs(dead):
        nxt = np.flatnonzero(ctrl[e:] >= 0)
        if len(nxt):
            changes.append((e, int(ctrl[e + nxt[0]])))
    changes.sort()
    for (f, t), nxt in zip(changes, changes[1:] + [(n, None)]):
        team[f:nxt[0]] = t
    first = np.flatnonzero(team >= 0)
    if len(first):
        team[:first[0]] = team[first[0]]
    return team


def predict(match, radius_m=1.5, margin_m=0.5, hold_s=0.4, dead_gap_s=3.0, dead_speed=2.0,
            out_m=1.0, min_dead_s=2.0):
    ctrl, col = control(match, radius_m, margin_m)
    dead = dead_mask(match, col, dead_gap_s, dead_speed, out_m, min_dead_s)
    team = possession(ctrl, dead, int(round(hold_s * F.FPS)))
    labels = np.where(team == 1, "A", "H").astype("<U1")
    labels[dead] = "D"
    labels[~match.graded] = "X"
    return labels
