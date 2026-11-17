"""Shared building blocks (not an algorithm): team-level per-frame features and a Viterbi decoder.

Everything is computed from positions only (players_xy, ball_xy, track_team, halves, pitch), so it
behaves the same on BAR-ATH (`common.Match`) and on the external benchmark (`ExternalMatch`). No
player identities, no provider speed field (the ATD speed unit is unverified), no Smart Tagging.
"""
import warnings

import numpy as np

FPS = 25


# -- small helpers -------------------------------------------------------------------------------

def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def rolling_mean(x, width):
    """Centred moving average over `width` frames, NaN-aware (NaN where the window has no data)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - width // 2), np.minimum(len(x), i + width // 2 + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt == 0] = np.nan
    return out


def speed(xy, k=2):
    """|velocity| in m/s from positions, central difference over +/- k frames. (n, ...)"""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    v[k:-k] = np.linalg.norm(xy[2 * k:] - xy[:-2 * k], axis=-1) * (FPS / (2 * k))
    return v


def gap_info(seen):
    """Per frame: length of the ball gap it sits in (s), time since last / until next sighting (s)."""
    n = len(seen)
    gap_len = np.zeros(n, np.float32)
    since = np.zeros(n, np.float32)
    until = np.zeros(n, np.float32)
    for s, e in runs(~seen):
        gap_len[s:e] = (e - s) / FPS
        since[s:e] = (np.arange(s, e) - s + 1) / FPS
        until[s:e] = (e - np.arange(s, e)) / FPS
    return gap_len, since, until


def interpolate_ball(ball, seen):
    """Linear interpolation of the ball across gaps (ends held)."""
    out = ball.astype(np.float64).copy()
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return out
    for d in range(2):
        out[:, d] = np.interp(np.arange(len(ball)), idx, ball[idx, d])
    return out


def home_attacks_right(match):
    """Per frame bool: does home attack towards x = L? From team centroids per half (no GK needed)."""
    out = np.zeros(match.n_frames, bool)
    team = match.track_team
    for _, first, last in match.halves:
        xy = match.players_xy[first:last + 1, :, 0]
        with np.errstate(all="ignore"):
            hx = np.nanmedian(np.nanmean(xy[:, team == 0], axis=1))
            ax = np.nanmedian(np.nanmean(xy[:, team == 1], axis=1))
        out[first:last + 1] = hx < ax  # home sits on the left on average -> attacks right
    return out


# -- features ------------------------------------------------------------------------------------

def frame_features(match):
    """(n_frames, n_features) float32 + names. Home-centric; see `mirror` for the away view."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _frame_features(match)


def _frame_features(match):
    L, W = match.pitch
    seen = match.ball_seen.copy()
    ball = interpolate_ball(match.ball_xy, seen)
    right = home_attacks_right(match)
    gap_len, since, until = gap_info(seen)
    xy = match.players_xy
    team = match.track_team
    present = ~np.isnan(xy[..., 0])
    pspeed = speed(xy)
    dist = np.linalg.norm(xy - ball[:, None, :], axis=-1)  # NaN where player missing
    F = {}

    def att(x):  # x in the home-attacking frame, 0 = home goal, 1 = away goal
        return np.where(right, x / L, 1 - x / L)

    bx, by = ball[:, 0], ball[:, 1]
    F["seen"] = seen.astype(np.float32)
    F["gap_len"], F["since_seen"], F["until_seen"] = gap_len, since, until
    F["ball_x_att"] = att(bx)
    F["ball_y_c"] = np.abs(by - W / 2) / W
    F["ball_out"] = np.maximum.reduce([-bx, bx - L, -by, by - W, np.zeros_like(bx)])
    bs = speed(np.where(seen[:, None], match.ball_xy, np.nan))
    F["ball_speed"] = np.where(seen, bs, np.nan)
    for t, name in ((0, "h"), (1, "a")):
        d = np.where(team[None, :] == t, dist, np.nan)
        with np.errstate(all="ignore"):
            ds = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
            F["d1_" + name] = np.where(np.isfinite(ds[:, 0]), ds[:, 0], np.nan)
            F["d2_" + name] = np.where(np.isfinite(ds[:, 1]), ds[:, 1], np.nan)
            F["n3_" + name] = (d < 3).sum(1).astype(np.float32)
            F["n8_" + name] = (d < 8).sum(1).astype(np.float32)
            px = np.where(team[None, :] == t, xy[..., 0], np.nan)
            py = np.where(team[None, :] == t, xy[..., 1], np.nan)
            F["cx_" + name] = att(np.nanmean(px, 1))
            F["sx_" + name] = np.nanstd(px, 1) / L
            F["sy_" + name] = np.nanstd(py, 1) / W
            F["v_" + name] = np.nanmean(np.where(team[None, :] == t, pspeed, np.nan), 1)
            F["n_" + name] = (present & (team[None, :] == t)).sum(1).astype(np.float32)
    with np.errstate(all="ignore"):
        F["v_all"] = np.nanmean(pspeed, 1)
        F["slow_all"] = np.nanmean(np.where(np.isnan(pspeed), np.nan, pspeed < 1.0), 1)
        F["vmax"] = np.nanmax(np.where(np.isnan(pspeed), -1, pspeed), 1)
    F["vmax"][F["vmax"] < 0] = np.nan
    seen_d = np.where(seen, F["d1_h"] - F["d1_a"], np.nan)
    F["dd"] = np.clip(seen_d, -20, 20)
    near = np.where(seen & (np.minimum(F["d1_h"], F["d1_a"]) < 2.0),
                    np.where(F["d1_h"] < F["d1_a"], 1.0, -1.0), np.nan)
    F["near_ctrl"] = near
    for w in (25, 75, 225, 625):
        F["near_ctrl_m%d" % w] = rolling_mean(near, w)
        F["dd_m%d" % w] = rolling_mean(F["dd"], w)
    for w in (75, 225):
        F["v_all_m%d" % w] = rolling_mean(F["v_all"], w)
        F["seen_m%d" % w] = rolling_mean(F["seen"], w)
        F["out_m%d" % w] = rolling_mean((F["ball_out"] > 0.5).astype(float), w)
        F["cxdiff_m%d" % w] = rolling_mean(F["cx_h"] + F["cx_a"] - 1.0, w)
    # last confident controller before / next after this frame (seconds away, signed team)
    F["last_ctrl"], F["last_ctrl_age"] = _last_controller(near)
    nl, na = _last_controller(near[::-1])
    F["next_ctrl"], F["next_ctrl_age"] = nl[::-1], na[::-1]
    names = sorted(F)
    X = np.stack([np.asarray(F[k], np.float32) for k in names], axis=1)
    return X, names


def _last_controller(near):
    """Most recent non-NaN value of `near` (+1/-1) and how long ago (s)."""
    idx = np.where(~np.isnan(near), np.arange(len(near)), -1)
    idx = np.maximum.accumulate(idx)
    val = np.where(idx >= 0, near[np.maximum(idx, 0)], np.nan)
    age = np.where(idx >= 0, (np.arange(len(near)) - idx) / FPS, np.nan)
    return val.astype(np.float32), age.astype(np.float32)


def mirror(X, names):
    """Swap home and away: _h <-> _a, attack-frame x flipped, signed features negated."""
    idx = {n: i for i, n in enumerate(names)}
    Y = X.copy()
    for n, i in idx.items():
        if n.endswith("_h") or "_h_" in n:
            j = idx.get(n.replace("_h", "_a"))
            if j is not None:
                Y[:, i], Y[:, j] = X[:, j], X[:, i]
    for n, i in idx.items():
        if n.startswith(("dd", "near_ctrl", "last_ctrl", "next_ctrl")) and not n.endswith("_age"):
            Y[:, i] = -X[:, i]
        if n in ("ball_x_att",) or n.startswith("cx_"):
            Y[:, i] = 1 - Y[:, i]
        if n.startswith("cxdiff"):
            Y[:, i] = -X[:, i]
    return Y


# -- decoding -----------------------------------------------------------------------------------

def viterbi(logp, switch_team, switch_dead):
    """Most likely H/A/D path. logp (n, 3) per-frame log-probabilities for H, A, D.
    Costs (in log units) for H<->A and for alive<->dead transitions."""
    n = len(logp)
    T = np.array([[0, switch_team, switch_dead],
                  [switch_team, 0, switch_dead],
                  [switch_dead, switch_dead, 0]], float)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].astype(float).copy()
    lp = logp.astype(float)
    for t in range(1, n):
        cand = score[:, None] - T  # from i to j
        b = cand.argmax(0)
        back[t] = b
        score = cand[b, [0, 1, 2]] + lp[t]
    path = np.empty(n, np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def to_labels(path, graded):
    labels = np.array(["H", "A", "D"])[path].astype("<U1")
    labels[~graded] = "X"
    return labels


def decode_halves(match, logp, switch_team, switch_dead):
    """Viterbi separately per half; ungraded frames X."""
    labels = np.full(match.n_frames, "X", dtype="<U1")
    for _, first, last in match.halves:
        path = viterbi(logp[first:last + 1], switch_team, switch_dead)
        labels[first:last + 1] = np.array(["H", "A", "D"])[path]
    return labels
