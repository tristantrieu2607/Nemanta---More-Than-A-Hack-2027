"""Make clean tracking look like the BAR-ATH automatic tracking (ATD).

What the ATD looks like (repo docs + aggregate figures other teams published from label-free
inspection; no BAR-ATH ground truth or video involved):
  * team-level only: 46 track slots (24 home, 22 away), no identities; slots appear, vanish and
    get re-used as the tracker loses and re-acquires people;
  * the ball is missing on ~41% of graded frames, in LONG blocks (median gap ~11 s, almost no
    missing frame sits in a gap < 5 s), and mostly while the ball is dead (seen ~27% of dead
    frames vs ~92% of live frames);
  * positions jitter; no ball height.

PROFILES are physical descriptions of a tracker, not tuned numbers. Every copy draws its own
values around the profile (domain randomisation, +/-`jitter`), so an algorithm is rewarded for
being robust to a family of trackers rather than to one setting. Score several seeds.
"""
import zlib

import numpy as np

from src.external.labels import runs

FPS = 25
SLOTS = (24, 22)  # home, away track slots, as in the ATD

PROFILES = {
    "clean": None,  # provider tracking as is (player identities still hidden behind slots)
    "atd": {
        "dead_seen": 0.27,           # share of each stoppage on which the ball is still tracked
        "live_missing": 0.08,        # share of live frames with no ball
        "live_gap_median_s": 6.0,    # live ball gaps: lognormal, clipped to [min, max]
        "live_gap_sigma": 0.7,
        "live_gap_min_s": 2.0,
        "live_gap_max_s": 40.0,
        "fast_ball_weight": 3.0,     # gaps start more often on fast (lofted / driven) balls
        "fast_ball_mps": 15.0,
        "close_gap_s": 1.0,          # provider gaps shorter than this are interpolated
        "min_seen_s": 0.5,           # ball sightings shorter than this inside gaps are dropped
        "ball_noise_m": 0.3,
        "player_noise_m": 0.15,
        "fragment_median_s": 45.0,   # a player's track is cut into fragments of this typical length
        "fragment_sigma": 1.0,
        "fragment_gap_prob": 0.6,    # chance a cut also loses the player for a moment
        "fragment_gap_median_s": 1.5,
        "player_drop": 0.03,         # extra share of player-frames lost in short occlusions
        "occlusion_median_s": 1.0,
        "gk_slot_prob": 0.9,         # GK fragments land on the team's flagged GK slot
        "jitter": 0.2,
    },
}
PROFILES["harsh"] = dict(PROFILES["atd"], dead_seen=0.15, live_missing=0.18, live_gap_median_s=9.0,
                         ball_noise_m=0.6, player_noise_m=0.3, fragment_median_s=20.0,
                         player_drop=0.08)


def rng_for(name, seed):
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def _cfg(profile, rng):
    base = PROFILES[profile]
    if base is None:
        return None
    j = base["jitter"]
    keep = {"jitter", "fast_ball_mps"}
    cfg = {k: (v if k in keep else v * rng.uniform(1 - j, 1 + j)) for k, v in base.items()}
    for k in ("dead_seen", "live_missing", "fragment_gap_prob", "player_drop", "gk_slot_prob"):
        cfg[k] = float(np.clip(cfg[k], 0.0, 1.0))
    return cfg


def _smooth(x, width=5):
    """Moving average along axis 0, NaN-aware (NaN stays NaN)."""
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    out = np.apply_along_axis(lambda v: np.convolve(np.nan_to_num(v), kernel, mode="same"), 0, x)
    return np.where(np.isnan(x), np.nan, out)


def _interpolate_short(ball, max_len):
    out = ball.copy()
    for s, e in runs(np.isnan(ball[:, 0])):
        if s > 0 and e < len(ball) and e - s <= max_len:
            w = (np.arange(s, e) - (s - 1)) / (e - s + 1)
            out[s:e] = ball[s - 1] * (1 - w[:, None]) + ball[e] * w[:, None]
    return out


def degrade_ball(ball, alive, graded, cfg, rng):
    ball = _interpolate_short(ball, int(cfg["close_gap_s"] * FPS))
    hide = np.zeros(len(ball), bool)
    # dead ball: one visible block per stoppage (usually the restart), the rest hidden. Providers
    # that already lose the ball while it is dead (Metrica, SkillCorner) get only what is missing.
    dead = ~alive & graded
    already = (~np.isnan(ball[:, 0]))[dead].mean() if dead.any() else 0.0
    keep = float(np.clip(cfg["dead_seen"] / max(already, 1e-9), 0.0, 1.0))
    for s, e in (runs(dead) if keep < 0.98 else []):
        n = e - s
        seen = int(round(n * rng.beta(2.0 * keep / (1 - keep + 1e-9) + 1e-3, 2.0)))
        seen = min(seen, n)
        where = rng.choice(["end", "start", "middle"], p=[0.6, 0.2, 0.2])
        lo = {"end": n - seen, "start": 0, "middle": int(rng.integers(0, n - seen + 1))}[where]
        hide[s:e] = True
        hide[s + lo:s + lo + seen] = False
    # live ball: long gaps until the live missing share is reached
    live = alive & graded
    speed = np.zeros(len(ball))
    speed[1:] = np.hypot(*np.diff(ball, axis=0).T) * FPS
    weight = np.where(live, 1.0 + cfg["fast_ball_weight"] * (np.nan_to_num(speed) > cfg["fast_ball_mps"]), 0.0)
    target = cfg["live_missing"] * live.sum()
    missing = lambda: ((hide | np.isnan(ball[:, 0])) & live).sum()
    tries = 0
    while missing() < target and weight.sum() > 0 and tries < 10000:
        tries += 1
        start = rng.choice(len(ball), p=weight / weight.sum())
        dur = np.exp(np.log(cfg["live_gap_median_s"]) + cfg["live_gap_sigma"] * rng.standard_normal())
        dur = int(np.clip(dur, cfg["live_gap_min_s"], cfg["live_gap_max_s"]) * FPS)
        hide[start:start + dur] = True
        weight[max(0, start - FPS):start + dur] = 0.0
    out = ball.copy()
    out[hide] = np.nan
    # drop tiny sightings stranded inside gaps (a tracker that interpolates would not report them)
    for s, e in runs(~np.isnan(out[:, 0])):
        if e - s < cfg["min_seen_s"] * FPS:
            out[s:e] = np.nan
    noise = _smooth(rng.normal(0, cfg["ball_noise_m"], out.shape))
    return (out + noise).astype(np.float32)


def _lognormal_frames(median_s, sigma, rng):
    return max(1, int(np.exp(np.log(median_s) + sigma * rng.standard_normal()) * FPS))


def degrade_players(xy, team, gk, cfg, rng, slots=SLOTS):
    """(n, p, 2) identified players -> (n, sum(slots), 2) anonymous fragmented slots."""
    n, p, _ = xy.shape
    xy = xy + _smooth(rng.normal(0, cfg["player_noise_m"], xy.shape)) if cfg else xy
    fragments = []  # (start, end, player)
    for j in range(p):
        present = ~np.isnan(xy[:, j, 0])
        if cfg:
            # occlusions
            n_drop = int(cfg["player_drop"] * present.sum() / max(1, cfg["occlusion_median_s"] * FPS))
            for start in rng.integers(0, n, size=n_drop):
                present[start:start + _lognormal_frames(cfg["occlusion_median_s"], 0.5, rng)] = False
        for s, e in runs(present):
            t = s
            while t < e:
                length = _lognormal_frames(cfg["fragment_median_s"], cfg["fragment_sigma"], rng) if cfg else e - t
                end = min(e, t + length)
                fragments.append((t, end, j))
                t = end
                if cfg and t < e and rng.random() < cfg["fragment_gap_prob"]:
                    t = min(e, t + _lognormal_frames(cfg["fragment_gap_median_s"], 0.5, rng))
    fragments.sort()
    offsets = (0, slots[0])
    track_team = np.array([0] * slots[0] + [1] * slots[1], np.int8)
    track_gk = np.zeros(sum(slots), bool)
    track_gk[[0, slots[0]]] = True
    busy_until = np.zeros(sum(slots), int)
    out = np.full((n, sum(slots), 2), np.nan, np.float32)
    for s, e, j in fragments:
        t = int(team[j])
        pool = np.arange(offsets[t], offsets[t] + slots[t])
        free = pool[busy_until[pool] <= s]
        if not len(free):
            continue  # more simultaneous fragments than slots: the tracker drops one
        gk_slot = offsets[t]
        if gk[j] and gk_slot in free and (not cfg or rng.random() < cfg["gk_slot_prob"]):
            slot = gk_slot
        else:
            outfield = free[free != gk_slot]
            slot = rng.choice(outfield) if len(outfield) else free[0]
        out[s:e, slot] = xy[s:e, j]
        busy_until[slot] = e
    return out, track_team, track_gk


def degrade(clean, profile="atd", seed=0):
    """-> dict(players_xy, track_team, track_is_gk, ball_xy) for one copy of `clean`."""
    rng = rng_for(clean.name, seed)
    cfg = _cfg(profile, rng)
    if cfg is None:
        ball = clean.ball_xy.copy()
    else:
        ball = degrade_ball(clean.ball_xy, clean.alive, clean.graded, cfg, rng)
    players, team, gk = degrade_players(clean.players_xy, clean.player_team, clean.player_gk, cfg, rng)
    return {"players_xy": players, "track_team": team, "track_is_gk": gk, "ball_xy": ball}
