"""SkillCorner Open Data: 20 A-League 2024/25 matches, 10 fps automatic *broadcast* tracking (MIT).

Real automatic-tracking noise (ball detected on part of the frames, players out of camera view
extrapolated by SkillCorner). Resampled to 25 fps by linear interpolation. Labels from
`phases_of_play`: the ball is alive inside a phase, and the phase's team_in_possession has it.
Gaps between phases shorter than LIVE_GAP_S are live with the owner unknown (loose ball, kept by
labels.fill_owner); longer gaps are dead. Phase edges are less precise than a tracking flag, so
this source is a robustness check rather than the main tuning target.

Credit: data courtesy of SkillCorner, https://github.com/SkillCorner/opendata
"""
import json
import os

import numpy as np
import pandas as pd

from src.external import schema
from src.external.fetch import EXTERNAL

SOURCE = "skillcorner"
FOLDER = os.path.join(EXTERNAL, "skillcorner")
SRC_FPS, FPS = 10, schema.FPS
LIVE_GAP_S = 1.0


def available():
    if not os.path.isdir(FOLDER):
        return []
    return sorted(d for d in os.listdir(FOLDER)
                  if os.path.isfile(os.path.join(FOLDER, d, d + "_tracking_extrapolated.jsonl")))


def _resample(values, n_out):
    """(n_in, ...) at 10 fps -> (n_out, ...) at 25 fps, linear; NaN if either neighbour is NaN."""
    pos = np.arange(n_out) * SRC_FPS / FPS
    lo = np.minimum(np.floor(pos).astype(int), len(values) - 1)
    hi = np.minimum(lo + 1, len(values) - 1)
    w = (pos - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return (values[lo] * (1 - w) + values[hi] * w).astype(np.float32)


def convert(match_id):
    match_id = str(match_id)
    folder = os.path.join(FOLDER, match_id)
    with open(os.path.join(folder, match_id + "_match.json")) as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    players = sorted(meta["players"], key=lambda p: (p["team_id"] != home, p["id"]))
    col = {p["id"]: j for j, p in enumerate(players)}
    team = [0 if p["team_id"] == home else 1 for p in players]
    gk = [(p.get("player_role") or {}).get("acronym") == "GK" for p in players]
    team_of = {home: 0, meta["away_team"]["id"]: 1}

    with open(os.path.join(folder, match_id + "_tracking_extrapolated.jsonl")) as handle:
        frames = [json.loads(line) for line in handle]
    by_frame = {f["frame"]: f for f in frames}
    phases = pd.read_csv(os.path.join(folder, match_id + "_phases_of_play.csv"))

    periods = []
    for period in meta["match_periods"]:
        first, last = int(period["start_frame"]), int(period["end_frame"])
        n_in = last - first + 1
        xy = np.full((n_in, len(players), 2), np.nan, np.float32)
        ball = np.full((n_in, 2), np.nan, np.float32)
        for i, f in enumerate(range(first, last + 1)):
            rec = by_frame.get(f)
            if rec is None:
                continue
            b = rec.get("ball_data") or {}
            if b.get("is_detected") and b.get("x") is not None:
                ball[i] = (b["x"] + length / 2, b["y"] + width / 2)
            for p in rec.get("player_data") or []:
                j = col.get(p["player_id"])
                if j is not None and p.get("x") is not None:
                    xy[i, j] = (p["x"] + length / 2, p["y"] + width / 2)
        n_out = int((n_in - 1) * FPS / SRC_FPS) + 1
        # source-frame index of each output frame, for the labels
        src = first + np.round(np.arange(n_out) * SRC_FPS / FPS).astype(int)
        alive = np.zeros(n_out, bool)
        owner = np.full(n_out, -1, np.int8)
        ph = phases[phases["period"] == period["period"]].sort_values("frame_start")
        prev_end = None
        for row in ph.itertuples(index=False):
            s, e = int(row.frame_start), int(row.frame_end)
            if prev_end is not None and 0 < s - prev_end <= LIVE_GAP_S * SRC_FPS:
                alive[(src > prev_end) & (src < s)] = True  # short gap: still live, owner unknown
            inside = (src >= s) & (src <= e)
            alive[inside] = True
            owner[inside] = team_of.get(row.team_in_possession_id, -1)
            prev_end = e
        periods.append({"players_xy": _resample(xy, n_out), "ball_xy": _resample(ball, n_out),
                        "alive": alive, "owner": owner})
    return schema.assemble(SOURCE, match_id, (length, width), periods, team, gk)
