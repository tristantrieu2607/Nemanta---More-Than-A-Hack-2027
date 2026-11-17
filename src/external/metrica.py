"""Metrica Sports sample data: 3 anonymised games, 25 Hz, same provider as the competition.

Game 3 is in the same FIFA EPTS format as the BAR-ATH ATD; games 1-2 are Metrica's CSV format.
Real ball gaps (~30-50% of frames). There is no per-frame possession flag: labels come from the
event data (events.py-style rules below). The repository has no licence file -- ask the
organisers before relying on it in the technical document.

Event rules (Metrica event definitions):
  * PASS, CARRY, SHOT, SET PIECE, RECOVERY by a team -> that team has the ball from the event start.
  * BALL LOST never starts a possession (it only confirms the team that already had it).
  * CHALLENGE (duels, aerials, tackles) never changes possession on its own.
  * Dead from BALL OUT end, SHOT ...OUT/GOAL end, FAULT RECEIVED start, BALL LOST OFFSIDE/END HALF,
    until the next restart (SET PIECE or goal kick); before kick-off and after the last event of a
    period the ball is dead.
"""
import json
import os

import numpy as np
import pandas as pd

from src.external import schema
from src.external.fetch import EXTERNAL

SOURCE = "metrica"
MATCHES = ("1", "2", "3")
FOLDER = os.path.join(EXTERNAL, "metrica")
PITCH = (105.0, 68.0)
OWNER_TYPES = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY"}
END_AFTER_LAST_EVENT = 50  # frames of live play after a period's last event


def _game_dir(game):
    return os.path.join(FOLDER, "Sample_Game_{}".format(game))


def available():
    return [g for g in MATCHES if os.path.isdir(_game_dir(g)) and len(os.listdir(_game_dir(g))) >= 3]


# -- tracking --------------------------------------------------------------------------------

def _csv_team(path):
    """Metrica CSV (3 header rows) -> period, frame, (n, p, 2) normalised xy, ball (n, 2)."""
    names = pd.read_csv(path, skiprows=2, nrows=0).columns.tolist()
    table = pd.read_csv(path, skiprows=3, header=None).to_numpy(np.float64)
    player_cols = [i for i, n in enumerate(names) if str(n).startswith("Player")]
    xy = np.stack([table[:, [i, i + 1]] for i in player_cols], axis=1)
    ball_col = names.index("Ball")
    return table[:, 0].astype(int), table[:, 1].astype(int), xy, table[:, [ball_col, ball_col + 1]]


def _tracking_csv(game):
    d = _game_dir(game)
    period, frame, home, ball = _csv_team(os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game)))
    _, frame_a, away, _ = _csv_team(os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(game)))
    if not np.array_equal(frame, frame_a):
        raise ValueError("game {}: home and away tracking frames differ".format(game))
    xy = np.concatenate([home, away], axis=1)
    team = [0] * home.shape[1] + [1] * away.shape[1]
    gk = [j in (0, home.shape[1]) for j in range(len(team))]  # first column of each file is the GK
    return period, frame, xy, ball, team, gk


def _tracking_epts(game):
    from kloppy import metrica

    d = _game_dir(game)
    ds = metrica.load_tracking_epts(meta_data=os.path.join(d, "Sample_Game_3_metadata.xml"),
                                    raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"),
                                    coordinates="metrica")
    frame = ds.to_df(engine="pandas").sort_values("frame_id")
    ids, team, gk = [], [], []
    for t, side in enumerate(ds.metadata.teams):
        for p in side.players:
            ids.append(p.player_id)
            team.append(t)
            gk.append("goalkeeper" in str(p.starting_position).lower())
    xy = np.full((len(frame), len(ids), 2), np.nan)
    for j, pid in enumerate(ids):
        if pid + "_x" in frame:
            xy[:, j, 0], xy[:, j, 1] = frame[pid + "_x"].to_numpy(float), frame[pid + "_y"].to_numpy(float)
    ball = frame[["ball_x", "ball_y"]].to_numpy(float)
    return (frame["period_id"].to_numpy(int), frame["frame_id"].to_numpy(int), xy, ball, team, gk,
            [t.team_id for t in ds.metadata.teams])


# -- events ----------------------------------------------------------------------------------

def _events(game, team_ids=None):
    """DataFrame: team (0/1/-1), type, subtype, period, start, end (tracking frame numbers)."""
    d = _game_dir(game)
    if game == "3":
        with open(os.path.join(d, "Sample_Game_3_events.json")) as handle:
            data = json.load(handle)["data"]
        rows = [{"team": team_ids.index(e["team"]["id"]) if e.get("team") and e["team"]["id"] in team_ids else -1,
                 "type": e["type"]["name"].upper(),
                 "subtype": (e.get("subtypes") or {}).get("name", "") if isinstance(e.get("subtypes"), dict)
                 else " ".join(s["name"] for s in (e.get("subtypes") or [])),
                 "period": e["period"], "start": e["start"]["frame"], "end": e["end"]["frame"]} for e in data]
        table = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(game)))
        table = pd.DataFrame({"team": raw["Team"].map({"Home": 0, "Away": 1}).fillna(-1).astype(int),
                              "type": raw["Type"].str.upper(), "subtype": raw["Subtype"].fillna("").str.upper(),
                              "period": raw["Period"], "start": raw["Start Frame"], "end": raw["End Frame"]})
    table["subtype"] = table["subtype"].fillna("").astype(str).str.upper()
    table["end"] = np.maximum(table["end"].fillna(table["start"]), table["start"])
    return table.sort_values(["period", "start"], kind="stable").reset_index(drop=True)


def _is_dead_start(row):
    t, s = row.type, row.subtype
    if t == "BALL OUT":
        return int(row.end)
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return int(row.end)
    if t == "FAULT RECEIVED":
        return int(row.start)
    if t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s):
        return int(row.end)
    return None


def _is_restart(row):
    return row.type == "SET PIECE" or (row.type == "PASS" and "GOAL KICK" in row.subtype)


def event_labels(events, frames):
    """alive (n,), owner (n,) for the tracking frames `frames` (one period)."""
    n = len(frames)
    first = frames[0]
    alive = np.zeros(n, bool)
    owner = np.full(n, -1, np.int8)
    rows = list(events.itertuples(index=False))
    if not rows:
        return alive, owner
    idx = lambda f: int(np.clip(f - first, 0, n))
    alive[idx(rows[0].start):idx(rows[-1].end + END_AFTER_LAST_EVENT)] = True
    current = -1
    for k, row in enumerate(rows):
        if row.team in (0, 1) and row.type in OWNER_TYPES:
            current = row.team
            owner[idx(row.start)] = current
        elif row.type == "BALL LOST" and row.team in (0, 1) and current == -1:
            current = row.team
            owner[idx(row.start)] = current
        dead_from = _is_dead_start(row)
        if dead_from is not None:
            nxt = next((r for r in rows[k + 1:] if _is_restart(r) and r.start >= dead_from), None)
            if nxt is None:
                nxt = next((r for r in rows[k + 1:] if r.type in OWNER_TYPES and r.start > dead_from), None)
            alive[idx(dead_from):idx(nxt.start) if nxt is not None else n] = False
    return alive, owner


def convert(game):
    game = str(game)
    if game == "3":
        period, frame, xy, ball, team, gk, team_ids = _tracking_epts(game)
    else:
        (period, frame, xy, ball, team, gk), team_ids = _tracking_csv(game), None
    events = _events(game, team_ids)
    scale = np.array(PITCH)
    periods = []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        alive, owner = event_labels(events[events["period"] == half], frame[rows])
        periods.append({"players_xy": xy[rows] * scale, "ball_xy": ball[rows] * scale,
                        "alive": alive, "owner": owner})
    return schema.assemble(SOURCE, game, PITCH, periods, team, gk)
