"""DFL / IDSSE: 7 Bundesliga 2022/23 matches, 25 Hz optical tracking (Bassek et al. 2025, CC BY 4.0).

Per frame the DFL position feed carries BallStatus (alive/dead) and BallPossession (owning team):
the same kind of flags the challenge ground truth is derived from. Parsed with kloppy (pinned in
requirements-starter.txt). BallPossession flips on every touch; labels.py applies the control rule.

Files expected in data/external/idsse/ (python -m src.external.fetch --only idsse, or by hand from
https://doi.org/10.6084/m9.figshare.28196177): one matchinformation XML and one positions XML per
match, file names containing DFL-MAT-<id>.
"""
import glob
import os
import re

import numpy as np

from src.external import schema
from src.external.fetch import EXTERNAL

SOURCE = "idsse"
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
FOLDER = os.path.join(EXTERNAL, "idsse")


def _files(match_id):
    names = glob.glob(os.path.join(FOLDER, "**", "*{}*".format(match_id)), recursive=True)
    names = [n for n in names if n.lower().endswith(".xml") and os.path.getsize(n) > 0]
    meta = [n for n in names if re.search("matchinformation|meta", os.path.basename(n), re.I)]
    raw = [n for n in names if re.search("positions|tracking", os.path.basename(n), re.I)]
    if not meta or not raw:
        return None
    return meta[0], raw[0]


def available():
    return [m for m in MATCHES if _files(m)]


def convert(match_id):
    from kloppy import sportec

    meta, raw = _files(match_id)
    dataset = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False,
                                    coordinates="sportec")
    dims = dataset.metadata.pitch_dimensions
    length, width = dims.x_dim.max - dims.x_dim.min, dims.y_dim.max - dims.y_dim.min
    teams = dataset.metadata.teams  # [home, away]
    team_of, players, gk = {}, [], []
    for t, team in enumerate(teams):
        team_of[team.team_id] = t
        for player in team.players:
            players.append(player.player_id)
            gk.append("goalkeeper" in str(player.starting_position).lower())
    frame = dataset.to_df(engine="pandas")
    periods = []
    for _, part in frame.sort_values(["period_id", "frame_id"]).groupby("period_id", sort=True):
        n = len(part)
        xy = np.full((n, len(players), 2), np.nan, np.float32)
        for j, pid in enumerate(players):
            if pid + "_x" in part:
                xy[:, j, 0] = part[pid + "_x"].to_numpy(np.float32) - dims.x_dim.min
                xy[:, j, 1] = part[pid + "_y"].to_numpy(np.float32) - dims.y_dim.min
        ball = np.stack([part["ball_x"].to_numpy(np.float32) - dims.x_dim.min,
                         part["ball_y"].to_numpy(np.float32) - dims.y_dim.min], axis=1)
        owner = part["ball_owning_team_id"].map(lambda v: team_of.get(str(v), -1) if v is not None else -1)
        periods.append({"players_xy": xy, "ball_xy": ball,
                        "alive": (part["ball_state"].astype(str) == "alive").to_numpy(),
                        "owner": owner.fillna(-1).to_numpy(np.int8)})
    team = [team_of[t.team_id] for t in teams for _ in t.players]
    return schema.assemble(SOURCE, match_id, (length, width), periods, team, gk)
