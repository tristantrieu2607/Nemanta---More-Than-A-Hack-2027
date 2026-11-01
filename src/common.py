"""Shared plumbing: load the match once, hand algorithms numpy arrays, write and score labels.

Algorithms never touch files. They receive a Match and return one label per video frame.
"""
import glob
import hashlib
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import possession_metrics as pm  # noqa: E402  the organisers' scorer, unmodified
import load_smart_tagging  # noqa: E402
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

HOME, AWAY = 0, 1
FPS = 25
X_CODE = pm.LABELS.index("X")
CACHE_DIR = os.path.join(ROOT, "data", "cache")            # under data/, so never committed
SUBMISSION_DIR = os.path.join(ROOT, "data", "submissions")  # likewise
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
LABELS_DIR = os.path.join(ROOT, "validation", "labels")
CALIBRATION_DIR = os.path.join(ROOT, "validation", "calibration")
FIELDS = 3  # x, y, speed per track and for the ball


# -- loading -----------------------------------------------------------------------------------

def _track_meta(meta_path):
    """Per track slot: team (0 home, 1 away) and goalkeeper flag. First team listed is home."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    team, goalkeeper = [], []
    for player in root.iter("Player"):
        team.append(HOME if player.get("teamId") == home_id else AWAY)
        params = {p.findtext("Name"): p.findtext("Value")
                  for p in player.iter("ProviderParameter")}
        goalkeeper.append(params.get("position_type") == "Goalkeeper")
    return np.array(team, dtype=np.int8), np.array(goalkeeper, dtype=bool)


def _parse_atd(raw_path, n_frames, n_tracks):
    """'frame_id:x,y,s;x,y,s;...:bx,by,bs' lines -> arrays indexed by video frame (frame_id - 1)."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).to_numpy()
    expected = 1 + (n_tracks + 1) * FIELDS
    if table.shape[1] != expected:
        raise ValueError("ATD has {} fields per line, expected {}".format(table.shape[1], expected))
    frames = table[:, 0].astype(np.int64) - 1
    players = np.full((n_frames, n_tracks, FIELDS), np.nan, dtype=np.float32)
    ball = np.full((n_frames, FIELDS), np.nan, dtype=np.float32)
    players[frames] = table[:, 1:1 + n_tracks * FIELDS].reshape(-1, n_tracks, FIELDS)
    ball[frames] = table[:, -FIELDS:]
    return players, ball


def _load_arrays(game):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    cache = os.path.join(CACHE_DIR, "{}_atd.npz".format(game))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    print("first run: parsing {} once, cached to {}".format(
        os.path.basename(raw_path), os.path.relpath(cache, ROOT)))
    team, goalkeeper = _track_meta(meta_path)
    players, ball = _parse_atd(raw_path, source_frames("tactical", game), len(team))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, players=players, ball=ball, team=team, gk=goalkeeper)
    return {"players": players, "ball": ball, "team": team, "gk": goalkeeper}


class Match:
    """Everything an algorithm may read. Arrays are indexed by video frame = submission row.

    players_xy     (n_frames, n_tracks, 2) metres, NaN where the slot is empty
    players_speed  (n_frames, n_tracks)    as delivered -- unit unverified, check before thresholding
    ball_xy        (n_frames, 2) metres, NaN where the ball is not tracked (~41% of graded frames)
    ball_speed     (n_frames,)
    ball_seen      (n_frames,) bool
    track_team     (n_tracks,) 0 = home = H = Barcelona, 1 = away = A = Athletic
    track_is_gk    (n_tracks,) bool
    graded         (n_frames,) bool, inside one of the two halves
    halves         [(half, first_frame, last_frame)], inclusive
    tags           Smart Tagging intervals, list of dicts (start_frame, end_frame, code, tags)
    pitch          (length_m, width_m)
    """

    def __init__(self, game=GAME):
        arrays = _load_arrays(game)
        length, width = GAMES[game]["pitch"]
        scale = np.array([length, width], dtype=np.float32)
        self.game = game
        self.pitch = (length, width)
        self.n_frames = source_frames("tactical", game)
        self.players_xy = arrays["players"][:, :, :2] * scale
        self.players_speed = arrays["players"][:, :, 2]
        self.ball_xy = arrays["ball"][:, :2] * scale
        self.ball_speed = arrays["ball"][:, 2]
        self.ball_seen = ~np.isnan(self.ball_xy[:, 0])
        self.track_team = arrays["team"]
        self.track_is_gk = arrays["gk"]
        self.halves = list(half_video_blocks(game))
        self.graded = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.halves:
            self.graded[first:last + 1] = True
        self._tags = None

    @property
    def tags(self):
        if self._tags is None:
            self._tags = load_smart_tagging.load(self.game)
        return self._tags

    def blank(self, fill="X"):
        """A label array to fill in: one entry per frame."""
        return np.full(self.n_frames, fill, dtype="<U1")


# -- submissions and scoring -------------------------------------------------------------------

def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def check_file(path, n_frames):
    """The organisers' exact validation. Returns label codes, raises with their reason."""
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_frames)
    if reason:
        raise ValueError("{} is not a valid submission: {}".format(path, reason))
    return pm.label_codes(table)


def public_gt(game=GAME):
    return pm.label_codes(pm.read_label_csv(PUBLIC_GT.format(game)))


SUMMARY = ("spell_f1_mean", "macro_f1", "spell_bootstrap_low", "spell_bootstrap_high",
           "n_pred_spells", "n_gt_spells", "mean_spell_duration_s", "gt_mean_spell_duration_s",
           "alive_dead_f1", "team_acc_gt_alive", "n_graded")


def _plain(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return round(float(value), 4)


def score(pred_codes, gt_codes, bootstrap_n=0):
    result = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    return {key: _plain(result[key]) for key in SUMMARY}


# -- our hand labels ---------------------------------------------------------------------------

def _seconds(text):
    total = 0.0
    for part in str(text).strip().split(":"):
        total = total * 60 + float(part)
    return total


def read_label_points(path, n_frames):
    """A hand-label file -> full-length codes, X outside the labelled stretch.

    Each row means "from here on it is <label>". Columns: time,label (seconds, mm:ss.ss or
    h:mm:ss.ss of the TACTICAL video) or frame,label. The last row must be X, closing the stretch.
    """
    table = pd.read_csv(path, dtype=str, comment="#", skip_blank_lines=True)
    table.columns = [c.strip().lower() for c in table.columns]
    if "frame" in table.columns:
        starts = table["frame"].astype(int).to_numpy()
    elif "time" in table.columns:
        starts = np.array([int(round(_seconds(t) * FPS)) for t in table["time"]])
    else:
        raise ValueError("{}: needs a 'time' or 'frame' column".format(path))
    labels = table["label"].str.strip().str.upper().to_numpy()
    bad = sorted(set(labels) - set(pm.LABELS))
    if bad:
        raise ValueError("{}: labels outside H/A/D/X: {}".format(path, bad))
    if labels[-1] != "X":
        raise ValueError("{}: the last row must be X, closing the labelled stretch".format(path))
    if np.any(np.diff(starts) <= 0):
        row = int(np.argmax(np.diff(starts) <= 0)) + 2
        raise ValueError("{}: times must strictly increase (row {})".format(path, row))
    codes = np.full(n_frames, X_CODE, dtype=np.int8)
    for start, end, label in zip(starts[:-1], starts[1:], labels[:-1]):
        codes[start:end] = pm.LABELS.index(label)
    return codes


def load_validation(n_frames, folder=LABELS_DIR):
    """All hand-labelled stretches merged into one ground truth. (codes or None, files, hash)."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        return None, [], None
    merged = np.full(n_frames, X_CODE, dtype=np.int8)
    for path in files:
        codes = read_label_points(path, n_frames)
        labelled = codes != X_CODE
        if (labelled & (merged != X_CODE)).any():
            raise ValueError("{} overlaps another label file".format(os.path.basename(path)))
        merged[labelled] = codes[labelled]
    return merged, files, hashlib.md5(merged.tobytes()).hexdigest()[:8]
