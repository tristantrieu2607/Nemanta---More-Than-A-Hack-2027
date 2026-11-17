"""One clean, provider-independent representation of an external match, cached as .npz.

Every source module (idsse, skillcorner, metrica) converts its files into a `Clean` with:

    players_xy   (n, p, 2) float32 metres, origin at a pitch corner (like Match), NaN = not on pitch
    player_team  (p,) int8   0 home, 1 away
    player_gk    (p,) bool
    ball_xy      (n, 2) float32 metres, NaN where the provider has no ball
    alive        (n,) bool   provider's ball-in-play flag (or derived from events / phases)
    owner        (n,) int8   provider's owning team 0/1, -1 unknown (raw, before the control rule)
    graded       (n,) bool   inside a period
    halves       [(half, first, last)] inclusive frame indices, 25 fps

Periods are laid end to end with BREAK frames of nothing in between (not graded), so smoothing
windows in an algorithm cannot leak across half-time.
"""
import os

import numpy as np

from src.common import ROOT

FPS = 25
BREAK = 250  # frames of empty timeline between periods
CACHE = os.path.join(ROOT, "data", "external", "cache")
FIELDS = ("players_xy", "player_team", "player_gk", "ball_xy", "alive", "owner", "graded")


class Clean:
    def __init__(self, source, match_id, pitch, halves, **arrays):
        self.source, self.match_id = source, str(match_id)
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.halves = [tuple(int(v) for v in h) for h in halves]
        for key in FIELDS:
            setattr(self, key, arrays[key])
        self.n_frames = len(self.alive)

    @property
    def name(self):
        return "{}:{}".format(self.source, self.match_id)

    def path(self):
        return cache_path(self.source, self.match_id)

    def save(self):
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(self.path(), source=self.source, match_id=self.match_id,
                            pitch=np.array(self.pitch), halves=np.array(self.halves),
                            **{k: getattr(self, k) for k in FIELDS})
        return self.path()


def cache_path(source, match_id):
    return os.path.join(CACHE, "{}_{}.npz".format(source, match_id))


def load(source, match_id):
    with np.load(cache_path(source, match_id), allow_pickle=False) as data:
        return Clean(str(data["source"]), str(data["match_id"]), data["pitch"],
                     data["halves"].tolist(), **{k: data[k] for k in FIELDS})


def cached(source):
    """Match ids of `source` already converted."""
    if not os.path.isdir(CACHE):
        return []
    prefix = source + "_"
    return sorted(f[len(prefix):-4] for f in os.listdir(CACHE)
                  if f.startswith(prefix) and f.endswith(".npz"))


def assemble(source, match_id, pitch, periods, player_team, player_gk):
    """Concatenate per-period dicts (players_xy, ball_xy, alive, owner) into one Clean."""
    parts, halves, start = {k: [] for k in ("players_xy", "ball_xy", "alive", "owner", "graded")}, [], 0
    p = len(player_team)
    for half, period in enumerate(periods, start=1):
        n = len(period["alive"])
        if half > 1:
            parts["players_xy"].append(np.full((BREAK, p, 2), np.nan, np.float32))
            parts["ball_xy"].append(np.full((BREAK, 2), np.nan, np.float32))
            parts["alive"].append(np.zeros(BREAK, bool))
            parts["owner"].append(np.full(BREAK, -1, np.int8))
            parts["graded"].append(np.zeros(BREAK, bool))
            start += BREAK
        parts["players_xy"].append(period["players_xy"].astype(np.float32))
        parts["ball_xy"].append(period["ball_xy"].astype(np.float32))
        parts["alive"].append(period["alive"].astype(bool))
        parts["owner"].append(period["owner"].astype(np.int8))
        parts["graded"].append(np.ones(n, bool))
        halves.append((half, start, start + n - 1))
        start += n
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    return Clean(source, match_id, pitch, halves, player_team=np.asarray(player_team, np.int8),
                 player_gk=np.asarray(player_gk, bool), **arrays)
