"""Organiser baseline: nearest tracked player to the ball has it, then a majority vote.

Re-implemented on the shared arrays. With window=50 it must reproduce baseline_smooth50 (public
spellF1 0.3256); if it does not, the data pipeline is broken, not the idea.
"""
import numpy as np

PARAMS = {"window": 50}  # +/- frames in the centred majority vote; 0 = no smoothing


def nearest_team(match):
    """Per frame: team (0/1) of the player nearest the ball; -1 where ball or all players missing."""
    gap = match.players_xy - match.ball_xy[:, None, :]
    distance = np.sqrt((gap ** 2).sum(axis=2))
    distance = np.where(np.isnan(distance), np.inf, distance)
    team = match.track_team[distance.argmin(axis=1)].astype(np.int8)
    team[~np.isfinite(distance.min(axis=1))] = -1
    return team


def majority(team, window):
    """Centred majority vote over +/- window frames. D where nobody was seen; ties go home."""
    if window <= 0:
        return np.where(team == 0, "H", np.where(team == 1, "A", "D"))
    home = np.concatenate(([0], np.cumsum(team == 0)))
    away = np.concatenate(([0], np.cumsum(team == 1)))
    index = np.arange(len(team))
    low = np.maximum(0, index - window)
    high = np.minimum(len(team), index + window + 1)
    n_home, n_away = home[high] - home[low], away[high] - away[low]
    labels = np.full(len(team), "D", dtype="<U1")
    seen = (n_home + n_away) > 0
    labels[seen] = np.where(n_home[seen] >= n_away[seen], "H", "A")
    return labels


def predict(match, window=50):
    return majority(nearest_team(match), window)
