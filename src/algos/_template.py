"""<One line: the idea.>

<The possession rule this encodes and why it should beat what came before. Write it for a reader:
this docstring goes straight into the technical document.>

Copy to src/algos/<idea>.py, then run:  python -m src.run <idea>
"""
import numpy as np

from src.algos.baseline import nearest_team  # reuse building blocks freely

# Every tunable number lives here, with a physical meaning. -p key=value overrides on the CLI.
PARAMS = {"radius_m": 1.5}


def predict(match, radius_m=1.5):
    """Return one label per video frame -- 'H', 'A', 'D' or 'X' -- length match.n_frames.

    Available: match.players_xy, players_speed, ball_xy, ball_speed, ball_seen, track_team,
    track_is_gk, graded, halves, tags, pitch. See src/common.py (class Match).
    """
    labels = match.blank("D")
    # ... your idea here ...
    return labels
