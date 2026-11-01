# Experiments

Scores live in `experiments/runs/` and `python -m src.leaderboard`. This file is the human side: what we tried, what we learned, what's next.

Decide on validation spellF1 + bootstrap, never on the public 5 minutes (17 possessions, sanity check only).

## Log
| # | Date | Who | Idea | Algo file | Best result | Verdict / what we learned |
|---|------|-----|------|-----------|-------------|---------------------------|
| 0 | 2026-11-01 | Tristan | Organiser baseline: nearest player + majority vote | `baseline.py` | public 0.3256 (window 50), full match 0.2433 | Floor to beat. window 125 wins public but loses full match — don't tune on public. |

## Backlog
Ranked. Move to the log once tried.

1. Control radius: only switch team when the nearest player is within ~1.5 m of the ball; otherwise keep the last team (loose-ball rule).
2. Hysteresis: require the new team to hold control for N frames before switching; backdate the switch to the first frame of control.
3. Ball gaps: interpolate short gaps; on long gaps keep the current team (or X) instead of guessing.
4. Dead-ball detection: ball outside the pitch, ball and players near-stationary, Smart Tagging set-piece intervals as restarts.
5. Sequence model: per-frame features → classifier → HMM/Viterbi with a switching penalty.
6. CV: detect the ball in the video on frames where the ATD has none.

## Validation set
Run `python -m src.check_labels` for the live state.
| File | Stretch | Labelled by | Possessions |
|------|---------|-------------|-------------|
