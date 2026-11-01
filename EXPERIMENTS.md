# Experiments

Scores live in `experiments/runs/` and `python -m src.leaderboard`. This file is the human side: what we tried, what we learned, what's next.

Decide on validation spellF1 + bootstrap once validation (other games) exists. The public 5 minutes are a pipeline check only. Choosing versions on them is against RULES.md.

## Log
| # | Date | Who | Idea | Algo file | Best result | Verdict / what we learned |
|---|------|-----|------|-----------|-------------|---------------------------|
| 0 | 2026-11-01 | Tristan | Organiser baseline: nearest player + majority vote | `baseline.py` | public 0.3256 (window 50), full match 0.2433 | Floor to beat. window 125 wins public but loses full match — don't tune on public. |

## Backlog
Ranked. Move to the log once tried.

0. **Validation set from other games** (blocks ranking): find openly published tracking + possession/event data, convert to per-frame H/A/D, degrade to look like ours (drop the ball ~41% of frames, break tracks), score every algo on it.
1. Control radius: only switch team when the nearest player is within ~1.5 m of the ball; otherwise keep the last team (loose-ball rule).
2. Hysteresis: require the new team to hold control for N frames before switching; backdate the switch to the first frame of control.
3. Ball gaps: interpolate short gaps; on long gaps keep the current team (or X) instead of guessing.
4. Dead-ball detection: ball outside the pitch, ball and players near-stationary, Smart Tagging set-piece intervals as restarts.
5. Sequence model: per-frame features → classifier → HMM/Viterbi with a switching penalty.
6. CV: detect the ball in the video on frames where the ATD has none.
