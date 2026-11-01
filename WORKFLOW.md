# Team workflow

The loop: **idea → algorithm file → run → compare → keep or drop → next idea.**
Setup and the short version are in README.md. This is the detail.

## 1. Idea
Pick from the backlog in `EXPERIMENTS.md`, or start from a failure you saw while watching the video next to your labels. Check the log first so we don't redo dropped ideas.

## 2. Code it
`/idea <description>` in Claude Code does steps 2–6 for you. By hand:
- Copy `src/algos/_template.py` to `src/algos/<idea>.py`. Never edit someone else's algo file; copy it or import its helpers.
- Write `predict(match, **params)`: return one of `H`/`A`/`D`/`X` per frame. `match` has everything as numpy arrays (see `class Match` in `src/common.py`).
- Every tunable number goes in `PARAMS`, with a physical meaning (metres, seconds), not a value fitted to this match.
- Write the docstring for a reader. It becomes the technical document.

## 3. Run it
```powershell
python -m src.run <idea> --note "what this tests"
python -m src.run <idea> -p radius_m=1.0,1.5,2.0        # one run per value
python -m src.run <idea> --bootstrap 1000                # intervals (meaningful on validation)
python -m src.run <idea> --no-log                        # quick look, nothing recorded
```
Writes `data/submissions/<idea>__<params>.csv`, re-checks it with the organisers' validator, scores it, logs one JSON to `experiments/runs/`.

## 4. Compare
`python -m src.leaderboard`. Two kinds of number:
- **check_*** (public 5 min): did it run, is the possession count sane (truth: 17 possessions, mean 14.2 s). Way above 17 = fragmenting, below = merging. **Not for choosing between versions. That's a rule.**
- **val_*** (other games, once built): the actual ranking. Gaps inside the bootstrap interval are noise.

## 5. Decide
One line in `EXPERIMENTS.md`: kept or dropped, and what you learned. Changelog line in `BUILD_NOTES.md` if the pipeline changed.

## 6. Share
```powershell
python -m src.share <idea> -m "<idea>: what it does"
```
Commits only the algo file, run logs, EXPERIMENTS.md and BUILD_NOTES.md, pulls teammates' work, pushes. Anything else (shared code in `src/common.py`, docs) → normal git:
```powershell
git pull --rebase
git add <files>
git commit -m "..."
git push
```
- Work on `main`. Algos and run logs are separate files, so conflicts are rare.
- `*` next to a commit on the leaderboard = run from uncommitted code. Rerun after sharing if you want a clean record.
- Never commit `data/`, notebook outputs, or anything frame-level from the tracking.

## Rules to keep in mind
- No hand labels or human-made data about this match, for any purpose.
- No constants tied to specific frames you saw in the video.
- Data from other games is fine; record what and from where in the technical doc.
