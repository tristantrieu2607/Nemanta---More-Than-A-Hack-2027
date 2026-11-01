# Build notes

Living description of the pipeline. Updated on every code change; becomes the technical doc.

## Architecture
- **Data layer** (`src/common.py`): parses the tactical ATD text (`frame_id:x,y,s;...:bx,by,bs`, 46 track slots + ball) into numpy arrays indexed by video frame, converts normalised coordinates to metres (105.3 × 68.0 m), reads team per slot (first team in the XML = home) and goalkeeper flags. Cached to `data/cache/BAR-ATH_atd.npz` after the first parse. Smart Tagging loaded lazily via the organiser loader.
- **Algorithms** (`src/algos/`): pure functions `predict(match, **params) -> labels`. No file I/O.
- **Runner** (`src/run.py`): writes the submission CSV, re-validates it with the organisers' `validate_submission`, scores it with the organisers' `possession_metrics.evaluate` against the public 5 min (pipeline check) and the validation set once it exists, logs a JSON record (params, commit, scores).
- **Leaderboard** (`src/leaderboard.py`): latest run per algo+params. Ranks on validation spellF1 only; without a validation set it lists runs by time with the public check alongside. Runs scored on an older validation set are hidden.
- **Sharing** (`src/share.py`, `.claude/commands/idea.md`): one command commits an algo + its run logs and pushes; `/idea` in Claude Code runs the whole loop for one idea.
- **Validation** (`validation/`): not built. Must come from other games (RULES.md).

## Current best
`baseline` (window 50): public spellF1 0.3256, full match 0.2433 (organiser figure).

## Changelog
- 2026-11-01: repo set up; CLAUDE.md, EXPERIMENTS.md, WORKFLOW.md added. Tracking metadata holds only x/y/speed per track + ball — no ball-owner or ball-alive fields.
- 2026-11-01: experiment framework (common, run, leaderboard, check_labels, algos/baseline, algos/_template). Baseline reproduces all four organiser public scores exactly (0.0378 / 0.2868 / 0.3256 / 0.4242), confirming frame alignment, team mapping and metre scaling.
- 2026-11-01: organisers' Oct 22–23 rule update: no hand labels on BAR-ATH for any purpose (incl. validation), public 5 min not for tuning or choosing versions. Removed hand-label validation and calibration tooling; leaderboard no longer ranks on the public check. Added `src/share.py`, `/idea` Claude Code command, team section at the top of README.md.
- 2026-11-01: to-do list added to the README team section.
