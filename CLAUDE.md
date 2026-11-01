# More Than A Hack 2027 — Phase 1 (Ball Possession)

Team of 3: Tristan (ML lead, owns repo + eval), CV teammate (ball recovery, dead-ball detection), business teammate (admin, technical doc, sourcing validation data from other games).

## The task
One match, FC Barcelona (H) vs Athletic Club (A), LaLiga 2024-25 MD2. For every frame of the tactical video output one label:
- `H` home in possession, ball alive · `A` away in possession, ball alive · `D` dead ball · `X` abstain
- 199,319 rows, 25 fps, `frame,label`, frame 0…199318 in order. Validate every file: `python evaluation/validate_submission.py <file>.csv`
- Graded frames: 1st half 11185–90651, 2nd half 110808–190197 (158,857 frames). Everything else is ungraded, write `X`.

## Scoring — read before optimising anything
- **spellF1 decides the ranking.** A possession = maximal run of H or of A. Predicted vs true possessions matched one-to-one by temporal IoU, F1 averaged over thresholds 0.3–0.7. Fragmenting a possession (a short flicker) is punished hard. 1 frame late is free, 1 s late ≈ 0.65.
- macroF1 = per-frame F1 averaged over H/A/D. Only the tie-break.
- `X` on graded frames is forward-filled with the previous label before spellF1. Free inside a possession, costly across a transition.
- Baseline to beat: `baseline_smooth50`, full-match spellF1 **0.2433**. Noise band on full match ≈ 0.04.

## Validation and the public 5 minutes — rules, not advice (RULES.md, updated Oct 22–23)
- Public ground truth = first 5 min (frames 11185–18684, 17 possessions). **Pipeline check only**: frames aligned, file well-formed, possession count sane. Not training data, not a tuning set, never used to choose between versions. (It also picks the wrong answer: smooth125 wins public, loses full match.)
- **No hand labels on BAR-ATH, in any amount, for any purpose** (not submission, not training, not validation).
- **No human-produced data about BAR-ATH from any source** (event data, corrected tracking, annotations).
- **No rules or constants tied to specific frames of this game** from watching the video. Watching to understand/debug is fine.
- Validation must come from **other games** (any amount, hand-labelled or not, documented in the tech doc). Not built yet — see validation/README.md. Until then there is no legal ranking; reason from first principles and football knowledge.
- Prefer parameters with a physical meaning (e.g. control radius ~1.5 m) over the best-scoring decimal.
- Unsupervised / self-supervised methods and algorithmic pseudo-labels on BAR-ATH are allowed (no person decides any label).

## Possession rules (CHALLENGE.md)
- Ball in play → exactly one team has possession, no neutral state. Loose ball / 50-50 → team that last had it keeps it.
- Possession changes only when a player of the other team has the ball **under control**. Not a change: single touch without control, deflected-but-not-intercepted pass, won aerial duel without control.
- Every restart (kick-off, throw-in, corner, goal kick, free kick) has a clear owner.
- GT is derived mechanically from corrected tracking (ball dead → D, else tracked owning team). The real problem is detecting *when* possession changes.

## Data (data/tactical/, gitignored — never commit, never publish derived frame-level exports)
- **ATD**: `BAR-ATH_tactical_atd.xml` (metadata) + `.txt` (196 MB). Load with kloppy:
  `metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"]).to_df()`
  - Per frame: x, y, speed for 46 track slots (24 Barcelona, 22 Athletic, 2 flagged GK) + ball. Coords normalised 0–1; pitch 105.3 × 68.0 m (use games.py values).
  - No player identities, slots swap/fragment. Ball missing on 41% of graded frames — the most important fact in the dataset.
  - `frame_id` is 1-based: `video_frame = frame_id - 1`.
  - XML says first half spans the whole file and second half is blank — ignore, use `games.half_video_blocks`.
  - XML labels speed as km/h; DATA.md says m/s after kloppy. Check before using speed thresholds.
  - kloppy adds `ball_state` / `ball_owning_team_id` columns — check whether populated (expected empty).
- **Smart Tagging**: `BAR-ATH_tactical_smart_tagging.xml`, 638 auto-inferred intervals (Build Up, Throw-in, Corner…). `video_frame = round(seconds * 25)`. Evidence, not answers. Throw-ins have Team `N/A`. Match codes/tags case-insensitively.
- **Homography**: per-frame 3×3, pitch (0–1) ↔ pixels. Tactical file 100% valid. Only needed for video work.
- **Video**: 5.2 GB, not needed for a submission. Broadcast feed is unaligned — ignore unless there's spare time.

## Rules
- Labels must come from an algorithm. No hand-editing, frame by frame, by segment or at transitions.
- Improving/replacing the tracking is fine only if done by code, never by hand.
- External data and labels must be about other games, accessible to the organisers, and explained in the technical doc (what, who labelled it, how much).
- Repo must regenerate the submission file exactly. Pin dependencies, fix random seeds.

## Deliverables (deadline Nov 17 23:59 CET = Tue Nov 18 09:59 Sydney; internal freeze Nov 16)
1. `<team_name>.csv` 2. public code repo URL 3. short technical doc `<team_name>.pdf`/`.md` (approach, assets used/dropped, possession rules chosen, limitations, reproduction). One Google Form per team, instructions arrive by Nov 14. All 3 members must register with the same team name before Nov 10.

## Repo layout and the experiment loop (see WORKFLOW.md)
- `src/common.py` — `Match`: all data as numpy arrays indexed by video frame (players_xy in metres, ball_xy, ball_seen, track_team, graded, halves, tags...). Parses the ATD once and caches to `data/cache/`.
- `src/algos/<idea>.py` — one file per algorithm: `PARAMS` dict of defaults + `predict(match, **params)` returning one of H/A/D/X per frame. Start from `_template.py`. `baseline.py` reproduces the organiser baseline exactly (public spellF1 0.3256 at window=50) — use it to check the pipeline.
- `python -m src.run <algo> [-p k=v1,v2] [--note ...] [--bootstrap 1000]` — writes `data/submissions/`, validates with the organisers' check, scores the public check (+ validation once it exists), logs JSON to `experiments/runs/`.
- `python -m src.leaderboard` — lists runs; ranks only once a validation set from other games exists.
- `python -m src.share <algo> -m "msg"` — commits the algo file + its run logs + EXPERIMENTS/BUILD_NOTES, pulls with rebase, pushes.
- `/idea <description>` (Claude Code command in `.claude/commands/idea.md`) — the whole loop for one idea.
- Organiser files (games.py, evaluation/, starter/) stay unmodified so scoring matches the official scorer.
- Never put code in `data/`. Never commit data, notebook outputs or frame-level tracking exports.

## How to help
- New idea → new file in `src/algos/`, never edit another algo in place.
- Read EXPERIMENTS.md (verdicts + backlog) and `python -m src.leaderboard` before suggesting ideas; don't re-suggest anything logged as dropped.
- Start from a concrete failure (where predicted possessions differ from truth), not a generic list.
- After any code change, update BUILD_NOTES.md so it accumulates into the architecture description.
