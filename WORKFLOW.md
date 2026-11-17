# Team workflow

The loop: **idea → algorithm file → run → leaderboard → keep or drop → next idea.**

## One-time setup (each teammate)
```powershell
git clone https://github.com/tristantrieu2607/Nemanta---More-Than-A-Hack-2027.git
cd Nemanta---More-Than-A-Hack-2027
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-starter.txt
```
Download these 4 files from the shared Drive (`...Dataset/tactical/`) into `data\tactical\`:
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_homography.csv`, `BAR-ATH_tactical_smart_tagging.xml`.
`data\` is gitignored — data never goes to GitHub.

Build the external benchmark (once, ~3 GB download; see `validation/README.md`):
```powershell
python -m src.external.fetch
python -m src.external.bench convert
```
Check it works (first run parses the tracking once, ~1 min, then it's cached):
```powershell
python -m src.run baseline --no-log
```
Expected: external `main` balanced spellF1 ~0.35 at window=50 (higher than the real match: the
benchmark is easier, compare versions not levels), public check 0.3256.

## The loop
1. **Idea** — pick from the backlog in `EXPERIMENTS.md`, or start from a failure you saw.
2. **Code it** — copy `src/algos/_template.py` to `src/algos/<idea>.py`. Put every tunable number in `PARAMS`. Write the docstring for a reader: it becomes the technical doc. Never edit someone else's algo file — copy it to a new name.
3. **Run it**
   ```powershell
   python -m src.run <idea> --note "what this tests"
   python -m src.run <idea> -p radius_m=1.0,1.5,2.0          # small sweep
   python -m src.run <idea> --external-only                   # no BAR-ATH data needed
   python -m src.run <idea> --profile harsh                   # stress test (not ranked)
   ```
   Scores the external benchmark (every DFL / SkillCorner / Metrica match x 2 degradation seeds), writes `data/submissions/<idea>__<params>.csv` (validated with the organisers' own check), prints the public check, logs one JSON to `experiments/runs/`.
4. **Rank** — `python -m src.leaderboard`
5. **Decide** — add one line to `EXPERIMENTS.md`: kept or dropped, and what you learned. Update `BUILD_NOTES.md` if the pipeline changed.
6. **Share** — commit and push (below).

## The no-code loop
Write an idea in words in `strategies/` (see `strategies/README.md`) and tell Claude "run the new strategies". Claude codes it, scores it with `python -m src.strategy <algo> --quick --title "..."` (10 matches, minutes) and writes the verdict into the idea file and `strategies/BOARD.md`. Winners get a full run (same command without `--quick`).

## Ranking rules
- Ranked on **external balanced spellF1** (`main` reading). Never on the public 5 minutes -- that is a pipeline check only, and the 3 Nov rules forbid tuning on it.
- A gap smaller than seed noise (~0.01) is not a win. It should also hold on the `short` reading and not wreck the worst match.
- Prefer parameters with a physical meaning over the best-scoring decimal.

## Git
```powershell
git pull --rebase            # always first
git add src experiments EXPERIMENTS.md BUILD_NOTES.md validation
git commit -m "control radius: switch only within 1.5 m"
git push
```
- Work on `main` directly; algos and run logs are separate files, so conflicts are rare.
- Rerun a result you want to keep **after** committing, so its log has a clean commit hash (a `*` on the leaderboard means it came from uncommitted code).
- Never commit `data/`, notebook outputs, or anything frame-level from the tracking.

## Validation
External matches only -- no hand labels on BAR-ATH (organiser rules of 3 Nov). See `validation/README.md`.
