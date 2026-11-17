# Validation: the external benchmark

**No hand labels on BAR-ATH, for any purpose** (RULES.md; organiser email of 3 Nov: "If your
algorithm has parameters that need to be tuned, you MUST tune them on external data"). Every
decision is made on other matches, made to look like this one. Code: `src/external/`.

## Setup (once per machine, ~3 GB download)
```powershell
pip install -r requirements-starter.txt          # kloppy==3.18.0 parses DFL and Metrica EPTS
python -m src.external.fetch                     # Metrica + SkillCorner (GitHub) + DFL (figshare)
python -m src.external.bench convert             # -> data/external/cache/*.npz (~20 MB / match)
python -m src.external.bench info                # what is cached, label stats per match
```
If figshare is blocked, download the DFL files by hand from
https://doi.org/10.6084/m9.figshare.28196177 (for each of the 7 matches: the `matchinformation`
XML and the `positions` XML, file names contain `DFL-MAT-<id>`), unzip them anywhere under
`data/external/idsse/`, then run `convert` again. Nothing under `data/` is ever committed.

## The datasets
| source | matches | tracking | labels (provider) | licence | role |
|---|---|---|---|---|---|
| `idsse` DFL Bundesliga 22/23 (Bassek et al. 2025, *Sci Data* 12:195) | 7 | 25 Hz optical, clean | per-frame BallStatus + BallPossession | CC BY 4.0; organisers OK'd it (issue #16) | main: the same kind of flags our GT is built from |
| `skillcorner` A-League 24/25 | 20 | 10 fps automatic broadcast, resampled to 25 | phases of play (alive + team) | MIT, credit SkillCorner | real automatic-tracking noise, second league |
| `metrica` sample data | 3 | 25 Hz, same provider; game 3 in the ATD's EPTS format | from events | no licence file -- **ask the organisers before citing it** | closest to the ATD |

## Labels: the challenge rule, applied to every source (`labels.py`)
1. Loose ball / no owner inside live play -> the team that last had it keeps it.
2. A spell shorter than `min_control_s` is absorbed by the possession it interrupts (a touch,
   deflection or duel win is not a change), except the first spell after a restart.

Readings: `main` = 2.0 s, `short` = 1.0 s, `raw` = provider flags as they are. The private GT's exact
convention is unknown, so runs report `main` and `short`: a change that only wins on one is fitting a
convention. Metrica events: PASS / CARRY / SHOT / SET PIECE / RECOVERY give the team the ball; BALL
LOST never starts a possession; CHALLENGE never changes it; dead from ball out / goal / foul /
offside until the next restart.

## Degradation: making clean tracking look like the ATD (`degrade.py`)
Profiles are physical descriptions of a tracker, drawn with +/-20% per copy (domain randomisation):
- team-level only: 46 anonymous slots (24 home, 22 away), tracks cut into fragments (~45 s) that
  land on random free slots, short occlusions, GK mostly on the flagged GK slot, position jitter;
- ball missing in long blocks: per stoppage one visible block (usually the restart), ~27% of dead
  frames seen; live gaps of ~6 s (2-40 s) until ~8% of live frames are missing, more often on fast
  balls; provider gaps < 1 s interpolated, sightings < 0.5 s inside gaps dropped. Sources that
  already lose the ball (Metrica, SkillCorner) only get what is missing.
- `clean` (no noise) and `harsh` (more and longer gaps, noisier) for stress tests.

Targets come from the repo docs (41% of graded frames without ball) and aggregate figures other teams
published from label-free inspection (median gap ~11 s, ball seen ~27% dead / ~92% live). After
`atd` degradation: Metrica 1 has 41% missing, 24% dead seen, 90% live seen.

**Realism check (not a target).** The organiser baselines keep their full-match order on the `atd`
benchmark (smooth50 > smooth125 > smooth25 > raw, as on the private full match: 0.2433 > 0.2242 >
0.1996 > 0.0486). `harsh` flips smooth50/125, like the public 5 minutes do. Absolute scores are
~0.1 higher than on BAR-ATH: the benchmark is easier than the real match, so trust differences
between versions, not the level.

## Reading the numbers
- `balanced` = mean of the source means (20 SkillCorner matches must not drown 7 DFL ones).
- Seed-to-seed noise ~0.005-0.01; differences below that are noise. Look at `worst` (lowest match).
- Smart Tagging exists only for BAR-ATH: `ExternalMatch.tags` is empty, so tag-based logic is
  untested by the benchmark. If an algo uses tags, report it with and without them.
- `ExternalMatch.players_speed` / `ball_speed` are m/s computed from positions; the ATD speed field's
  unit is unverified -- compute speed from positions in algos so both behave the same.
