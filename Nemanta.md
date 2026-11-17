# More than a Hack 2027 — Phase 1: technical document

Team: **Nemanta** · Repository: https://github.com/tristantrieu2607/Nemanta---More-Than-A-Hack-2027
Submitted algorithm: `src/algos/gbdt_tristan.py` · one command regenerates the file (section 7).

## 1. Approach in one paragraph

Every frame of BAR–ATH is labelled by a gradient-boosted tree classifier that reads ~240 team-level
features computed from the automatic tracking (positions only: no player identities, no Smart
Tagging, no video), followed by a three-state Viterbi decoder (H, A, D) that charges a fixed cost
for every possession switch. The classifier is trained **only on open external matches** (DFL /
IDSSE, SkillCorner, Metrica sample data) that we relabelled to the challenge's possession rule and
degraded to look like this match's tracking. Every design choice was decided on that external
benchmark, never on the provided game. The overall shape — team-level features, a per-frame model
trained on degraded external matches, a switch-cost decoder — is the pattern most public entries
converged on; our contribution is in the features (section 3), the benchmark and the testing.

## 2. Why this and not something else

spellF1 counts possessions, not frames: a 1.5 s possession is worth as much as a 40 s one, and a
single flicker splits a possession in two. Two things therefore matter most: (a) knowing which team
controls the ball when the tracking does not say, and (b) not flickering. A learned per-frame model
handles (a) far better than hand rules; the switch-cost decoder handles (b).

External balanced spellF1 (mean of the three source means; organisers' scorer):

| step | spellF1 |
|---|---|
| organiser baseline (nearest player, majority vote, window 50) | 0.350 |
| hand rules: control radius 1.5 m, 0.4 s hold, loose ball stays with the last team | 0.562 |
| gradient-boosted trees + Viterbi on team-level features | 0.714 |
| + context before vs after each frame | 0.733 |
| + team shape before vs after (spread, depth, defensive line, players goal-side) | 0.753 |
| + DFL added; training weights on short possessions and change points (`gbdt_weighted`) | 0.766 (30 matches, 2 seeds) |
| + observation features (section 3.3): **submitted** (`gbdt_tristan`) | 0.761 vs 0.752 for `gbdt_weighted` on the 17-match screen (better in all three leagues, both readings) |

Tried and dropped (no gain beyond seed noise, ~0.005): set-piece layout flags, snapping each change
to the new team's first touch, a minimum possession length, a minimum stoppage length, the
rule-based tracker's output as features.

## 3. The pipeline

### 3.1 Input
The tactical ATD only: per frame, x/y of the 46 track slots with their team (from the XML) and the
ball's x/y or "missing". Speeds are recomputed from positions (m/s); the provided speed field is not
used. Attack direction per half comes from the teams' mean positions.

### 3.2 Base features (per frame, per team)
Ball: seen / missing, length of the current gap, time since / until the ball is seen, ball position
in the home-attack frame, distance outside the pitch, ball speed. Per team: nearest and second
nearest player to the (interpolated) ball, players within 3 m and 8 m, centroid, spread, mean
speed, tracked players. Rolling context over 1–25 s; the last / next team in close control.
Change points: control and speed over the past vs the next 1, 3 and 10 s. Team shape: depth,
width, length, defensive-line height (2nd-deepest player), players goal-side of the ball, each over
the past vs the next 1–20 s, in each team's own attacking frame.

### 3.3 Observation features (from watching football, written as general rules)
These come from watching how possession changes look in general, then expressing each as a rule
any match obeys. No frame, minute or event of BAR–ATH is encoded.
- **Convergence**: players near a loose ball close in on it; if both teams do, it is contested →
  closing speed of each team's two nearest players; players within 15 m approaching at > 2 m/s.
- **Control lasts**: a team has the ball once one player stays close for about 2 s → length of the
  current spell of close control (≤ 1.5 m and 0.5 m closer than any opponent), longest spell in the
  last / next 3 s.
- **Reaction to a win**: attackers push up and the nearest players charge, the other side holds or
  drops → mean speed towards the attacked goal and share of players sprinting (> 4 m/s), before vs
  after.
- **Restart marking**: at a throw-in the restarting team is marked, the other team's players are
  free → near the ball, mean distance to the nearest opponent and players marked within 2 m.

### 3.4 Model and decoder
sklearn `HistGradientBoostingClassifier` (300 iterations, learning rate 0.08, 31 leaves, min 400
samples per leaf, L2 1.0, fixed seed 2027, no early stopping) on every 16th graded frame of each
external match × 2 degradation seeds. Each training row is duplicated with home and away swapped,
and at prediction the model is averaged with its swapped view, so it cannot learn "home has the ball
more". Frames inside possessions shorter than 4 s and within 0.5 s of a change get weight 2. The
Viterbi costs are 16 (H↔A) and 12 (alive↔dead) in log units; the score is flat between 8 and 32.
Kick-off rule: the graded stretch before the second-half kick-off (frames 110,808–113,741, which
FAQ.md documents as dead ball) is forced to `D`. No graded frame is `X`.

## 4. Possession rules implemented
The external labels are converted to the challenge's rule, and the model learns that rule:
- ball dead → `D`; dead until the restart is taken;
- loose ball / 50-50 / ball in flight → the team that last had it keeps it;
- a change needs control: a spell shorter than 2 s is absorbed by the possession it interrupts
  (a touch, deflection or won header), except the first spell after a restart;
- every restart belongs to the restarting team.
Because the organisers' exact convention is not public, every result is also reported with a
1 s threshold ("loose" reading); a change was kept only if it improved both.

## 5. Assets: used and dropped
| asset | used? | why |
|---|---|---|
| Tactical ATD + XML | yes | the only input |
| Smart Tagging | no | does not exist for the external matches, so its effect cannot be validated; public teams that measured it found ~0–0.035 |
| Homography, tactical video, broadcast video + ATD | no | public measurements put a perfect ball detector's ceiling at about +0.01; the video may not go to cloud services |
| Public 5-minute ground truth | pipeline check only | scored once for the submitted file: spellF1 0.607, macroF1 0.901 (12 predicted vs 17 true possessions); no decision used it |

## 6. Training and validation data (none from the provided game)
| dataset | matches | labels | licence |
|---|---|---|---|
| DFL / IDSSE, Bassek, Rein, Weber et al., *Sci Data* 12, 195 (2025), doi:10.6084/m9.figshare.28196177 | 7 Bundesliga 22/23 | per-frame BallStatus + BallPossession from the provider (no hand labels by us) | CC BY 4.0; allowed by the organisers in issue #16 |
| SkillCorner Open Data (github.com/SkillCorner/opendata, commit 4340d27) | 20 A-League 24/25 | phases of play (provider) | MIT, © SkillCorner |
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit e706dd5) | 3 anonymised | derived by script from the provider's events | public repository, no licence file |

Labels were derived by script (`src/external/labels.py`); nobody on the team labelled any frame.
To resemble BAR–ATH, each external match is degraded (`src/external/degrade.py`): identities
removed and tracks cut into fragments on 46 anonymous slots, occlusions and position jitter, the
ball hidden ~41% of graded frames in long gaps, mostly during stoppages (seen ~27% of dead frames,
~92% of live ones), with parameters drawn ±20% per copy. Check: the organiser baselines keep their
published full-match order on this benchmark (window 50 > 125 > 25 > raw).
Validation: grouped 5-fold over matches (a match is never scored by a model trained on it),
2 degradation seeds, organisers' `possession_metrics.evaluate`.

## 7. Reproduction
```bash
python -m venv .venv && .venv\Scripts\activate        # Python 3.12
pip install -r requirements-team.txt                   # numpy, pandas, scikit-learn, kloppy pinned
# match data in data/tactical/ (DATA.md); external data:
python -m src.external.fetch                           # SkillCorner + Metrica; DFL from figshare
python -m src.external.bench convert                   # (DFL: download by browser if blocked, see validation/README.md)
python -m src.run gbdt_tristan --no-external           # -> data/submissions/gbdt_tristan__*.csv
python evaluation/validate_submission.py data/submissions/<file>.csv
```
The model has a fixed seed and no early stopping, so the same data and pinned versions give the
same file. SHA-256 of the submitted file: `319594ea5e5c1bd976d02ed100e607ad0e33cbc4201d7d6bffc78fd350db52fb`.

## 8. Limitations and what we would do next
- **Short possessions** are still the main loss: the switch cost that removes flicker also merges
  real 1–2 s possessions (we predict ~6–12% fewer possessions than the external truth).
- **Label convention**: scores drop ~0.06 under the 1 s reading; if the private truth keeps very
  short possessions, we lose most there.
- **Benchmark realism**: the external benchmark is easier than BAR–ATH (the baseline scores ~0.35
  there vs 0.24 on the full match); we trust differences between versions, not the level.
- Next: average several training seeds (one seed moves scores by up to ~0.02 in public reports), a
  duration-aware decoder for short possessions, more training leagues (e.g. PFF FC World Cup 2022).

## 9. Credits
Organisers' scorer and baselines (Metrica Sports, MIT). Data: DFL/IDSSE (CC BY 4.0), SkillCorner
(MIT), Metrica sample data. Ideas taken from public entries and credited: training on degraded
external matches with a switch-cost decoder (Team Sigue, Agamemnon, Barca-Hackathon, False9), the
ball-gap statistics used for degradation (Barca-Hackathon), extra loss weight on short possessions
(Barca-Hackathon, False9). Code written by our team with AI assistance (Claude); no match data was
given to any hosted AI service.
