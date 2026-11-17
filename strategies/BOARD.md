# Strategy board

Scored on other matches made to look like ours (never on Barça–Athletic). Higher is better.
`score` = balanced spellF1, main labelling. `loose` = same with the looser 1 s labelling: a real
improvement goes up on both. `worst` = lowest single match. Differences under 0.01 are noise.
Quick and full runs are separate leagues: only compare rows of the same mode.

| date | strategy | algo file | mode | score | loose | worst | vs best before | verdict |
|---|---|---|---|---|---|---|---|---|
| 2026-11-14 | Organiser baseline: nearest player to the ball, smoothed over 2 s | `baseline` | quick (no DFL) | 0.342 | 0.344 | 0.254 | first | floor |
| 2026-11-14 | Rules only: a team has the ball when its player is within 1.5 m for 0.4 s; loose ball stays with the last team | `control_tracker` | quick (no DFL) | 0.554 | 0.531 | 0.409 | +0.212 | kept as reference |
| 2026-11-14 | Current best: learned model on team-level features (distance to ball, team shape before vs after) + smoothing that charges for every possession switch | `gbdt_plus` | quick (no DFL) | 0.744 | 0.694 | 0.678 | +0.190 | the one to beat |
| 2026-11-16 | Current best with German league (DFL) matches added to training and testing | `gbdt_final` | quick | 0.744 | 0.682 | 0.665 | first | the one to beat (DFL era) |
| 2026-11-16 | Train the model to care more about short possessions (under 4 s) and the half-second around every change | `gbdt_weighted` | quick | 0.752 | 0.694 | 0.680 | +0.008 | needs a full run: up on DFL (+0.017) and Metrica (+0.012), slightly down on SkillCorner (-0.005) |
| 2026-11-17 | Train the model to care more about short possessions (under 4 s) and the half-second around every change | `gbdt_weighted` | full | 0.766 | 0.703 | 0.692 | +0.016 vs gbdt_final full | **kept**: better in all three leagues and on both labellings; new best |
| 2026-11-17 | Current best with German league (DFL) matches added to training and testing | `gbdt_final` | full | 0.750 | 0.683 | 0.660 | -0.016 | full-run reference |
| 2026-11-18 | Tristan's observations: players converging on a loose ball, 2 s of close control, forward runs after a win, marking at restarts | `gbdt_tristan` | quick | 0.761 | 0.701 | 0.693 | +0.009 |  |
