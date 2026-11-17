# Experiments

Scores live in `experiments/runs/` and `python -m src.leaderboard`. This file is the human side: what we tried, what we learned, what's next.

Decide on the external benchmark (balanced spellF1, `main` reading, also check `short` and the worst match), never on the public 5 minutes (17 possessions, pipeline check only). See validation/README.md.

## Log
| # | Date | Who | Idea | Algo file | Best result | Verdict / what we learned |
|---|------|-----|------|-----------|-------------|---------------------------|
| 0 | 2026-11-01 | Tristan | Organiser baseline: nearest player + majority vote | `baseline.py` | public 0.3256 (window 50), full match 0.2433 | Floor to beat. window 125 wins public but loses full match — don't tune on public. |
| 1 | 2026-11-08 | Tristan | External benchmark built (SkillCorner 20 + Metrica 3; DFL pending download) | `baseline.py` | ext balanced main: raw 0.047, w25 0.259, **w50 0.347**, w125 0.332 (seed 0) | Same order as the private full match (50 > 125 > 25 > raw) -> the `atd` benchmark ranks like the real match. Levels ~0.1 higher than BAR-ATH: compare versions, not levels. `harsh` flips 50/125 (like the public window). |
| 2 | 2026-11-09 | Claude | Rules: control radius 1.5 m + 0.4 s hold + loose ball keeps last team + dead = long ball gap & slow players / ball off pitch | `control_tracker.py` | ext 0.562 (2 seeds), short 0.537 | +0.21 over baseline with physical defaults; 2.5 m/0.2 s collapses (0.27), so the radius matters. Kept as a feature source. |
| 3 | 2026-11-09 | Claude | HistGB per-frame P(H/A/D) on 51 team-level features, mirrored training + mirrored prediction, Viterbi switch costs | `gbdt_viterbi.py` | **ext 0.714 (2 seeds)**, short 0.663, worst 0.651 | +0.36 over baseline. Flat in switch_team 8-32 / switch_dead 6-24 -> costs are not a knife-edge. Predicts 0.88-0.94x the true possession count. |
| 4 | 2026-11-09 | Claude | Snap each H<->A change to the new team's first touch | `gbdt_snap.py` | ext 0.700 (2 s back), 0.713 (1 s) | Dropped: hurts or neutral. The classifier already sees future context, its changes are well placed. |
| 5 | 2026-11-09 | Claude | gbdt_viterbi + control_tracker state as features | `gbdt_hybrid.py` | ext 0.720 (2 seeds), short 0.666 | Paired per match +0.002 (14/23 better, sd 0.01) = noise. Not worth the extra moving part on its own. |
| 6 | 2026-11-09 | Claude | Set-piece layouts as features (throw-in / corner / goal-kick / kick-off / free-kick wall from player positions) | `gbdt_plus.py` groups=+setpiece | ext 0.721 vs 0.722 without (seed 0) | Dropped: flags do separate dead from live (throw-in 15% vs 5% of frames) but the model already gets dead balls from ball gaps + speed. |
| 7 | 2026-11-09 | Claude | Change-point context: control and speed over the past vs the next 1/3/10 s | `gbdt_plus.py` groups=base,rule,asym | ext 0.733 vs 0.722 (seed 0), short 0.677 vs 0.666 | **Kept.** Gain on both sources and both readings, above seed noise (~0.003). Without the rule features 0.726, so rule is worth ~0.007 once asym is in. |
| 8 | 2026-11-09 | Claude | Absorb predicted possessions < 1-2 s (the labels' own control rule) | `gbdt_plus.py` min_spell_s | +/-0.001 | Dropped: Viterbi already leaves no short spells. |
| 9 | 2026-11-09 | Claude | Team shape before vs after: depth, width, length, players within 8 m of the ball, speed, each team in its own attacking frame | `gbdt_plus.py` groups=base,rule,asym,shape | **ext 0.751** (seed 0), short 0.693, worst 0.665 | **Kept.** +0.018 / +0.016 on both sources: a team that wins the ball spreads and pushes up, the one that loses it compacts and drops. |
| 10 | 2026-11-09 | Claude | + defensive-line height (2nd-deepest player), team length, players goal-side of the ball, 1-20 s before vs after; rule features dropped | `gbdt_plus.py` groups=base,asym,shape,shape2 (**new default**) | **ext 0.753** (seed 0), short 0.701, worst 0.677 | Kept, small: +0.005 over base,asym,shape (0.749). Rule features now worth +0.002 -> dropped (one moving part less). |
| 11 | 2026-11-09 | Claude | Robustness: train on `atd`, test on `atd` / `harsh` / `clean` (unseen seed 7, stride 16) | `gbdt_plus.py` stride=16 | atd 0.754, harsh 0.721, clean 0.748 | Degrades gracefully on noisier tracking (-0.033) and does not overfit our noise model (clean ~ atd). stride 16 = stride 8 in score, half the memory -> new default. Mixed atd+harsh training run hit the 2 h limit, still to do. |
| 12 | 2026-11-16 | Claude | Wait before the 2nd-half kickoff is `D` (organisers' FAQ: frames 110,808-113,741 are graded and all dead) | `gbdt_final.py` | same as gbdt_plus on the benchmark by construction | **Kept** (documented fact, not tuned). Barca-Hackathon found their network called 63% of that stretch live before adding the same rule; False9 added it on Nov 14. |
| 13 | 2026-11-17 | Claude | Training weight x2 on frames inside possessions under 4 s and x2 within 0.5 s of a change (idea from Barca-Hackathon / False9) | `gbdt_weighted.py` | **full run, 30 matches, 2 seeds: 0.766** vs gbdt_final 0.750; short 0.703 vs 0.683; worst match 0.692 vs 0.660 | **Kept, new best.** Up in every league on both readings (DFL 0.744 vs 0.725, SkillCorner 0.761 vs 0.757, Metrica 0.792 vs 0.768); seed sd <= 0.005. |
| 14 | 2026-11-18 | Claude | Minimum stoppage length 1 s / 2 s: a dead-ball blip inside live play is given back to the team before it (Barca-Hackathon: worth 0.061 to them; PerosinhoFC: 1.4 s) | `gbdt_minstop.py` | quick: 0.7523 vs 0.7516 (both readings +0.0005) | Dropped: only 6 blips in 17 matches. Our costs (team switch 16 < two dead switches 24) already make a fake stoppage dearer than a direct change, so the shortcut those teams closed never opens. |

## Backlog
Ranked. Move to the log once tried.

Seen in other teams' Nov 9-16 updates: Agamemnon cut their own estimate 0.68 -> 0.646 and Smart Tagging's value 0.09 -> 0.035 (their imitation tags were too tidy) -> we keep tags out. Barca-Hackathon: video/broadcast ball worth +0.004-0.010 at most -> skip; seed ensembles +0.005-0.010; loss weight on short possessions. False9: adding DFL to training +0.03-0.07. AndaAnda: spell-balanced weights failed.

Field scan 16 Nov (64 of the 169 repos updated, 190 new ones): honest external scores cluster at 0.55-0.72 spellF1. Repeated findings: snapping changes to a touch does nothing (SYEDMDSAAD, us); one training seed moves the score by up to 0.018, so average seeds (younis-y, Barca-Hackathon); bigger networks tie small ones (hfarid19); a learned dead-ball model beats hand rules (SYEDMDSAAD, DFL 0.365 -> 0.431); possession from team shape while the ball is hidden is worth ~0.09 on DFL (NextConsulting); PFF FC's free World Cup 2022 data (48 games) is a fourth training source (hfarid19, 7 repos mention it). Rule-risk patterns still around: training on the public window, ball detectors fine-tuned on this match's own video.

Field scan 18 Nov (final pushes before the deadline): Barca-Hackathon submitted v3 (8 TCNs, held-out Metrica 0.853, alt reading 0.721; Smart Tagging measured at +0.001 and dropped; video ceiling +0.010, broadcast +0.004; extra loss weight on short possessions, as in our row 13). PerosinhoFC submitted a stoppage decoder with a 1.4 s minimum (held out 0.650). NextConsulting moved to learned GBDT evidence trained on Sportec. sulimovp hand-reviewed 192 transitions of BAR-ATH (frames 11466...) -- hand labels on the provided game, a rule risk; don't copy.

Done: 1-5 (control radius, hysteresis, ball gaps, dead-ball rules, classifier + Viterbi) -> rows 2-9.
1. More shape/change-point features: defensive line height, players ahead of the ball, longer windows (20 s).
2. Robustness: train on a mix of `atd` + `harsh` noise (the run timed out; the atd-only result is row 11).
3. Cross-league transfer: train SkillCorner only -> test Metrica (same provider as BAR-ATH).
4. Label reading: `train_reading=mix` is implemented in `_learned.py` (half the rows labelled each way), not yet benchmarked. Short is ~0.05 lower than main.
5. DFL / IDSSE once downloaded: re-run everything on 3 sources before freezing.
6. CV: detect the ball in the video on frames where the ATD has none (input only; nothing trained or tuned on the video).
7. Smart Tagging: only testable with imitation tags generated from external labels (ask before relying on it).

## Validation set
External benchmark, see validation/README.md. `python -m src.external.bench info` lists the cached matches.
| Source | Matches | Status |
|--------|---------|--------|
| SkillCorner A-League 24/25 | 20 | cached |
| Metrica sample data | 3 | cached (licence question to organisers pending) |
| DFL / IDSSE Bundesliga 22/23 | 7 | cached 2026-11-16 (downloaded by browser; figshare blocks scripts). 267-315 possessions per match under `main`, mean 9-13 s. Baseline on DFL alone: w25 0.272 / w50 0.330 / w125 0.283, same order as the organisers' full-match figures |
