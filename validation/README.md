# Validation

**Not built yet.** Until it is, there is no legal way to rank versions against each other.

## What the rules allow (RULES.md, updated by the organisers on Oct 22–23)
- **No hand labels on the competition match (BAR-ATH), in any amount, for any purpose** — not for the submission, not to train, not to validate. Watching the video to understand and debug is fine; turning what you saw into labels, corrections, or constants tied to specific frames is not.
- **The public 5 minutes are a pipeline check only** — not training data, not a tuning set, not for choosing between versions.
- **No human-produced data about this match from any source** (Opta, StatsBomb, corrected tracking...).
- **Data from other games is fine, in any amount**, hand-labelled or not, as long as the organisers can access it and the technical document explains it.

## Plan
Build the validation set from **other games** with tracking + possession information, e.g. openly published tracking + event data, converted to the same `H/A/D` per-frame format and run through the same `predict()` functions. Ideally degraded to look like our data (ball missing ~41% of frames, broken tracks) so it tests what matters.

Every source used goes in the technical document: what it is, who labelled it, how much.
