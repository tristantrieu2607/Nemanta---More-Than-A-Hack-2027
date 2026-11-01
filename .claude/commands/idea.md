---
description: Turn a possession idea into an algorithm, run it, log it, and share it with the team
argument-hint: <the idea in plain words>
---
Implement this possession idea as a new algorithm for the team's experiment loop.

Idea: $ARGUMENTS

1. Read CLAUDE.md and EXPERIMENTS.md, and run `python -m src.leaderboard`. If the idea is already logged (especially as dropped), say so and stop unless I insist. If it would break RULES.md (hand labels or human data about this match, tuning on the public 5 minutes, constants tied to specific frames), say so and stop.
2. Pick a short snake_case name. Copy `src/algos/_template.py` to `src/algos/<name>.py`. Never edit another algorithm's file; import its helpers instead.
3. Implement `predict()`. Every tunable number goes in `PARAMS`, chosen for its physical meaning, not tuned. Write the module docstring for a reader: the possession rule and why it should help. It goes into the technical document.
4. Run `python -m src.run <name> --note "<idea in a few words>"`. Fix any error and rerun.
5. Show `python -m src.leaderboard` and explain in 2–3 sentences what changed: possession count and mean duration vs the public check, and anything that looks like a bug. Don't present the public score as proof one version beats another. Only validation (other games) can rank.
6. Add a row to the EXPERIMENTS.md log and a changelog line to BUILD_NOTES.md.
7. Ask me before sharing. On yes, run `python -m src.share <name> -m "<name>: <idea>"`.
