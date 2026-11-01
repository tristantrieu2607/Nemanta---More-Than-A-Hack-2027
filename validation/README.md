# Our hand labels

Used only to **validate** algorithms (allowed by RULES.md). Submissions must come from code.

## Format
One CSV per stretch of video. Each row means "from this moment on, it is <label>". The last row is `X`, closing the stretch.
```
time,label
12:30.40,H
12:41.12,A
12:43.00,D
12:58.84,A
14:30.00,X
```
- `time` is the time in the **tactical** video (`mm:ss.ss` or `h:mm:ss.ss` or seconds). `frame,label` also works.
- `H` Barcelona has the ball · `A` Athletic has it · `D` ball out of play.
- Graded halves: 1st half 7:27.40–1:00:26.04, 2nd half 1:13:52.32–2:06:47.88 (video time).

## Rules for labelling (match CHALLENGE.md)
- Possession changes only when a player of the other team has the ball **under control**. A touch, a deflection, or a won header without control is not a change.
- Loose ball / 50-50: the team that last had it keeps it.
- Ball out of play → `D` until the restart is taken (throw-in thrown, corner kicked...). The restart goes to the team taking it.
- Use a player that steps frame by frame and shows fractions of a second (e.g. mpv: `.` steps a frame, `o` shows the time).

## Folders
- `labels/` — validation stretches, outside the public 5 minutes. Aim for 20–30 min in total, spread across both halves. Name files `h1_1230-1430.csv` etc.
- `calibration/` — label 2–3 min **inside** the public window (7:27.40–12:27.36) *without looking at the answer key*. `python -m src.check_labels` scores it against the official truth. If it's low, our idea of "control" differs from theirs: compare and adjust before labelling more.

Check files any time: `python -m src.check_labels`
