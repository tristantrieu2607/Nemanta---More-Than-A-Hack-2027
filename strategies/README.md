# Strategies: write the idea in words, Claude does the rest

No coding needed. One idea = one small text file.

## How to add an idea
1. Copy `_TEMPLATE.md`, rename it (e.g. `keeper-has-it.md`), fill it in. Plain football language is fine.
2. Give it to Claude, any of these works:
   - drop the file into this `strategies/` folder and say **"run the new strategies"**;
   - attach the file (or just type the idea) in the chat and say **"test this strategy"**.
3. Claude turns it into code with real numbers, scores it on other matches, and writes the result
   at the bottom of your file and as a new row in `BOARD.md`.
4. Read the verdict, tweak the idea or write the next one. Repeat.

## What makes a good idea
- It describes something you could *see* from dots on a pitch: 22 players (we know the team, not
  the name), and a ball that disappears about 40% of the time. No video, no player names.
- It says when the ball changes team, when play is stopped, or who has the ball while it's hidden.
- Give rough numbers if you have a feel ("about 2 metres", "a couple of seconds"). Claude picks
  sensible physical values and checks nearby ones; it never fishes for a lucky decimal.

## What the score means
`BOARD.md`: higher is better, 1.0 is perfect. Under 0.01 difference is noise. An idea is **kept**
only if it beats the current best on both the `score` and `loose` columns and doesn't wreck `worst`.
Quick runs (minutes) screen ideas; anything promising gets a full run before we trust it.

## Rules we can't break (competition rules)
- No idea may use the answer for our match, or anything a person noted by watching it.
- No numbers chosen by looking at our match's 5 public minutes. Everything is tuned on other matches.
