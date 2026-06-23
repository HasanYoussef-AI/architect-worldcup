# Tournament structure inputs

This folder holds the committed World Cup 2026 tournament structure used by Layer
5 (simulate.py). These are controlled, versioned inputs the project owns, sourced
once from official material and reproduced into files, not scraped at run time.

## Files

- `wc2026_groups_2026-06-10.csv`: the real, verified 2026 group draw, 48 teams in
  12 groups of four (columns `group`, `team`).
- `r32_thirdplace_assignment_2026.csv`: the official FIFA Annex C table, all 495
  combinations of which eight third-placed groups qualify, each mapping every
  group winner that faces a third-placed team to the third-place group it plays.
  Parsed deterministically from the published FIFA combinations table and
  validated: 495 distinct combinations, each assignment a permutation of the
  qualifying groups with every third in its winner slot's eligible set.

## Names matched to the data

Three group names use the results dataset's spelling so every team resolves to a
rating: Czech Republic (for Czechia), Turkey (for Turkiye), and Curacao with a
cedilla (for Curacao). These are the same nations under their older dataset names.

## Format and progression

- Group stage is a round robin; a win is worth three points and a draw one.
- The top two of each group advance, plus the eight best third-placed teams,
  giving a round of 32, then single elimination to the final.
- Within-group ties follow the FIFA ladder, head-to-head points then head-to-head
  goal difference then head-to-head goals, re-assessed among still-tied teams,
  before overall goal difference and goals, then fair play, FIFA ranking, lots.
- Third-placed teams are ranked on a separate ladder with no head-to-head, since
  they never met: points, goal difference, goals, fair play, FIFA ranking.
- The round-of-32 bracket and the knockout progression are encoded in simulate.py
  from the published match schedule.

## Venue handling

Hosts United States, Canada, and Mexico play their group matches at home; every
other group match and every knockout match is neutral, so no phantom home
advantage biases the bracket.

## Fair play fallback

The model cannot predict cards, so a simulated tie that reaches the fair-play step
has no card data and falls back to the FIFA ranking, proxied by team rating, then
to a random draw. How often this fallback is reached is counted and reported. This
is a deliberate, honest choice, not a fabricated card-prediction model.

## Sources

- 2026 FIFA World Cup knockout stage, the round-of-32 schedule, the 495-row
  combinations table, and the bracket progression:
  https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
- 2026 FIFA World Cup: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
