# Tournament structure snapshots

This folder holds dated, committed snapshots of the World Cup tournament
structure used by Layer 5 (simulate.py). Each file is a controlled, versioned
input that the project owns, not a live scrape.

## Format and progression rules

`wc2026_groups_2026-06-10.csv` encodes the official World Cup 2026 format as of
the pre-tournament cutoff:

- 48 teams in 12 groups of four (columns `group`, `team`).
- Group stage is a round robin; a win is worth three points and a draw one.
- The top two of each group advance, plus the eight best third-placed teams,
  giving a round of 32.
- The knockout is single elimination: round of 32, round of 16, quarter-final,
  semi-final, final.

## Important note on the group assignments

The 48 teams and their group placements are the project's dated draw snapshot
for modelling, consistent with the committed squad-value snapshot. They follow
the official format and were seeded by pot strength, but they are not a scrape
of the official FIFA draw and can be refined later without changing any model
code; only the snapshot file changes. The same honesty applies here as to the
squad values: this is a controlled, reproducible input, not an official record.

## Venue handling

World Cup 2026 is hosted by the United States, Canada, and Mexico. The simulator
treats every match as neutral except a group match involving a host nation,
which that host plays at home. Knockout matches are treated as neutral. Neutral
is the default and host-home is the explicit exception, so no phantom home
advantage biases the bracket.
