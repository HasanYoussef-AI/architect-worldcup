# Squad value snapshots

This folder holds dated, committed snapshots of national-team squad market
values used by Layer 4 (squad.py). Each file is a controlled, versioned input,
not a live scrape. Transfermarkt is deliberately not scraped at run time
because that is fragile and breaks reproducibility.

## Important note on the figures

The values in `squad_values_2026-06-10.csv` are an approximate dated snapshot
for modelling, not official figures. They are reasonable present-day estimates
of total squad market value in millions of euros, gathered from general
knowledge to give Layer 4 a controlled input. They can be refined later with
more precise figures without changing any model code; only the snapshot file
changes.

## File format

- Filename encodes the snapshot date: `squad_values_YYYY-MM-DD.csv`.
- Columns: `team`, `market_value_eur_m`.
- `team` must match the team naming used in the match dataset (Layer 1).
- `market_value_eur_m` is total squad market value in millions of euros.
- The snapshot date should align with the run `as_of_date` so the squad
  picture respects the same leakage boundary as the rest of the pipeline.
