# Architect WorldCup

A live World Cup 2026 bracket re-simulator built verification-first. The point of
this project is not the forecast. The point is that you can trust it. Every number
it produces is defended by a real test, leakage is a build failure rather than a
footnote, and the experiments that did not work are kept in the record next to the
ones that did.

Out of sample, across eight non-overlapping backtest windows, the goal model
scores a mean Ranked Probability Score of **0.1575** with a standard deviation of
**0.0140**, against a base-rate baseline of **0.2292** on the same windows. Lower
is better. That spread, not any single lucky window, is the headline. A single
frozen calibration window, used throughout as the reproduction anchor, sits at
**0.1611** against a **0.2135** baseline.

---

## 1. Summary

Architect WorldCup ingests the full history of international football, rates every
national team, fits a Dixon-Coles goal model, adjusts for current squad value,
and runs a Monte Carlo simulation of the real 2026 tournament with the actual FIFA
group and bracket rules. It produces, for every team, the probability of reaching
each knockout round and of winning the tournament.

It is a command-line system. It runs on a manual trigger, writes versioned JSON
artifacts with full provenance, and is reproducible to the digit from a fixed seed
and immutable dated data snapshots. There is no web app and no scheduler. The
architecture and the honesty are the product.

---

## 2. Thesis: independence and verifiability over raw accuracy

The goal was never to beat the betting market. The market, with its liquidity and
its army of professional modelers, is extremely hard to beat, and a system whose
only claim is "slightly more accurate than a bookmaker" is both unlikely to be
true and impossible to verify from the outside.

This project makes a different claim: that the forecast is **independent** and
**verifiable**.

Independent, because it deliberately uses no bookmaker odds. A model that ingests
market prices is, in large part, copying the market. By refusing odds as an input,
this system can later be benchmarked against the market as a genuinely separate
opinion rather than a derivative of it.

Verifiable, because every claim is backed by a test that runs in the normal suite.
The leakage guard is not a comment promising good behavior, it is an assertion
that fails the build. The headline accuracy number is not a single run, it is a
rolling-origin backtest with a measured spread. The features are constructed
point-in-time, and a test proves that a feature for a match cannot see that match
or anything later. When an idea was tried and did not help, the measurement that
rejected it is committed alongside the code.

Accuracy matters, but it is downstream of trust. A number you cannot verify is
worth nothing, however good it looks.

---

## 3. Architecture

The system is a seven-layer pipeline. Each layer is a module with a single
responsibility, and the layers communicate through plain data, not through shared
mutable state. The model layers emit a versioned JSON artifact, and every
downstream consumer reads that JSON rather than recomputing anything. This
separation of model from presentation is the spine of the design.

| Layer | Module | Responsibility |
| --- | --- | --- |
| 1 | `ingest.py` | Load match data into immutable dated snapshots, enforce the leakage guard |
| 2 | `ratings.py` | Transparent hand-written Elo ratings |
| 3 | `model.py` | Dixon-Coles goal model via penaltyblog |
| 4 | `squad.py` | Bounded squad-value adjustment to the ratings |
| 5 | `simulate.py` | Monte Carlo simulation of the real 2026 tournament |
| 6 | `calibrate.py` | RPS scoring, single-window and walk-forward backtests |
| 7 | `ablation.py` | With-and-without harness to measure each layer |

Two further modules support the layers without being layers themselves:
`artifact.py` owns the versioned JSON output and provenance log, and `pipeline.py`
is the single entry point that wires the layers together. Two more modules exist
to interrogate the system: `audit.py` runs the data-integrity and overconfidence
checks, and `ensemble.py` holds a gradient-boosting experiment that is documented
below precisely because it did not win.

### How the layers actually call each other

```mermaid
flowchart TD
    D1["martj42 results snapshot"]:::data
    D2["squad value snapshot"]:::data
    D3["group draw + Annex C R32 table"]:::data

    L1["Layer 1 ingest.py<br/>load + leakage guard"]:::layer
    L2["Layer 2 ratings.py<br/>Elo"]:::layer
    L4["Layer 4 squad.py<br/>adjust Elo by squad value"]:::layer
    L3["Layer 3 model.py<br/>Dixon-Coles goal model"]:::layer
    L5["Layer 5 simulate.py<br/>Monte Carlo bracket"]:::layer
    ART["artifact.py<br/>versioned JSON + provenance"]:::out
    CON["Consumers: verification gates,<br/>comparison, this README, future LLM layer"]:::out

    D1 --> L1
    L1 -->|guarded matches| L2 --> L4
    L1 -->|guarded matches| L3
    D2 --> L4
    L4 -->|squad-adjusted ratings| L5
    L3 -->|goal model| L5
    D3 --> L5
    L5 -->|predictions| ART
    ART -->|reads JSON, never recomputes| CON

    L6["Layer 6 calibrate.py<br/>RPS, walk-forward backtest"]:::layer
    L7["Layer 7 ablation.py<br/>with/without harness"]:::layer
    L1 --> L6
    L3 --> L6
    L2 --> L7
    L6 --> L7
    L6 -->|RPS artifacts| ART
    L7 -->|ablation artifacts| ART

    classDef layer fill:#0A1A1F,stroke:#C9A84C,stroke-width:2px,color:#E8EAEC
    classDef data fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
    classDef out fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
```

The forecast path runs left to right: guarded matches feed both the rating layers
and the goal model, squad value nudges the ratings, and the simulator consumes the
adjusted ratings and the goal model to produce predictions, which become the JSON
artifact. Calibration and ablation form a separate evaluation track that reuses
the same rating and goal-model code rather than forking it, and they emit their own
versioned artifacts.

### How the goal model and the simulator communicate

The goal model and the simulator are deliberately separate. Dixon-Coles owns the
scoreline distribution. The simulator owns the tournament logic and never second
guesses the goal model. They meet at a single seam: a sampler that draws a
scoreline from the Dixon-Coles joint goal matrix.

```mermaid
flowchart LR
    DC["model.fit_model<br/>Dixon-Coles on guarded matches"]:::layer
    GRID["model.predict<br/>joint goal matrix per fixture"]:::layer
    SF["build_score_fn<br/>sample a scoreline"]:::accent
    GROUP["group match<br/>real result if played,<br/>else sampled scoreline"]:::layer
    STAND["tiebreaker ladder<br/>head-to-head before overall GD"]:::layer
    KO["knockout match<br/>90 minutes, draw resolved by<br/>strength-weighted coin flip"]:::layer
    PROB["per-team stage and title probabilities"]:::out

    DC --> GRID --> SF --> GROUP --> STAND --> KO --> PROB

    classDef layer fill:#0A1A1F,stroke:#C9A84C,stroke-width:2px,color:#E8EAEC
    classDef accent fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
    classDef out fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
```

The simulator encodes the genuine 2026 format: 12 groups of four, the real group
draw, the FIFA within-group tiebreaker ladder where head-to-head results are
applied before overall goal difference, a separate ranking of the third-placed
teams to fill the round of 32, and the official 495-row Annex C table that maps
each combination of qualifying third-placed teams to specific bracket slots,
parsed from the published schedule and committed as a controlled input.

---

## 4. Verification and anti-leakage

This is the heart of the project. A forecasting system is only as trustworthy as
its weakest path to seeing the future, so leakage is treated as the primary risk
and is engineered against at every layer.

```mermaid
flowchart TD
    ASOF["as_of cutoff date"]:::accent
    GUARD["apply_leakage_guard<br/>keep only date &lt;= as_of"]:::layer
    SPLIT["prepare_backtest<br/>split into training and holdout"]:::layer
    ASSERT["assert_no_leakage<br/>training max date &lt; holdout start"]:::layer
    SCORE["score the holdout with RPS"]:::layer
    FAIL["FAIL THE BUILD"]:::accent
    PIT["point-in-time features<br/>ability fit only on data before each match"]:::layer
    GATES["pytest gates run on every change:<br/>no-leakage, same-window invariant,<br/>reproduction, probability validity"]:::out

    ASOF --> GUARD --> SPLIT --> ASSERT
    ASSERT -->|overlap detected| FAIL
    ASSERT -->|clean| SCORE
    PIT --> SCORE
    GATES -.->|enforce| ASSERT
    GATES -.->|enforce| PIT

    classDef layer fill:#0A1A1F,stroke:#C9A84C,stroke-width:2px,color:#E8EAEC
    classDef accent fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
    classDef out fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
```

**Leakage as a hard failure.** Every backtest window is built by a single shared
function that splits the data at a cutoff and then asserts that the latest training
match falls strictly before the first holdout match. If that assertion ever fails,
the run raises and the test suite goes red. It is not possible to score a window
whose training data overlaps its holdout.

**The same-window invariant.** A reviewer pointed out that a low score is exactly
when a careful engineer audits for hidden leakage rather than celebrating. That
audit became a permanent gate. The data carried two benign duplicate fixtures, and
rather than gate the proxy of "no duplicate rows," the suite gates the real
concern: no match, identified by date and the two teams, may appear in both the
training set and the holdout set of the same window. Because a duplicate shares an
exact date and the split is purely by date, both copies always land on the same
side, which the gate verifies across every window.

**Point-in-time feature construction.** The hardest leakage risk in the whole
project was the gradient-boosting experiment described in section 6, whose strongest
feature is a Dixon-Coles ability estimate. The naive way to build that feature, a
single model fit over the whole window read back onto every training row, lets each
row see a strength partly shaped by its own result. Instead the ability is built
from a grid of refits, where each refit uses only data strictly before its grid
date, and a match takes the abilities from the latest grid date at or before it. A
test proves the guarantee directly: tamper with a match's own result and every
later result, and the feature for that match does not change.

These checks live in the normal `pytest` suite, alongside schema validation,
ratings sanity, goal-model correctness, simulator tiebreaker logic, determinism,
and probability validity. The suite is the contract.

---

## 5. Results, honestly framed

The model is evaluated with the Ranked Probability Score, the appropriate metric
for ordered three-outcome football predictions, on a rolling-origin walk-forward
backtest. The most recent window is anchored exactly on the frozen single-window
calibration, which it reproduces to the digit, and seven further non-overlapping
windows of 150 matches each walk backward through time. Non-overlap is deliberate:
overlapping windows share matches and would fake a tighter spread than the data
supports.

```mermaid
flowchart TB
    TL["Played international matches, 1872 to 2026-06-10"]:::accent
    W1["Window 1, holdout 2026-03-31 to 2026-06-10<br/>anchor, reproduces the frozen 0.1611"]:::layer
    W2["Window 2, holdout 2025-12-24 to 2026-03-30"]:::layer
    W3["Window 3, holdout 2025-11-15 to 2025-12-23"]:::layer
    DOTS["windows 4 to 7, stepping backward"]:::layer
    W8["Window 8, holdout 2024-12-26 to 2025-03-24<br/>still trains on 48,082 matches"]:::layer
    AGG["Aggregate over 8 windows<br/>mean RPS 0.1575, std 0.0140 (the noise floor)"]:::out

    TL --> W1 --> W2 --> W3 --> DOTS --> W8 --> AGG

    classDef layer fill:#0A1A1F,stroke:#C9A84C,stroke-width:2px,color:#E8EAEC
    classDef accent fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
    classDef out fill:#0A1A1F,stroke:#00D4FF,stroke-width:2px,color:#E8EAEC
```

Each window trains on everything strictly before its own holdout, scores its 150
held-out matches, and recomputes its own base-rate baseline from its own training
data. The next window's cutoff is the previous window's training maximum, so the
holdouts are contiguous and share no matches. Even the oldest window trains on more
than 48,000 matches, so every window has a deep history behind it.

**The result.** Mean RPS **0.1575**, standard deviation **0.0140**, minimum 0.1336,
maximum 0.1764, against a base-rate baseline that averages **0.2292** over the same
windows. The model beat its own baseline in every one of the eight windows. The
single-window anchor, kept frozen and never overwritten, reads **0.1611** against a
**0.2135** baseline.

That standard deviation of 0.0140 is treated as the noise floor for the whole
project. No later change counts as a real improvement unless it moves the aggregate
mean by more than this spread. It is the bar every experiment in the next section
had to clear.

**A caveat against overclaiming.** International football is an easier distribution
to predict than club football. National teams play less often, the talent gap
between the strongest and weakest sides is wider than in a top domestic league, and
results are correspondingly more predictable. An RPS in this range is good, but it
is good on a friendly distribution, and it should not be read as a club-level
result. The honest framing is that the system is well calibrated for what it
models, not that it has solved football.

---

## 6. What did not work

These are kept in the record as a matter of rigor, not apology. A verification-first
project that only reported its successes would be contradicting its own thesis. Two
ideas were built properly, measured against the 0.0140 noise floor, and rejected.

**Friendly-match downweighting.** Friendlies are about 37 percent of the training
data and are low-stakes, often played with experimental lineups, so the hypothesis
was that downweighting them would sharpen the fit. It was implemented as a tunable
weight that multiplies onto the existing time decay, defaulted off so the headline
numbers could not move silently, and measured. On the single calibration window it
moved the score from 0.1611 to 0.1617, a change of plus 0.0006, well inside the
0.0140 noise floor and in the wrong direction. It did not help. The default stays
off, and both numbers live in the record.

**The hybrid ensemble.** This was the academically strongest idea in the project: a
gradient-boosting model in the Groll and Zeileis style, taking ability estimates
plus covariates as features, the kind of approach that wins forecasting papers. It
was built in full, with four leakage-safe point-in-time features, the
squad-adjusted Elo difference, the Dixon-Coles ability difference, squad value, and
rest-days, and evaluated on the exact same eight walk-forward windows as the goal
model.

It did not beat Dixon-Coles. The ensemble scored a mean of **0.1618** against the
goal model's **0.1575**, a difference of plus 0.0044, comfortably inside the 0.0140
noise floor and slightly worse on the point estimate. The permutation feature
importances explain why:

| Feature | Importance |
| --- | --- |
| Elo difference | +0.2182 |
| Dixon-Coles ability difference | +0.1452 |
| Squad value | +0.0025 |
| Rest-days | -0.0038 |

Almost all of the signal is in the two strength features, which encode the same
information the goal model already uses. Squad value and rest-days are close to
noise. A more sophisticated technique built from the same strength signal lands on
top of the simple model, not beyond it. The lesson is the standard one in applied
machine learning and worth restating: features drive accuracy, not technique. The
ensemble and its measurement stay in the repository as a documented, gated
experiment.

---

## 7. Live forecast

> **Status: pending.** The definitive live forecast will be the post-group-stage
> run, once all 72 group matches are complete. The mid-tournament numbers below
> exist and are reproducible, but they are not the final word and are not presented
> as such. This section is structured to receive the definitive run.

The live pipeline is built and works. It takes a clean cutoff, anchors on the real
results played so far by fixing those group fixtures and simulating only the
remainder, and writes a separate dated artifact so each forecast is a frozen
record. A leakage proof is enforced at the live cutoff exactly as in the backtest:
the latest training match must fall strictly before the cutoff boundary.

A mid-group-stage run at a cutoff of 2026-06-22, with 44 real group results
anchored, is on disk and can be referenced. It is provisional. The post-group-stage
forecast will replace this paragraph with the definitive numbers.

There are three frozen forecast artifacts worth distinguishing precisely, since the
structure of the simulator changed during development:

1. The original forecast on a **placeholder group structure**, from before the real
   2026 draw was installed. Superseded, kept only as history.
2. The **pre-tournament forecast on the real structure**, at a cutoff of
   2026-06-10 with no tournament results yet. This is the honest pre-tournament
   baseline going forward.
3. The **live forecast on the real structure**, at a cutoff of 2026-06-22, anchored
   on the real results so far. Provisional, to be superseded by the post-group-stage
   run.

When the numbers are compared, the meaningful comparison is between the second and
the third, since both use the real structure and the only difference between them
is the ingested results. Comparing against the first would conflate the structure
change with the effect of real data and would be misleading.

---

## 8. Math versus LLM comparison

> **Status: planned, not yet built.** This section describes a phase that has not
> been implemented. Results will be added once it exists.

The intended capstone experiment is a head-to-head between this mathematical system
and a large language model, scored honestly with the same metric.

The design is deliberately simple. The mathematical pipeline produces a frozen,
timestamped prediction. The language model is then given the same situation and
asked to produce its own independent prediction, forward-only, meaning it forecasts
upcoming matches it cannot have seen the results of. Both predictions are scored
with the Ranked Probability Score against what actually happens.

The forward-only constraint is the entire point. It prevents the language model
from quietly retrieving results that have already occurred and reporting them as a
forecast, which would make the comparison meaningless. By fixing both predictions
before the matches are played and scoring them afterward, the comparison stays fair
and the result, whichever way it falls, is verifiable.

Results will be added here once the phase is built.

---

## 9. Limitations

Stated plainly, because a verification-first project should be the first to name its
own weaknesses.

- **Squad values are approximate.** They come from a single committed, dated
  snapshot of national-team market values, gathered as reasonable present-day
  estimates rather than official figures. They are a controlled, versioned input
  that can be refined without touching model code, but they are not exact, and as a
  feature they proved close to noise in any case.
- **International data is an easier distribution than club football.** The accuracy
  numbers are good for what they model and should not be read as club-level
  performance, as noted in the results section.
- **The model captures current strength, not tournament-specific dynamics.** It is
  trained on all international football, of which actual World Cup finals matches are
  only about 2 percent. It knows how good teams are; it does not specifically model
  the psychology and tactics of a finals knockout.
- **No bookmaker odds, by deliberate choice.** This is a limitation on raw accuracy
  and a feature of the thesis. The system trades some achievable accuracy for
  independence, so that it can be benchmarked against the market rather than
  derived from it.
- **The fair-play tiebreaker falls back.** The model cannot predict cards, so when a
  simulated tie reaches the fair-play step it falls back to a FIFA-ranking proxy and
  then to a random draw. This step is engaged often, in roughly 59 percent of live
  simulations, almost entirely at the margin where the eight best third-placed teams
  are separated and twelve teams cluster on points and goals. It rarely changes the
  title picture, since it resolves by relative strength, but it is engaged
  frequently and is reported honestly rather than hidden.

---

## 10. Reproducibility

The system is reproducible to the digit. The same seed and the same immutable data
snapshot produce the same output, and every run writes a provenance log recording
the configuration, the cutoff date, the seed, the git commit, and the UTC time.

**Run it.**

```
uv sync
uv run wc-predict          # pre-tournament forecast at the config cutoff
uv run wc-calibrate        # single-window RPS backtest, the 0.1611 anchor
uv run wc-walk-backtest    # the 8-window walk-forward, the headline 0.1575
uv run wc-ablate           # the with-and-without ablation harness
uv run wc-ensemble         # the rejected gradient-boosting experiment
uv run wc-audit            # data-integrity and overconfidence checks
uv run wc-forecast-live    # the live dated forecast, anchored on real results
```

**The gates.** The verification claims are not prose, they are tests. The suite
spans schema validation, the no-leakage guard, ratings sanity, goal-model
correctness, the squad adjustment, the simulator tiebreaker and bracket logic, the
calibration scorer, the ablation reproduction, the walk-forward invariants, the
data audit, and the ensemble leakage safety.

```
uv run ruff check .
uv run pytest -q
```

**The artifacts.** Every model and evaluation run writes a versioned JSON artifact
with provenance to `outputs/`. Presentation reads those artifacts and never
recomputes. Raw data snapshots and run outputs are immutable and are not committed
to version control; the directories are kept, the contents are local.

**License.** Proprietary, all rights reserved. This repository is public for
evaluation and reference only and is not licensed for reuse. See `LICENSE`.

---

Built by The Architect AI.
