"""The math versus LLM analyst comparison.

The math model produces Prediction A for a knockout round and freezes it. An LLM
produces Prediction B for the same round, blind to A. A third pass, Prediction C,
reads A and B and issues a reconciled call. A is never edited by anything. All
three are scored head to head with the same RPS the rest of the project uses.

This package is a consumer of the model spine, not part of it. The math layers
(ingest, ratings, squad, model, simulate) own Prediction A; this package adds the
LLM analyst, the leakage quarantine, the scoring extension, and the artifacts.

Reproducibility, stated honestly: Prediction A is bit-reproducible because it is
analytic (the Dixon-Coles three-way plus the Elo shootout lean, fixed seed and
dated data). Predictions B and C are not bit-reproducible; LLM generation is not
bit-stable and temperature is removed on Opus 4.8. The frozen, committed research
dossier is the real reproducibility anchor for B and C: it freezes their inputs,
and its commit before the round's first kickoff is the forward-only leakage proof.

Leakage-safety bias, a stated property: the quarantine gate prefers false
positives to false negatives. It will occasionally drop a legitimate forward-stakes
fact, for example a "team X may need to beat team Y to advance" line, because a real
leak can wear a conditional and a leakage guard must not trust modal verbs to tell a
hypothetical from a result. When the gate drops such a fact it is not hidden: the
research flow filters and records it, so the coverage manifest shows it as a raw hit
that did not become an admissible finding (raw greater than admissible), and the
per-match rationale is expected to state the drop. The run is never aborted by a
single flagged finding.

The package holds the frozen weights and anchor mapping, the round-aware
quarantine gate, the per-match research call with allow-listed web search, the
coverage manifest and its gate, the analytic Prediction A math, Prediction B and
the reconciler C, the single model-call boundary, the per-tie live orchestrator,
the RPS scoring extension, schema validation, and the offline plumbing smoke test.
"""

from __future__ import annotations
