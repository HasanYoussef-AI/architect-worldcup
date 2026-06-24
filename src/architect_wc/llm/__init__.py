"""Phase 8, the math versus LLM analyst comparison.

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

This module currently holds the offline skeleton: the frozen weights, the
round-aware quarantine gate, the analytic Prediction A math, the RPS scoring
extension, schema validation, and the offline plumbing smoke test. The live
two-phase Anthropic calls (research with web search, then prediction with no
tools) are wired in a later step, after the skeleton and the smoke test are green.
"""

from __future__ import annotations
