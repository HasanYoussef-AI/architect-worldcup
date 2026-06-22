"""Layer 1, data integrity.

Owns ingestion of match and team data into dated, immutable raw snapshots, and
the leakage guard that ensures no information dated after as_of_date enters the
model. Stub for Phase 0.
"""
