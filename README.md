# architect-worldcup

Live World Cup 2026 bracket re-simulator with a verification-first architecture.

It runs from the command line on manual trigger. The model layer produces
structured JSON and nothing else; every downstream consumer reads that JSON.
Reproducibility is non-negotiable: a fixed seed, immutable dated raw snapshots,
and a provenance log on every run.

## Usage

```
uv sync
uv run wc-predict
```

This reads `config.yaml`, runs the pipeline, and writes a timestamped
predictions JSON to `outputs/predictions/` plus a run log to `outputs/logs/`.

## Methodology

Placeholder. To be written as the model layers are built.

## Calibration

Placeholder. To be written once the calibration gate is in place.

## License

Proprietary, all rights reserved. This repository is public for evaluation and reference only. See LICENSE.
