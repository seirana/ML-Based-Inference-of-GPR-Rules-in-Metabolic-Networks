# Reports and generated artifacts

This directory contains both historical project material and run-generated artifacts.

## Generated outputs

Current pipeline runs write to `metrics/`, `models/`, `candidates/`, and `run_metadata.json`.

New generated outputs are ignored by Git by default.

## Historical material

`Report.txt` and the pre-existing Word report under `metrics/` were created before the current quality upgrade.

They are retained as historical project artifacts. Some methodological descriptions may refer to the earlier implementation, in which gene fingerprints were built globally. The maintained code and the current root documentation define the leakage-aware evaluation methodology.

For current methodology, use `../README.md`, `../EVALUATION.md`, and `../MODEL_CARD.md`.
