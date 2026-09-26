# Data

Raw metabolic models and generated processed tables are intentionally separated.

## Raw input

By default the pipeline expects:

```text
data/raw/iJO1366.xml
```

The SBML file is not distributed by this repository.

For a reproducible experiment, record the model/database source, model identifier, release/version, retrieval date, file checksum, and any modifications made before running GPR-ML.

## Processed tables

The pipeline generates files under `data/processed/`.

### reactions.parquet

One row per reaction with reaction ID/name, subsystem, original GPR rule text, curated gene IDs, metabolite IDs, and counts.

### genes.parquet

A gene inventory and curated reaction count.

It intentionally does **not** store full-model metabolic fingerprints. Fingerprints are reconstructed later from the eligible training reference reactions to prevent held-out leakage.

### split_reactions.json

The persisted reaction-wise train/test split used throughout the run.

### pairs.parquet

Positive and sampled negative reaction–gene pairs.

### pair_build_metadata.json

Includes candidate-reference scope and counts of held-out positive genes that were unseen during training.

### features.parquet

Leakage-aware engineered features for the pair table.

### feature_metadata.json

Records the feature list and fingerprint policy.

## Generated data policy

Processed tables are ignored by Git by default because they are run-specific. Preserve them with an archived experiment when they are needed for exact reproduction.
