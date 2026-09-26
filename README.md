# GPR-ML: Machine Learning for Gene–Reaction Association Inference

GPR-ML is an interpretable machine-learning research pipeline for ranking candidate gene–reaction associations in genome-scale metabolic models.

The current implementation is designed around a central evaluation question:

> Given a reaction whose curated gene associations are held out, can network-derived features rank known gene associations above sampled alternatives without using that held-out reaction to construct the gene fingerprint?

The repository includes a simple structural heuristic, logistic regression, XGBoost, reaction-level cross-validation, grouped bootstrap confidence intervals, automated tests, CI, and a Docker workflow.

> **Scope:** GPR-ML ranks candidate gene–reaction associations. It does not infer full Boolean GPR logic such as AND/OR enzyme-complex rules, and a high model score is not experimental evidence that a gene catalyzes a reaction.

## Why leakage control matters

A reaction-wise split alone is not sufficient if gene fingerprints are built from **all** reactions before the split. In that case, a held-out reaction can still contribute its metabolites or subsystem to the fingerprint of its true gene, leaking information from the test label into the features.

The quality-upgraded pipeline avoids that problem:

```text
SBML reactions
    |
    v
reaction-wise split
    |
    +--------------------------+
    |                          |
    v                          v
training reactions         held-out reactions
    |
    v
reference graph + gene fingerprints
    |
    +---- training pair feature:
    |     exclude the reaction currently being scored
    |
    +---- held-out pair feature:
          held-out reaction is never in the reference graph
```

Candidate pools and gene fingerprints are derived from the **training reactions only**. For a training positive pair, the reaction currently being scored is removed from that gene's fingerprint. For a test reaction, the reaction was never part of the reference graph in the first place.

See [EVALUATION.md](EVALUATION.md) for the exact evaluation protocol and limitations.

## Models and baselines

| Method | Role |
|---|---|
| Metabolite-Jaccard heuristic | non-ML structural baseline |
| Logistic Regression | interpretable standardized ML baseline |
| XGBoost | non-linear tree-based model |

The logistic model uses `StandardScaler` before fitting. XGBoost uses a validation split drawn only from the training reactions for early stopping.

## Features

The maintained feature set is intentionally small and interpretable:

| Feature | Meaning |
|---|---|
| `jacc_mets` | Jaccard similarity between reaction metabolites and the gene fingerprint |
| `overlap_mets` | number of shared metabolites |
| `n_mets_rxn` | number of metabolites in the reaction |
| `n_mets_gene_fp` | size of the gene metabolite fingerprint |
| `subsystem_match` | whether the reaction subsystem appears in the gene fingerprint |
| `n_subsys_gene_fp` | number of subsystems in the gene fingerprint |

The gene fingerprint is reconstructed from eligible training reactions rather than stored globally before splitting.

## Repository structure

```text
.
├── src/
│   ├── gpr_ml/
│   │   ├── core.py
│   │   └── pipeline.py
│   ├── 00_parse_model.py
│   ├── 00b_make_split.py
│   ├── 01_build_pairs.py
│   ├── 02_features.py
│   ├── 03a_evaluate_heuristic.py
│   ├── 03_train_logreg.py
│   ├── 04_train_xgb.py
│   ├── 05_evaluate.py
│   ├── 06_rank_candidates.py
│   ├── 07_group_cv_logreg.py
│   └── utils.py
├── tests/
├── data/
├── reports/
├── run_pipeline.py
├── run_pipeline.sh
├── EVALUATION.md
├── MODEL_CARD.md
├── pyproject.toml
├── requirements.txt
└── Dockerfile
```

`src/utils.py` is retained only as a compatibility layer for historical imports. The maintained reusable implementation lives under `src/gpr_ml/`.

## Installation

```bash
git clone https://github.com/seirana/ML-Based-Inference-of-GPR-Rules-in-Metabolic-Networks.git
cd ML-Based-Inference-of-GPR-Rules-in-Metabolic-Networks

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Input model

The default example workflow expects an SBML model at:

```text
data/raw/iJO1366.xml
```

The model itself is not committed to this repository. Record its source, version/release, download date, and checksum for reproducible experiments.

See [data/README.md](data/README.md).

## Run the complete pipeline

After placing the SBML file under `data/raw/`:

```bash
gprml-pipeline \
  --model data/raw/iJO1366.xml \
  --seed 13 \
  --test_size 0.2 \
  --neg_per_pos 10 \
  --bootstrap_reps 500
```

The older shell command remains available as a thin compatibility wrapper:

```bash
./run_pipeline.sh
```

The portable Python entry point is preferred.

## Pipeline stages

```text
0  parse SBML
1  create persisted reaction split
2  build positive + sampled negative pairs
3  build leakage-aware features
4  evaluate non-ML Jaccard baseline
5  train/evaluate Logistic Regression
6  train/evaluate XGBoost
7  evaluate saved XGBoost model
8  rank candidate genes
```

The split is created once and reused downstream instead of being silently regenerated by each model script.

## Evaluation outputs

Typical outputs include:

```text
data/processed/
  reactions.parquet
  genes.parquet
  split_reactions.json
  pairs.parquet
  pair_build_metadata.json
  features.parquet
  feature_metadata.json

reports/metrics/
  heuristic_jaccard_metrics.json
  logreg_metrics.json
  logreg_coefficients.json
  xgb_metrics.json
  xgb_feature_importances.json
  eval_xgb.json

reports/models/
  logreg.joblib
  xgb.joblib

reports/candidates/
  top_candidates_xgb.csv

reports/run_metadata.json
```

Generated model/metric/candidate outputs are ignored by Git by default. The pipeline also records the Git commit SHA when available and a SHA-256 checksum of the input SBML file in `reports/run_metadata.json`.

## Metrics

The pipeline reports:

- average precision / PR-AUC;
- ROC-AUC when both classes are present;
- reaction-level hit@5, hit@10, and hit@20;
- grouped bootstrap 95% confidence intervals;
- held-out positive links excluded because the gene was never observed in training.

That last quantity is important. The standard pairwise evaluation is conditional on the **training gene vocabulary**. An unseen held-out gene cannot be ranked from a fingerprint that does not exist, so it is excluded and explicitly counted rather than silently treated as a normal candidate.

## Leakage-aware cross-validation

A separate fold-rebuilding evaluation is available:

```bash
python src/07_group_cv_logreg.py \
  --procdir data/processed \
  --n_splits 5 \
  --neg_per_pos 10 \
  --seed 13
```

Each fold independently rebuilds the training reference graph, sampled pairs, leave-one-reaction-out fingerprints, features, and logistic model. This is intentionally more expensive than cross-validating one precomputed global feature table, because the latter would reintroduce leakage.

## Candidate ranking

After training:

```bash
python src/06_rank_candidates.py \
  --procdir data/processed \
  --model_path reports/models/xgb.joblib \
  --reaction_scope all \
  --topk 10
```

Candidate scores are prioritization signals for curation or follow-up. They are not calibrated biochemical truth and do not prove that a missing GPR association is correct.

## Testing

```bash
python -m pytest
python -m ruff check src tests run_pipeline.py
```

Tests specifically cover disjoint persisted reaction splits, prevention of own-reaction fingerprint leakage, exclusion/reporting of test genes unseen during training, deterministic negative sampling, feature-table integrity, and reaction-grouped bootstrap evaluation.

GitHub Actions runs tests and linting on Python 3.10, 3.11, and 3.12 and builds the Docker image.

## Docker

```bash
docker build -t gpr-ml .
```

Run with input/output directories mounted from the host:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  gpr-ml \
  --model data/raw/iJO1366.xml
```

## Scientific limitations

The current project uses only information derived from the metabolic model. It does not use sequence similarity, protein domains, orthology, expression, proteomics, enzyme databases, or experimental validation.

Metrics are also influenced by the negative-sampling strategy. Performance against sampled negatives should not be interpreted as performance against the complete universe of biologically plausible genes.

See [MODEL_CARD.md](MODEL_CARD.md) and [EVALUATION.md](EVALUATION.md).

## Historical reports

The repository contains older report artifacts produced before this quality upgrade. They are retained for project history but may describe the earlier feature-generation/evaluation implementation. The maintained code and current documentation define the upgraded methodology.

## License

No explicit license file is currently included. Public repository visibility alone does not grant reuse rights. Add a license if redistribution or reuse is intended.
