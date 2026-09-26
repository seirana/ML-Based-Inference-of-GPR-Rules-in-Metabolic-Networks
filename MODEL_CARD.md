# Model card

## System

**Name:** GPR-ML

**Task:** rank candidate gene–reaction associations in genome-scale metabolic models.

**Current example domain:** an *E. coli* genome-scale metabolic model such as iJO1366, supplied by the user as SBML.

## Intended use

The project is intended for research on metabolic-network link prediction, prioritizing candidate associations for manual curation, comparing interpretable structural features and classical ML models, and studying evaluation leakage in network-derived features.

## Not intended for

The project is not a replacement for experimental validation or expert metabolic-model curation.

A model score should not be treated as proof of enzyme activity, direct catalysis, protein-complex membership, gene essentiality, GPR Boolean logic, or physiological relevance in a particular condition.

## Inputs

The maintained pipeline derives features from the supplied metabolic model:

- reaction metabolites;
- reaction subsystem annotations;
- curated gene–reaction associations.

No omics or sequence information is used in the current version.

## Models

### Structural heuristic

Uses metabolite Jaccard similarity directly.

### Logistic Regression

Uses standardized interpretable features with class balancing.

### XGBoost

Uses the same feature set with non-linear tree interactions and train-only grouped validation for early stopping.

## Evaluation population

The primary benchmark consists of held-out reactions and genes that are already represented somewhere in the training reaction graph.

Held-out positive genes that are absent from the training gene vocabulary are counted and excluded from the standard pairwise benchmark because the current method cannot construct a training-derived fingerprint for them.

This limitation must be reported alongside performance metrics.

## Leakage controls

The maintained pipeline creates the reaction split before feature construction, derives candidate-reference indices from training reactions only, excludes the current reaction from training-pair gene fingerprints, reuses the same persisted split during evaluation, and rebuilds features independently for each cross-validation fold.

## Metrics

Reported metrics may include average precision, ROC-AUC, reaction-level hit@K, and grouped bootstrap 95% intervals.

Because negative genes are sampled rather than exhaustively enumerated, the classification metrics are benchmark-specific.

## Interpretability

Logistic standardized coefficients and XGBoost feature importances are exported.

These describe how the fitted model uses the engineered features. They are not causal biological explanations.

## Candidate outputs

`reports/candidates/` contains ranked candidate genes after training.

Candidate lists are hypotheses for downstream validation. Recommended follow-up evidence could include sequence/domain evidence, orthology, enzyme databases, literature, gene expression/proteomics, and expert review.

## Reproducibility

For any reported experiment, preserve the Git commit SHA, SBML model source/version and checksum, split JSON, random seed, negative-sampling ratio, model hyperparameters, dependency versions, generated metrics, and run metadata.

## Known limitations

- no zero-shot representation for genes unseen in training;
- no direct AND/OR GPR-rule inference;
- no explicit reaction directionality or stoichiometric coefficients in the current features;
- no external omics evidence;
- no independent-model external validation;
- no probability-calibration study;
- candidate evaluation depends on negative sampling;
- subsystem annotations can vary between metabolic models.
