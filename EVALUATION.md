# Evaluation protocol

This document defines the evaluation design used by the maintained GPR-ML pipeline.

## Prediction target

The supervised target is a **gene–reaction association**:

```text
(reaction, gene) -> associated / not associated
```

The project does not currently predict the Boolean structure of full GPR rules. In particular, it does not distinguish whether multiple genes should be joined by `AND` versus `OR`.

## Primary leakage risk

The main feature family uses a gene's metabolic fingerprint: metabolites and subsystems seen in reactions already associated with that gene.

If fingerprints are built from the full model before a train/test split, then a held-out reaction contributes information to the fingerprint of its true gene. A reaction-wise row split would then look clean while the features still contain held-out label information.

The upgraded pipeline prevents that.

## Split order

The workflow is:

```text
parse reactions
    |
    v
persist reaction split
    |
    v
construct training-only reference graph
    |
    +--> build candidate pairs
    |
    +--> build gene fingerprints
    |
    v
train/evaluate
```

The same `split_reactions.json` is reused by pair construction, feature construction, training, evaluation, and candidate ranking.

## Training-pair feature rule

For a training pair involving reaction `R` and gene `G`, the fingerprint of `G` is computed from training reactions associated with `G` **excluding `R` itself**.

This leave-one-reaction-out rule prevents an easy positive label from directly injecting the scored reaction's metabolites/subsystem into its gene fingerprint.

## Held-out-pair feature rule

A held-out reaction is never part of the training reference graph. Its gene features are therefore built only from other reactions in the training set.

The reaction's own metabolites and subsystem remain available as reaction-side input features. Those are assumed to be known characteristics of the reaction being annotated.

## Gene vocabulary

The current method requires a gene to have some training-side context from which a fingerprint can be constructed.

If a test reaction contains a positive gene that never appears in any training reaction:

- that positive link is excluded from the standard pairwise ranking benchmark;
- the gene is recorded in `pair_build_metadata.json`;
- `n_test_positive_links_excluded` reports how many held-out positive links were not evaluable.

Therefore the standard pairwise benchmark measures performance **conditional on genes observed in the training reaction network**.

It does not measure zero-shot discovery of entirely unseen genes.

## Negative sampling

For each reaction, the pipeline first builds a biologically motivated hard-negative pool using genes associated with the same subsystem in training reactions and genes associated with reactions sharing metabolites.

If that pool is too small, it is filled from the training gene vocabulary.

Known curated positives for the scored reaction are removed from the negative pool.

The candidate set is sorted before random sampling so a fixed NumPy seed is reproducible independently of Python set/hash iteration order.

## Metrics

Point metrics include average precision, ROC-AUC when both classes are present, and reaction-level hit@5, hit@10, and hit@20.

Because negatives are sampled, PR-AUC and ROC-AUC describe the constructed benchmark, not an exhaustive all-gene search space.

## Grouped bootstrap confidence intervals

The test evaluator resamples **reactions**, not individual pair rows.

This preserves within-reaction dependence more faithfully than treating every reaction–gene pair as statistically independent.

The pipeline reports percentile 95% bootstrap intervals for the available metrics. The bootstrap is intended as uncertainty characterization for the constructed held-out benchmark, not as a biological confidence interval.

## XGBoost validation

The persisted held-out test set is not used for early stopping.

XGBoost creates a second grouped split **inside the training reactions**. The validation set is used for early stopping; the persisted test reactions remain untouched until final evaluation.

## Cross-validation

`src/07_group_cv_logreg.py` performs reaction-level K-fold evaluation.

Critically, it rebuilds pairs and fingerprints independently inside every fold. Reusing one global feature table across folds would invalidate the leakage guarantees.

## Baseline

The non-ML baseline scores each pair with metabolite Jaccard similarity only.

This provides a simple reference for asking whether the learned models add value beyond the strongest directly encoded structural-overlap feature.

## Candidate ranking versus benchmark evaluation

Candidate ranking and held-out benchmarking answer different questions.

**Held-out benchmark:** can known associations for unseen reactions be recovered using training-derived gene context?

**Candidate ranking:** which currently uncurated genes should be prioritized for a reaction under the trained model?

A candidate ranking is a hypothesis-generation output. It should be followed by external biological evidence and expert curation.

## Remaining limitations

The benchmark does not currently include external independent metabolic models, sequence/domain similarity baselines, orthology-based baselines, complete all-gene negative sets, experimental validation, calibration analysis, organism-to-organism transfer, or Boolean AND/OR GPR logic inference.

Those are appropriate future extensions rather than claims supported by the current repository.
