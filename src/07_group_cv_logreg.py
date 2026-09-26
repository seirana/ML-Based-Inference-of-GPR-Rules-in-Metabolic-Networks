#!/usr/bin/env python3
"""Leakage-aware grouped cross-validation for the logistic baseline.

Each fold rebuilds candidate pairs and gene fingerprints from the fold's training
reactions only. This is slower than reusing one global feature table, but it avoids
label leakage from held-out reactions into gene fingerprints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpr_ml import (
    FEATURE_COLS,
    build_feature_table,
    build_pairs,
    classification_metrics,
    ensure_dir,
    indices_from_split,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-aware reaction-level "
            "cross-validation for logistic regression."
        )
    )
    parser.add_argument(
        "--procdir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/metrics"),
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )
    return parser


def mean_std(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "std": float(
            finite.std(ddof=1)
            if len(finite) > 1
            else 0.0
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2"
        )

    reactions = pd.read_parquet(
        args.procdir / "reactions.parquet"
    )
    if len(reactions) < args.n_splits:
        raise ValueError(
            "n_splits cannot exceed the number "
            "of reactions."
        )

    splitter = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    folds: list[dict[str, object]] = []

    for fold_index, (
        train_reaction_idx,
        test_reaction_idx,
    ) in enumerate(
        splitter.split(reactions),
        start=1,
    ):
        train_reactions = (
            reactions.iloc[
                train_reaction_idx
            ]["reaction_id"]
            .astype(str)
            .tolist()
        )
        test_reactions = (
            reactions.iloc[
                test_reaction_idx
            ]["reaction_id"]
            .astype(str)
            .tolist()
        )

        pairs, pair_metadata = build_pairs(
            reactions,
            train_reactions=train_reactions,
            test_reactions=test_reactions,
            neg_per_pos=args.neg_per_pos,
            seed=args.seed + fold_index,
        )
        features = build_feature_table(
            pairs,
            reactions,
            train_reactions=train_reactions,
        )
        train_idx, test_idx = (
            indices_from_split(
                features,
                train_reactions,
                test_reactions,
            )
        )

        x = features[
            FEATURE_COLS
        ].to_numpy(dtype=np.float32)
        y = features[
            "label"
        ].to_numpy(dtype=int)

        if (
            len(np.unique(y[train_idx])) < 2
            or len(np.unique(y[test_idx])) < 2
        ):
            raise ValueError(
                f"Fold {fold_index} lacks both classes."
            )

        model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=(
                            args.seed
                            + fold_index
                        ),
                    ),
                ),
            ]
        )
        model.fit(
            x[train_idx],
            y[train_idx],
        )
        scores = model.predict_proba(
            x[test_idx]
        )[:, 1]

        metrics = classification_metrics(
            features.iloc[
                test_idx
            ],
            scores,
        )
        folds.append(
            {
                "fold": fold_index,
                "metrics": metrics,
                "pair_metadata": (
                    pair_metadata
                ),
            }
        )

    metric_names = [
        "average_precision",
        "roc_auc",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
    ]
    aggregate = {
        metric: mean_std(
            [
                float(
                    fold["metrics"][
                        metric
                    ]
                )
                for fold in folds
            ]
        )
        for metric in metric_names
    }

    result = {
        "model": "logistic_regression",
        "evaluation": (
            "reaction-level cross-validation "
            "with fold-specific pair and "
            "fingerprint reconstruction"
        ),
        "n_splits": args.n_splits,
        "neg_per_pos": args.neg_per_pos,
        "seed": args.seed,
        "feature_cols": FEATURE_COLS,
        "aggregate": aggregate,
        "folds": folds,
    }

    outdir = ensure_dir(args.outdir)
    output_path = (
        outdir / "logreg_group_cv.json"
    )
    save_json(result, output_path)
    print(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
