#!/usr/bin/env python3
"""Train the logistic-regression baseline on the stored reaction split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gpr_ml import (
    FEATURE_COLS,
    classification_metrics,
    ensure_dir,
    indices_from_split,
    load_split,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a standardized logistic-regression "
            "baseline for GPR association prediction."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path("reports/metrics"),
        type=Path,
    )
    parser.add_argument(
        "--model_out",
        default=Path(
            "reports/models/logreg.joblib"
        ),
        type=Path,
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
    )
    parser.add_argument(
        "--bootstrap_reps",
        default=500,
        type=int,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError(
            "bootstrap_reps must be non-negative"
        )

    ensure_dir(args.model_out.parent)
    outdir = ensure_dir(args.outdir)

    frame = pd.read_parquet(
        args.procdir / "features.parquet"
    )
    train_reactions, test_reactions = (
        load_split(
            args.procdir
            / "split_reactions.json"
        )
    )
    train_idx, test_idx = indices_from_split(
        frame,
        train_reactions,
        test_reactions,
    )

    x = frame[FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    y = frame["label"].to_numpy(
        dtype=int
    )

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training split must contain both classes."
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
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(
        x_train,
        y_train,
    )

    scores = model.predict_proba(
        x_test
    )[:, 1]
    test_frame = frame.iloc[
        test_idx
    ].copy()

    metrics = classification_metrics(
        test_frame,
        scores,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    metrics.update(
        {
            "model": "logistic_regression",
            "feature_cols": FEATURE_COLS,
            "seed": int(args.seed),
            "split_source": (
                "data/processed/"
                "split_reactions.json"
            ),
            "scaling": "StandardScaler",
        }
    )

    save_json(
        metrics,
        outdir / "logreg_metrics.json",
    )
    joblib.dump(
        model,
        args.model_out,
    )

    scaler = model.named_steps["scale"]
    classifier = model.named_steps["model"]
    save_json(
        {
            "feature_cols": FEATURE_COLS,
            "standardized_coefficients": {
                name: float(value)
                for name, value
                in zip(
                    FEATURE_COLS,
                    classifier.coef_[0],
                    strict=True,
                )
            },
            "intercept": float(
                classifier.intercept_[0]
            ),
            "scaler_mean": {
                name: float(value)
                for name, value
                in zip(
                    FEATURE_COLS,
                    scaler.mean_,
                    strict=True,
                )
            },
            "scaler_scale": {
                name: float(value)
                for name, value
                in zip(
                    FEATURE_COLS,
                    scaler.scale_,
                    strict=True,
                )
            },
        },
        outdir
        / "logreg_coefficients.json",
    )

    print(metrics)
    print(
        f"Saved model: {args.model_out}"
    )


if __name__ == "__main__":
    main()
