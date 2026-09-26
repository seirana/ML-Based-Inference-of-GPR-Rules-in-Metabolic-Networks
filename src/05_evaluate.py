#!/usr/bin/env python3
"""Evaluate a saved model on the persisted held-out reaction split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gpr_ml import (
    FEATURE_COLS,
    classification_metrics,
    ensure_dir,
    indices_from_split,
    load_split,
    save_json,
    score_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GPR-ML model on "
            "the stored reaction-wise test split."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path("reports/metrics"),
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
    _, test_idx = indices_from_split(
        frame,
        train_reactions,
        test_reactions,
    )

    features = frame[
        FEATURE_COLS
    ].to_numpy(dtype=np.float32)
    model = joblib.load(
        args.model_path
    )
    scores = score_model(
        model,
        features[test_idx],
    )

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
            "model_path": str(
                args.model_path
            ),
            "feature_cols": FEATURE_COLS,
            "split_source": (
                "data/processed/"
                "split_reactions.json"
            ),
            "seed": int(args.seed),
        }
    )

    model_name = (
        args.model_path.stem
    )
    output_path = (
        outdir
        / f"eval_{model_name}.json"
    )
    save_json(
        metrics,
        output_path,
    )

    print(metrics)
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
