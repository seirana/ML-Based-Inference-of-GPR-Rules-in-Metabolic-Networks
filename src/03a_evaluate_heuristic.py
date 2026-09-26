#!/usr/bin/env python3
"""Evaluate a simple metabolite-overlap heuristic on the held-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpr_ml import (
    classification_metrics,
    ensure_dir,
    indices_from_split,
    load_split,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate jaccard metabolite similarity "
            "as a non-ML structural baseline."
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
    frame = pd.read_parquet(
        args.procdir / "features.parquet"
    )
    train_reactions, test_reactions = load_split(
        args.procdir
        / "split_reactions.json"
    )
    _, test_idx = indices_from_split(
        frame,
        train_reactions,
        test_reactions,
    )
    test_frame = frame.iloc[
        test_idx
    ].copy()

    metrics = classification_metrics(
        test_frame,
        test_frame[
            "jacc_mets"
        ].to_numpy(dtype=float),
        bootstrap_reps=(
            args.bootstrap_reps
        ),
        seed=args.seed,
    )
    metrics.update(
        {
            "model": (
                "heuristic_jaccard_baseline"
            ),
            "score_feature": "jacc_mets",
            "seed": int(args.seed),
        }
    )

    outdir = ensure_dir(args.outdir)
    save_json(
        metrics,
        outdir
        / "heuristic_jaccard_metrics.json",
    )
    print(metrics)


if __name__ == "__main__":
    main()
