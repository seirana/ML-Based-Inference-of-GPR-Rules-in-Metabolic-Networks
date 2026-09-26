#!/usr/bin/env python3
"""Create leakage-aware features for reaction-gene pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpr_ml import (
    FEATURE_COLS,
    build_feature_table,
    ensure_dir,
    load_split,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reaction-gene features from "
            "training-reaction reference information."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--split",
        default=None,
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = ensure_dir(args.outdir)
    split_path = (
        args.split
        if args.split is not None
        else args.procdir
        / "split_reactions.json"
    )

    train_reactions, _ = load_split(
        split_path
    )
    reactions_df = pd.read_parquet(
        args.procdir / "reactions.parquet"
    )
    pairs_df = pd.read_parquet(
        args.procdir / "pairs.parquet"
    )

    features_df = build_feature_table(
        pairs_df,
        reactions_df,
        train_reactions=train_reactions,
    )

    output_path = (
        outdir / "features.parquet"
    )
    features_df.to_parquet(
        output_path,
        index=False,
    )
    save_json(
        {
            "feature_cols": FEATURE_COLS,
            "reference_scope": (
                "training reactions only"
            ),
            "training_pair_policy": (
                "gene fingerprints exclude the "
                "reaction currently being scored"
            ),
            "n_rows": int(
                len(features_df)
            ),
        },
        outdir
        / "feature_metadata.json",
    )

    print(
        f"Saved: {output_path} "
        f"({len(features_df)} rows)"
    )


if __name__ == "__main__":
    main()
