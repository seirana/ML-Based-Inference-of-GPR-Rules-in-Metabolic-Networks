#!/usr/bin/env python3
"""Create and persist a reaction-wise train/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpr_ml import (
    ensure_dir,
    make_reaction_split,
    save_split,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible reaction-wise "
            "train/test split."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=Path(
            "data/processed/split_reactions.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
    )
    parser.add_argument(
        "--test_size",
        default=0.2,
        type=float,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out.parent)

    reactions_path = (
        args.procdir / "reactions.parquet"
    )
    reactions_df = pd.read_parquet(
        reactions_path
    )

    train_df, test_df = (
        make_reaction_split(
            reactions_df,
            seed=args.seed,
            test_size=args.test_size,
        )
    )
    save_split(
        train_df,
        test_df,
        args.out,
        seed=args.seed,
        test_size=args.test_size,
    )

    print(f"Saved split to: {args.out}")
    print(
        f"Train reactions: {len(train_df)}"
    )
    print(
        f"Test reactions:  {len(test_df)}"
    )


if __name__ == "__main__":
    main()
