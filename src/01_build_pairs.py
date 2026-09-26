#!/usr/bin/env python3
"""Build positive and sampled negative reaction-gene pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpr_ml import (
    build_pairs,
    ensure_dir,
    load_split,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reaction-gene pairs using only "
            "training reactions as the candidate-reference graph."
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
        help=(
            "Reaction split JSON. Defaults to "
            "<procdir>/split_reactions.json."
        ),
    )
    parser.add_argument(
        "--neg_per_pos",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
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
    train_reactions, test_reactions = (
        load_split(split_path)
    )

    reactions_df = pd.read_parquet(
        args.procdir / "reactions.parquet"
    )
    pairs_df, metadata = build_pairs(
        reactions_df,
        train_reactions=train_reactions,
        test_reactions=test_reactions,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )

    pairs_path = outdir / "pairs.parquet"
    pairs_df.to_parquet(
        pairs_path,
        index=False,
    )
    save_json(
        metadata,
        outdir
        / "pair_build_metadata.json",
    )

    print(
        f"Saved: {pairs_path} "
        f"({len(pairs_df)} pairs, "
        f"pos={int(pairs_df['label'].sum())})"
    )
    print(
        "Held-out positive links excluded because "
        "their genes were unseen in training: "
        f"{metadata['n_test_positive_links_excluded']}"
    )


if __name__ == "__main__":
    main()
