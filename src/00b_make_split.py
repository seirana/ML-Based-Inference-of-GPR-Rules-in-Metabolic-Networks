#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir, make_reaction_split, save_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--out", default="data/processed/split_reactions.json", type=str)
    ap.add_argument("--seed", default=13, type=int)
    ap.add_argument("--test_size", default=0.2, type=float)
    args = ap.parse_args()

    procdir = Path(args.procdir)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    reactions_df = pd.read_parquet(procdir / "reactions.parquet")
    train_r, test_r = make_reaction_split(reactions_df, seed=args.seed, test_size=args.test_size)
    save_split(train_r, test_r, out_path)

    print(f"Saved split to: {out_path}")
    print(f"Train reactions: {len(train_r)}")
    print(f"Test reactions:  {len(test_r)}")


if __name__ == "__main__":
    main()
