#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

'''
Seed randomness (seed_everything and np.random.default_rng)
Load:
reactions.parquet (must include columns like reaction_id, genes, metabolites, subsystem)
genes.parquet (must include gene_id)
Build indices (subsys_to_genes, met_to_genes)
For each reaction r:
pos = set(r["genes"])
skip if no genes
Add positives: one row per gene in pos labeled 1
Sample negatives: neg_per_pos * len(pos) genes labeled 0
Create a dataframe, deduplicate (reaction_id, gene_id) pairs
Save to pairs.parquet
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir, seed_everything


def build_indices(reactions_df: pd.DataFrame) -> Dict:
    # Map subsystem -> genes in that subsystem (via reactions' curated genes)
    subsys_to_genes: Dict[str, Set[str]] = {}
    # Map metabolite -> genes in reactions containing that metabolite
    met_to_genes: Dict[str, Set[str]] = {}

    for _, row in reactions_df.iterrows():
        genes = set(row["genes"])
        subsys = row["subsystem"]
        if subsys and isinstance(subsys, str):
            subsys_to_genes.setdefault(subsys, set()).update(genes)
        for met in row["metabolites"]:
            met_to_genes.setdefault(met, set()).update(genes)

    return {"subsys_to_genes": subsys_to_genes, "met_to_genes": met_to_genes}


def sample_hard_negatives(
    reaction_row: pd.Series,
    all_genes: np.ndarray,
    subsys_to_genes: Dict[str, Set[str]],
    met_to_genes: Dict[str, Set[str]],
    positives: Set[str],
    n_needed: int,
    rng: np.random.Generator,
) -> List[str]:
    candidates: Set[str] = set()

    subsys = reaction_row["subsystem"]
    if subsys and isinstance(subsys, str) and subsys in subsys_to_genes:
        candidates |= subsys_to_genes[subsys]

    # neighborhood by shared metabolites
    for met in reaction_row["metabolites"]:
        if met in met_to_genes:
            candidates |= met_to_genes[met]

    candidates -= positives

    # If still small, fill from global
    if len(candidates) < n_needed:
        candidates |= set(all_genes.tolist())
        candidates -= positives

    candidates = list(candidates)
    if len(candidates) <= n_needed:
        return candidates

    return rng.choice(candidates, size=n_needed, replace=False).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--outdir", default="data/processed", type=str)
    ap.add_argument("--neg_per_pos", default=10, type=int)
    ap.add_argument("--seed", default=13, type=int)
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    procdir = Path(args.procdir)
    outdir = ensure_dir(args.outdir)

    reactions_df = pd.read_parquet(procdir / "reactions.parquet")
    genes_df = pd.read_parquet(procdir / "genes.parquet")

    all_genes = genes_df["gene_id"].values
    idx = build_indices(reactions_df)
    subsys_to_genes = idx["subsys_to_genes"]
    met_to_genes = idx["met_to_genes"]

    rows = []
    for _, r in tqdm(reactions_df.iterrows(), total=len(reactions_df), desc="Building pairs"):
        pos = set(r["genes"])
        if len(pos) == 0:
            continue

        # positives
        for g in pos:
            rows.append({"reaction_id": r["reaction_id"], "gene_id": g, "label": 1})

        # negatives
        n_neg = args.neg_per_pos * len(pos)
        negs = sample_hard_negatives(
            reaction_row=r,
            all_genes=all_genes,
            subsys_to_genes=subsys_to_genes,
            met_to_genes=met_to_genes,
            positives=pos,
            n_needed=n_neg,
            rng=rng,
        )
        for g in negs:
            rows.append({"reaction_id": r["reaction_id"], "gene_id": g, "label": 0})

    pairs_df = pd.DataFrame(rows).drop_duplicates(subset=["reaction_id", "gene_id"])
    pairs_df.to_parquet(outdir / "pairs.parquet", index=False)
    print(f"Saved: {outdir/'pairs.parquet'} ({len(pairs_df)} pairs, pos={pairs_df.label.sum()})")


if __name__ == "__main__":
    main()
