#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def load_maps(procdir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str], Dict[str, Set[str]]]:
    reactions_df = pd.read_parquet(procdir / "reactions.parquet")
    genes_df = pd.read_parquet(procdir / "genes.parquet")

    rxn_to_mets = {r.reaction_id: set(r.metabolites) for r in reactions_df.itertuples(index=False)}
    rxn_to_subsys = {r.reaction_id: (r.subsystem if isinstance(r.subsystem, str) else "") for r in reactions_df.itertuples(index=False)}

    gene_to_metsfp = {g.gene_id: set(g.metabolites_fp) for g in genes_df.itertuples(index=False)}
    gene_to_subsysfp = {g.gene_id: set(g.subsystems_fp) for g in genes_df.itertuples(index=False)}

    return rxn_to_mets, gene_to_metsfp, rxn_to_subsys, gene_to_subsysfp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--outdir", default="data/processed", type=str)
    args = ap.parse_args()

    procdir = Path(args.procdir)
    outdir = ensure_dir(args.outdir)

    pairs_df = pd.read_parquet(procdir / "pairs.parquet")
    rxn_to_mets, gene_to_metsfp, rxn_to_subsys, gene_to_subsysfp = load_maps(procdir)

    feats = []
    for r_id, g_id, y in tqdm(pairs_df[["reaction_id", "gene_id", "label"]].itertuples(index=False), total=len(pairs_df), desc="Computing features"):
        rm = rxn_to_mets.get(r_id, set())
        gm = gene_to_metsfp.get(g_id, set())

        subsys = rxn_to_subsys.get(r_id, "")
        g_sub = gene_to_subsysfp.get(g_id, set())

        inter = len(rm & gm)
        feats.append(
            {
                "reaction_id": r_id,
                "gene_id": g_id,
                "label": int(y),
                "jacc_mets": jaccard(rm, gm),
                "overlap_mets": inter,
                "n_mets_rxn": len(rm),
                "n_mets_gene_fp": len(gm),
                "subsystem_match": 1 if (subsys and subsys in g_sub) else 0,
                "n_subsys_gene_fp": len(g_sub),
            }
        )

    feat_df = pd.DataFrame(feats)
    feat_df.to_parquet(outdir / "features.parquet", index=False)
    print(f"Saved: {outdir/'features.parquet'} ({len(feat_df)} rows)")


if __name__ == "__main__":
    main()


