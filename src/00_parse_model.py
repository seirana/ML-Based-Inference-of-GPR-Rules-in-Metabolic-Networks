#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

'''
Reads an SBML metabolic model file via cobra.io.read_sbml_model(...).
For every reaction in the model:
collects reaction metadata (id, name, subsystem)
extracts the GPR rule (gene–protein–reaction rule, boolean rule like “b0002 and (b0003 or b0004)”)
lists all genes referenced by that reaction (rxn.genes)
lists all metabolites participating in the reaction (rxn.metabolites), using metabolite IDs including compartment suffix (e.g. glc__D_c)
Builds gene fingerprints by aggregating across all reactions each gene appears in:
metabolites_fp: union of all metabolites in reactions catalyzed by that gene
subsystems_fp: union of subsystems of those reactions (if available)
Saves both tables as Parquet files:
reactions.parquet
genes.parquet
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cobra
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir


def reaction_metabolites(reaction: cobra.Reaction) -> Set[str]:
    # Use metabolite IDs (include compartment suffix like _c/_p/_e)
    return {m.id for m in reaction.metabolites.keys()}


def parse_model(model_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = cobra.io.read_sbml_model(str(model_path))

    # Reactions table
    rows_rxn: List[Dict] = []
    for rxn in model.reactions:
        gpr = (rxn.gene_reaction_rule or "").strip()
        genes = sorted({g.id for g in rxn.genes})  # genes referenced in GPR
        mets = sorted(reaction_metabolites(rxn))
        rows_rxn.append(
            {
                "reaction_id": rxn.id,
                "reaction_name": rxn.name,
                "subsystem": getattr(rxn, "subsystem", None),
                "gpr_rule": gpr,
                "genes": genes,
                "metabolites": mets,
                "n_genes": len(genes),
                "n_metabolites": len(mets),
            }
        )
    reactions_df = pd.DataFrame(rows_rxn)

    # Gene fingerprints: metabolites/subsystems seen in reactions catalyzed by that gene
    gene_to_mets: Dict[str, Set[str]] = {}
    gene_to_subsys: Dict[str, Set[str]] = {}

    for rxn in tqdm(model.reactions, desc="Building gene fingerprints"):
        mets = reaction_metabolites(rxn)
        subsys = getattr(rxn, "subsystem", None)
        for g in rxn.genes:
            gid = g.id
            gene_to_mets.setdefault(gid, set()).update(mets)
            if subsys:
                gene_to_subsys.setdefault(gid, set()).add(str(subsys))

    genes_df = pd.DataFrame(
        [
            {
                "gene_id": gid,
                "metabolites_fp": sorted(list(mets)),
                "subsystems_fp": sorted(list(gene_to_subsys.get(gid, set()))),
                "n_mets_fp": len(mets),
                "n_subsys_fp": len(gene_to_subsys.get(gid, set())),
            }
            for gid, mets in gene_to_mets.items()
        ]
    )

    return reactions_df, genes_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="Path to iJO1366 SBML file (e.g., data/raw/iJO1366.xml)")
    ap.add_argument("--outdir", default="data/processed", type=str)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    reactions_df, genes_df = parse_model(Path(args.model))

    reactions_df.to_parquet(outdir / "reactions.parquet", index=False)
    genes_df.to_parquet(outdir / "genes.parquet", index=False)

    print(f"Saved: {outdir/'reactions.parquet'} ({len(reactions_df)} reactions)")
    print(f"Saved: {outdir/'genes.parquet'} ({len(genes_df)} genes)")


if __name__ == "__main__":
    main()
