#!/usr/bin/env python3
"""Parse an SBML model into reaction and gene inventory tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import cobra
import pandas as pd
from tqdm import tqdm

from gpr_ml import ensure_dir


def reaction_metabolites(reaction: cobra.Reaction) -> set[str]:
    return {metabolite.id for metabolite in reaction.metabolites}


def parse_model(
    model_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = cobra.io.read_sbml_model(str(model_path))

    reaction_rows: list[dict[str, object]] = []
    gene_reaction_counts: dict[str, int] = {}

    for reaction in tqdm(
        model.reactions,
        desc="Parsing reactions",
    ):
        genes = sorted(
            {gene.id for gene in reaction.genes}
        )
        metabolites = sorted(
            reaction_metabolites(reaction)
        )
        subsystem = getattr(
            reaction,
            "subsystem",
            None,
        )

        reaction_rows.append(
            {
                "reaction_id": reaction.id,
                "reaction_name": reaction.name,
                "subsystem": (
                    str(subsystem)
                    if subsystem
                    else ""
                ),
                "gpr_rule": (
                    reaction.gene_reaction_rule
                    or ""
                ).strip(),
                "genes": genes,
                "metabolites": metabolites,
                "n_genes": len(genes),
                "n_metabolites": len(metabolites),
            }
        )

        for gene_id in genes:
            gene_reaction_counts[gene_id] = (
                gene_reaction_counts.get(
                    gene_id,
                    0,
                )
                + 1
            )

    reactions_df = pd.DataFrame(
        reaction_rows
    )
    if reactions_df.empty:
        raise ValueError(
            "The SBML model contains no reactions."
        )
    if reactions_df[
        "reaction_id"
    ].duplicated().any():
        raise ValueError(
            "The SBML model contains duplicate reaction IDs."
        )

    genes_df = pd.DataFrame(
        [
            {
                "gene_id": gene_id,
                "n_curated_reactions": count,
            }
            for gene_id, count
            in sorted(
                gene_reaction_counts.items()
            )
        ],
        columns=[
            "gene_id",
            "n_curated_reactions",
        ],
    )

    return reactions_df, genes_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an SBML metabolic model into "
            "GPR-ML reaction/gene tables."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to an SBML model, e.g. data/raw/iJO1366.xml",
    )
    parser.add_argument(
        "--outdir",
        default=Path("data/processed"),
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}"
        )

    outdir = ensure_dir(args.outdir)
    reactions_df, genes_df = parse_model(
        args.model
    )

    reactions_path = (
        outdir / "reactions.parquet"
    )
    genes_path = outdir / "genes.parquet"

    reactions_df.to_parquet(
        reactions_path,
        index=False,
    )
    genes_df.to_parquet(
        genes_path,
        index=False,
    )

    print(
        f"Saved: {reactions_path} "
        f"({len(reactions_df)} reactions)"
    )
    print(
        f"Saved: {genes_path} "
        f"({len(genes_df)} genes)"
    )


if __name__ == "__main__":
    main()
