#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""
'''
Rank candidate genes for each metabolic reaction using a trained gene–reaction association model.

For every reaction in reactions.parquet, the script builds a plausible candidate pool of genes
(using subsystem and shared-metabolite neighborhoods), computes the same engineered features used
during model training, scores each candidate with a pretrained model (default: XGBoost), and
exports the top-K ranked genes per reaction to a CSV file for downstream inspection and validation.
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir


FEATURE_COLS = [
    "jacc_mets",
    "overlap_mets",
    "n_mets_rxn",
    "n_mets_gene_fp",
    "subsystem_match",
    "n_subsys_gene_fp",
]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def build_indices(reactions_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    subsys_to_genes: Dict[str, Set[str]] = {}
    met_to_genes: Dict[str, Set[str]] = {}

    for _, row in reactions_df.iterrows():
        genes = set(row["genes"])
        subsys = row["subsystem"]
        if subsys and isinstance(subsys, str):
            subsys_to_genes.setdefault(subsys, set()).update(genes)
        for met in row["metabolites"]:
            met_to_genes.setdefault(met, set()).update(genes)

    return subsys_to_genes, met_to_genes


def candidate_pool(
    r: pd.Series,
    all_genes: np.ndarray,
    subsys_to_genes: Dict[str, Set[str]],
    met_to_genes: Dict[str, Set[str]],
    max_candidates: int,
    rng: np.random.Generator,
) -> List[str]:
    pos = set(r["genes"])
    cand: Set[str] = set()

    subsys = r["subsystem"]
    if subsys and isinstance(subsys, str) and subsys in subsys_to_genes:
        cand |= subsys_to_genes[subsys]

    for met in r["metabolites"]:
        if met in met_to_genes:
            cand |= met_to_genes[met]

    cand -= pos

    # Keep it bounded for speed; if still too big, sample
    cand = list(cand)
    if len(cand) > max_candidates:
        cand = rng.choice(cand, size=max_candidates, replace=False).tolist()

    # If empty, fallback to random global sampling (bounded)
    if len(cand) == 0:
        cand = rng.choice(all_genes, size=min(max_candidates, len(all_genes)), replace=False).tolist()
        cand = [g for g in cand if g not in pos]

    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--model_path", default="reports/models/xgb.joblib", type=str)
    ap.add_argument("--outdir", default="reports/candidates", type=str)
    ap.add_argument("--topk", default=10, type=int)
    ap.add_argument("--max_candidates", default=3000, type=int, help="Candidate pool size per reaction")
    ap.add_argument("--seed", default=13, type=int)

    # Optional: rank only reactions with >= this many curated genes (avoid trivial singletons)
    ap.add_argument("--min_curated_genes", default=1, type=int)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    procdir = Path(args.procdir)
    outdir = ensure_dir(args.outdir)

    reactions_df = pd.read_parquet(procdir / "reactions.parquet")
    genes_df = pd.read_parquet(procdir / "genes.parquet")

    all_genes = genes_df["gene_id"].values
    gene_to_metsfp = {g.gene_id: set(g.metabolites_fp) for g in genes_df.itertuples(index=False)}
    gene_to_subsysfp = {g.gene_id: set(g.subsystems_fp) for g in genes_df.itertuples(index=False)}

    subsys_to_genes, met_to_genes = build_indices(reactions_df)

    model = joblib.load(args.model_path)
    use_predict_proba = hasattr(model, "predict_proba")

    rows_out = []
    for _, r in tqdm(reactions_df.iterrows(), total=len(reactions_df), desc="Ranking candidates"):
        curated = set(r["genes"])
        if len(curated) < args.min_curated_genes:
            continue

        rm = set(r["metabolites"])
        subsys = r["subsystem"] if isinstance(r["subsystem"], str) else ""

        cands = candidate_pool(
            r=r,
            all_genes=all_genes,
            subsys_to_genes=subsys_to_genes,
            met_to_genes=met_to_genes,
            max_candidates=args.max_candidates,
            rng=rng,
        )

        feats = []
        for g in cands:
            gm = gene_to_metsfp.get(g, set())
            gsub = gene_to_subsysfp.get(g, set())
            inter = len(rm & gm)
            feats.append(
                [
                    jaccard(rm, gm),
                    inter,
                    len(rm),
                    len(gm),
                    1 if (subsys and subsys in gsub) else 0,
                    len(gsub),
                ]
            )

        if len(feats) == 0:
            continue

        X = np.asarray(feats, dtype=np.float32)
        if use_predict_proba:
            scores = model.predict_proba(X)[:, 1]
        else:
            # fallback for models without predict_proba
            scores = model.predict(X)

        # Take topK
        order = np.argsort(-scores)
        top = order[: args.topk]

        for idx in top:
            rows_out.append(
                {
                    "reaction_id": r["reaction_id"],
                    "reaction_name": r["reaction_name"],
                    "subsystem": subsys,
                    "candidate_gene": cands[idx],
                    "score": float(scores[idx]),
                    "curated_genes": ";".join(sorted(curated)),
                    "n_curated_genes": int(len(curated)),
                }
            )

    out_df = pd.DataFrame(rows_out).sort_values(["reaction_id", "score"], ascending=[True, False])
    out_path = outdir / "top_candidates_xgb.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
