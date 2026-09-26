#!/usr/bin/env python3
"""Rank candidate genes using the trained model and training-only reference context."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from gpr_ml import (
    FEATURE_COLS,
    build_reference_indices,
    ensure_dir,
    load_split,
    score_model,
)
from gpr_ml.core import pair_features


def _as_set(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item) for item in value}
    if value is None:
        return set()
    return {str(value)}


def candidate_pool(
    reaction: pd.Series,
    *,
    indices: dict[str, object],
    max_candidates: int,
    rng: np.random.Generator,
) -> list[str]:
    if max_candidates <= 0:
        raise ValueError(
            "max_candidates must be greater than 0"
        )

    curated = _as_set(
        reaction["genes"]
    )
    candidates: set[str] = set()

    subsystem = (
        reaction["subsystem"]
        if isinstance(
            reaction["subsystem"],
            str,
        )
        else ""
    )
    if subsystem:
        candidates.update(
            indices[
                "subsystem_to_genes"
            ].get(subsystem, set())
        )

    for metabolite in _as_set(
        reaction["metabolites"]
    ):
        candidates.update(
            indices[
                "metabolite_to_genes"
            ].get(metabolite, set())
        )

    candidates -= curated

    if not candidates:
        candidates.update(
            indices["gene_vocabulary"]
        )
        candidates -= curated

    ordered = np.asarray(
        sorted(candidates),
        dtype=str,
    )
    if len(ordered) > max_candidates:
        ordered = rng.choice(
            ordered,
            size=max_candidates,
            replace=False,
        )

    return sorted(
        map(str, ordered.tolist())
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate genes for each reaction "
            "using training-only reference fingerprints."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--model_path",
        default=Path(
            "reports/models/xgb.joblib"
        ),
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path("reports/candidates"),
        type=Path,
    )
    parser.add_argument(
        "--topk",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--max_candidates",
        default=3000,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
    )
    parser.add_argument(
        "--min_curated_genes",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--reaction_scope",
        choices=[
            "all",
            "test",
            "train",
        ],
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.topk <= 0:
        raise ValueError(
            "topk must be greater than 0"
        )
    if args.min_curated_genes < 0:
        raise ValueError(
            "min_curated_genes must be non-negative"
        )

    rng = np.random.default_rng(
        args.seed
    )
    outdir = ensure_dir(args.outdir)

    reactions_df = pd.read_parquet(
        args.procdir / "reactions.parquet"
    )
    train_reactions, test_reactions = load_split(
        args.procdir
        / "split_reactions.json"
    )
    train_set = set(train_reactions)
    test_set = set(test_reactions)

    reference_indices = (
        build_reference_indices(
            reactions_df,
            train_reactions,
        )
    )

    if args.reaction_scope == "test":
        target_df = reactions_df[
            reactions_df[
                "reaction_id"
            ].astype(str).isin(test_set)
        ].copy()
    elif args.reaction_scope == "train":
        target_df = reactions_df[
            reactions_df[
                "reaction_id"
            ].astype(str).isin(train_set)
        ].copy()
    else:
        target_df = reactions_df.copy()

    model = joblib.load(
        args.model_path
    )
    rows: list[dict[str, object]] = []

    for _, reaction in tqdm(
        target_df.iterrows(),
        total=len(target_df),
        desc="Ranking candidates",
    ):
        curated = _as_set(
            reaction["genes"]
        )
        if len(curated) < (
            args.min_curated_genes
        ):
            continue

        candidates = candidate_pool(
            reaction,
            indices=reference_indices,
            max_candidates=(
                args.max_candidates
            ),
            rng=rng,
        )
        if not candidates:
            continue

        feature_rows = [
            pair_features(
                reaction,
                gene_id,
                indices=reference_indices,
            )
            for gene_id in candidates
        ]
        feature_frame = pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLS,
        )
        scores = score_model(
            model,
            feature_frame.to_numpy(
                dtype=np.float32
            ),
        )

        order = np.argsort(-scores)[
            : args.topk
        ]
        for rank, index in enumerate(
            order,
            start=1,
        ):
            rows.append(
                {
                    "reaction_id": str(
                        reaction[
                            "reaction_id"
                        ]
                    ),
                    "reaction_name": str(
                        reaction.get(
                            "reaction_name",
                            "",
                        )
                    ),
                    "subsystem": (
                        reaction[
                            "subsystem"
                        ]
                        if isinstance(
                            reaction[
                                "subsystem"
                            ],
                            str,
                        )
                        else ""
                    ),
                    "candidate_gene": (
                        candidates[
                            int(index)
                        ]
                    ),
                    "rank": rank,
                    "score": float(
                        scores[
                            int(index)
                        ]
                    ),
                    "curated_genes": ";".join(
                        sorted(curated)
                    ),
                    "n_curated_genes": (
                        len(curated)
                    ),
                    "reference_scope": (
                        "training reactions"
                    ),
                }
            )

    output = pd.DataFrame(
        rows,
        columns=[
            "reaction_id",
            "reaction_name",
            "subsystem",
            "candidate_gene",
            "rank",
            "score",
            "curated_genes",
            "n_curated_genes",
            "reference_scope",
        ],
    )

    model_name = args.model_path.stem
    output_path = (
        outdir
        / f"top_candidates_{model_name}.csv"
    )
    output.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path} "
        f"({len(output)} rows)"
    )


if __name__ == "__main__":
    main()
