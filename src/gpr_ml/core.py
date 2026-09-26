from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


PathLike = str | Path

FEATURE_COLS = [
    "jacc_mets",
    "overlap_mets",
    "n_mets_rxn",
    "n_mets_gene_fp",
    "subsystem_match",
    "n_subsys_gene_fp",
]


def ensure_dir(path: PathLike) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def save_json(
    obj: Mapping[str, Any],
    path: PathLike,
    *,
    indent: int = 2,
) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(
        json.dumps(obj, indent=indent, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(destination)
    return destination


def _validate_reaction_ids(reactions_df: pd.DataFrame) -> None:
    if "reaction_id" not in reactions_df.columns:
        raise ValueError("reactions_df must contain a 'reaction_id' column")
    if reactions_df["reaction_id"].isna().any():
        raise ValueError("reaction_id values must not be missing")
    if reactions_df["reaction_id"].duplicated().any():
        raise ValueError("reaction_id values must be unique")


def make_reaction_split(
    reactions_df: pd.DataFrame,
    *,
    seed: int = 13,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_reaction_ids(reactions_df)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(reactions_df) < 2:
        raise ValueError("At least two reactions are required for a split")

    groups = reactions_df["reaction_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=int(seed),
    )
    dummy = np.zeros((len(reactions_df), 1), dtype=np.float32)
    labels = np.zeros(len(reactions_df), dtype=np.int8)
    train_idx, test_idx = next(
        splitter.split(dummy, labels, groups=groups)
    )

    train_df = reactions_df.iloc[train_idx].reset_index(drop=True)
    test_df = reactions_df.iloc[test_idx].reset_index(drop=True)

    train_ids = set(train_df["reaction_id"].astype(str))
    test_ids = set(test_df["reaction_id"].astype(str))
    if train_ids & test_ids:
        raise RuntimeError("Reaction split is not disjoint")

    return train_df, test_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: PathLike,
    *,
    seed: int | None = None,
    test_size: float | None = None,
) -> Path:
    _validate_reaction_ids(train_df)
    _validate_reaction_ids(test_df)

    train_ids = train_df["reaction_id"].astype(str).tolist()
    test_ids = test_df["reaction_id"].astype(str).tolist()
    if set(train_ids) & set(test_ids):
        raise ValueError("Train/test reaction sets must be disjoint")

    payload: dict[str, Any] = {
        "train_reactions": train_ids,
        "test_reactions": test_ids,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if test_size is not None:
        payload["test_size"] = float(test_size)

    return save_json(payload, out_path)


def load_split(path: PathLike) -> tuple[list[str], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train = [str(value) for value in payload.get("train_reactions", [])]
    test = [str(value) for value in payload.get("test_reactions", [])]
    if not train or not test:
        raise ValueError("Split file must contain non-empty train and test reaction lists")
    if set(train) & set(test):
        raise ValueError("Split file contains reactions in both train and test")
    return train, test


def indices_from_split(
    frame: pd.DataFrame,
    train_reactions: Iterable[str],
    test_reactions: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    if "reaction_id" not in frame.columns:
        raise ValueError("frame must contain reaction_id")

    train_set = set(map(str, train_reactions))
    test_set = set(map(str, test_reactions))
    if train_set & test_set:
        raise ValueError("Train/test reaction sets must be disjoint")

    reaction_ids = frame["reaction_id"].astype(str)
    train_mask = reaction_ids.isin(train_set).to_numpy()
    test_mask = reaction_ids.isin(test_set).to_numpy()

    if not train_mask.any():
        raise ValueError("No rows match training reactions")
    if not test_mask.any():
        raise ValueError("No rows match test reactions")
    if np.any(train_mask & test_mask):
        raise RuntimeError("A row was assigned to both train and test")

    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item) for item in value if str(item)}
    return {str(value)} if str(value) else set()


def build_reference_indices(
    reactions_df: pd.DataFrame,
    reference_reactions: Sequence[str],
) -> dict[str, Any]:
    required = {"reaction_id", "genes", "metabolites", "subsystem"}
    missing = required - set(reactions_df.columns)
    if missing:
        raise ValueError(f"Reaction table is missing columns: {sorted(missing)}")

    reference_set = set(map(str, reference_reactions))
    reference_df = reactions_df[
        reactions_df["reaction_id"].astype(str).isin(reference_set)
    ].copy()
    if reference_df.empty:
        raise ValueError("No reference reactions were found in the reaction table")

    reaction_to_mets: dict[str, set[str]] = {}
    reaction_to_subsystem: dict[str, str] = {}
    gene_to_reactions: dict[str, set[str]] = {}
    subsystem_to_genes: dict[str, set[str]] = {}
    metabolite_to_genes: dict[str, set[str]] = {}

    for row in reference_df.itertuples(index=False):
        reaction_id = str(row.reaction_id)
        genes = _as_string_set(row.genes)
        metabolites = _as_string_set(row.metabolites)
        subsystem = row.subsystem if isinstance(row.subsystem, str) else ""

        reaction_to_mets[reaction_id] = metabolites
        reaction_to_subsystem[reaction_id] = subsystem

        for gene in genes:
            gene_to_reactions.setdefault(gene, set()).add(reaction_id)

        if subsystem:
            subsystem_to_genes.setdefault(subsystem, set()).update(genes)
        for metabolite in metabolites:
            metabolite_to_genes.setdefault(metabolite, set()).update(genes)

    return {
        "reaction_to_mets": reaction_to_mets,
        "reaction_to_subsystem": reaction_to_subsystem,
        "gene_to_reactions": gene_to_reactions,
        "subsystem_to_genes": subsystem_to_genes,
        "metabolite_to_genes": metabolite_to_genes,
        "gene_vocabulary": sorted(gene_to_reactions),
    }


def gene_fingerprint(
    gene_id: str,
    *,
    current_reaction_id: str,
    indices: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Build a gene fingerprint from reference reactions, excluding the scored reaction."""

    reaction_ids = set(indices["gene_to_reactions"].get(str(gene_id), set()))
    reaction_ids.discard(str(current_reaction_id))

    metabolites: set[str] = set()
    subsystems: set[str] = set()
    for reaction_id in reaction_ids:
        metabolites.update(
            indices["reaction_to_mets"].get(reaction_id, set())
        )
        subsystem = indices["reaction_to_subsystem"].get(reaction_id, "")
        if subsystem:
            subsystems.add(subsystem)

    return metabolites, subsystems


def pair_features(
    reaction_row: pd.Series,
    gene_id: str,
    *,
    indices: Mapping[str, Any],
) -> dict[str, float | int]:
    reaction_id = str(reaction_row["reaction_id"])
    reaction_mets = _as_string_set(reaction_row["metabolites"])
    subsystem = (
        reaction_row["subsystem"]
        if isinstance(reaction_row["subsystem"], str)
        else ""
    )
    gene_mets, gene_subsystems = gene_fingerprint(
        gene_id,
        current_reaction_id=reaction_id,
        indices=indices,
    )

    overlap = len(reaction_mets & gene_mets)
    return {
        "jacc_mets": jaccard(reaction_mets, gene_mets),
        "overlap_mets": int(overlap),
        "n_mets_rxn": int(len(reaction_mets)),
        "n_mets_gene_fp": int(len(gene_mets)),
        "subsystem_match": int(bool(subsystem and subsystem in gene_subsystems)),
        "n_subsys_gene_fp": int(len(gene_subsystems)),
    }


def _sample_candidates(
    candidates: set[str],
    *,
    n_needed: int,
    rng: np.random.Generator,
) -> list[str]:
    if n_needed <= 0 or not candidates:
        return []

    ordered = np.asarray(sorted(candidates), dtype=str)
    if len(ordered) <= n_needed:
        return ordered.tolist()

    chosen = rng.choice(
        ordered,
        size=n_needed,
        replace=False,
    )
    return sorted(map(str, chosen.tolist()))


def build_pairs(
    reactions_df: pd.DataFrame,
    *,
    train_reactions: Sequence[str],
    test_reactions: Sequence[str],
    neg_per_pos: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if neg_per_pos <= 0:
        raise ValueError("neg_per_pos must be greater than 0")

    train_set = set(map(str, train_reactions))
    test_set = set(map(str, test_reactions))
    if train_set & test_set:
        raise ValueError("Train/test reaction sets must be disjoint")

    indices = build_reference_indices(reactions_df, list(train_set))
    train_gene_vocabulary = set(indices["gene_vocabulary"])
    rng = np.random.default_rng(int(seed))

    rows: list[dict[str, Any]] = []
    unseen_test_positive_genes: set[str] = set()
    n_test_positive_links_excluded = 0

    for _, reaction in reactions_df.iterrows():
        reaction_id = str(reaction["reaction_id"])
        if reaction_id not in train_set and reaction_id not in test_set:
            continue

        curated = _as_string_set(reaction["genes"])
        if reaction_id in test_set:
            unseen = curated - train_gene_vocabulary
            unseen_test_positive_genes.update(unseen)
            n_test_positive_links_excluded += len(unseen)
            positives = curated & train_gene_vocabulary
        else:
            positives = curated

        if not positives:
            continue

        for gene in sorted(positives):
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "gene_id": gene,
                    "label": 1,
                }
            )

        candidates: set[str] = set()
        subsystem = (
            reaction["subsystem"]
            if isinstance(reaction["subsystem"], str)
            else ""
        )
        if subsystem:
            candidates.update(
                indices["subsystem_to_genes"].get(subsystem, set())
            )
        for metabolite in _as_string_set(reaction["metabolites"]):
            candidates.update(
                indices["metabolite_to_genes"].get(metabolite, set())
            )

        candidates.update(train_gene_vocabulary)
        candidates -= curated
        n_negatives = neg_per_pos * len(positives)

        for gene in _sample_candidates(
            candidates,
            n_needed=n_negatives,
            rng=rng,
        ):
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "gene_id": gene,
                    "label": 0,
                }
            )

    pairs = pd.DataFrame(
        rows,
        columns=["reaction_id", "gene_id", "label"],
    ).drop_duplicates(subset=["reaction_id", "gene_id"])

    if pairs.empty:
        raise ValueError("Pair construction produced no rows")

    metadata = {
        "seed": int(seed),
        "neg_per_pos": int(neg_per_pos),
        "n_pairs": int(len(pairs)),
        "n_positive_pairs": int(pairs["label"].sum()),
        "n_negative_pairs": int((pairs["label"] == 0).sum()),
        "train_gene_vocabulary_size": int(len(train_gene_vocabulary)),
        "unseen_test_positive_genes": sorted(unseen_test_positive_genes),
        "n_test_positive_links_excluded": int(n_test_positive_links_excluded),
        "candidate_reference_scope": "training reactions only",
    }
    return pairs, metadata


def build_feature_table(
    pairs_df: pd.DataFrame,
    reactions_df: pd.DataFrame,
    *,
    train_reactions: Sequence[str],
) -> pd.DataFrame:
    required_pairs = {"reaction_id", "gene_id", "label"}
    if not required_pairs.issubset(pairs_df.columns):
        raise ValueError(
            f"Pair table is missing columns: {sorted(required_pairs - set(pairs_df.columns))}"
        )

    indices = build_reference_indices(reactions_df, train_reactions)
    reaction_rows = {
        str(row["reaction_id"]): row
        for _, row in reactions_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for pair in pairs_df.itertuples(index=False):
        reaction_id = str(pair.reaction_id)
        reaction = reaction_rows.get(reaction_id)
        if reaction is None:
            raise ValueError(f"Unknown reaction_id in pairs: {reaction_id}")

        features = pair_features(
            reaction,
            str(pair.gene_id),
            indices=indices,
        )
        rows.append(
            {
                "reaction_id": reaction_id,
                "gene_id": str(pair.gene_id),
                "label": int(pair.label),
                **features,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["reaction_id", "gene_id", "label", *FEATURE_COLS],
    )


def score_model(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(features), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
    else:
        scores = np.asarray(model.predict(features), dtype=float)

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(features):
        raise ValueError("Model returned an unexpected score shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned non-finite scores")
    return scores


def hit_at_k(scored_df: pd.DataFrame, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    hits: list[int] = []
    for _, group in scored_df.groupby("reaction_id", sort=False):
        if not (group["label"] == 1).any():
            continue
        top = group.sort_values("score", ascending=False).head(k)
        hits.append(int((top["label"] == 1).any()))

    return float(np.mean(hits)) if hits else 0.0


def _point_metrics(scored_df: pd.DataFrame) -> dict[str, float]:
    labels = scored_df["label"].to_numpy(dtype=int)
    scores = scored_df["score"].to_numpy(dtype=float)

    metrics = {
        "average_precision": float(average_precision_score(labels, scores)),
        "hit_at_5": hit_at_k(scored_df, 5),
        "hit_at_10": hit_at_k(scored_df, 10),
        "hit_at_20": hit_at_k(scored_df, 20),
    }
    if len(np.unique(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def classification_metrics(
    frame: pd.DataFrame,
    scores: Sequence[float],
    *,
    bootstrap_reps: int = 0,
    seed: int = 13,
) -> dict[str, Any]:
    scored = frame[["reaction_id", "label"]].copy()
    scored["score"] = np.asarray(scores, dtype=float)

    if len(scored) == 0:
        raise ValueError("Cannot evaluate an empty frame")
    if not np.all(np.isfinite(scored["score"])):
        raise ValueError("Scores must be finite")

    point = _point_metrics(scored)
    result: dict[str, Any] = {
        **point,
        "n_pairs": int(len(scored)),
        "n_positive_pairs": int(scored["label"].sum()),
        "n_reactions": int(scored["reaction_id"].nunique()),
    }

    if bootstrap_reps <= 0:
        return result

    reaction_ids = np.asarray(
        sorted(scored["reaction_id"].astype(str).unique()),
        dtype=str,
    )
    if len(reaction_ids) < 2:
        result["bootstrap"] = {
            "reps_requested": int(bootstrap_reps),
            "reps_valid": 0,
            "ci95": {},
        }
        return result

    grouped = {
        reaction_id: scored[
            scored["reaction_id"].astype(str) == reaction_id
        ].copy()
        for reaction_id in reaction_ids
    }
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        key: []
        for key in point
    }

    for _ in range(int(bootstrap_reps)):
        drawn = rng.choice(
            reaction_ids,
            size=len(reaction_ids),
            replace=True,
        )
        pieces = []
        for draw_index, reaction_id in enumerate(drawn):
            piece = grouped[str(reaction_id)].copy()
            piece["reaction_id"] = (
                piece["reaction_id"].astype(str)
                + f"__bootstrap_{draw_index}"
            )
            pieces.append(piece)

        sample = pd.concat(pieces, ignore_index=True)
        if sample["label"].nunique() < 2:
            continue
        metrics = _point_metrics(sample)
        for name, value in metrics.items():
            if np.isfinite(value):
                samples[name].append(float(value))

    ci95: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if not values:
            continue
        lower, upper = np.quantile(values, [0.025, 0.975])
        ci95[name] = {
            "low": float(lower),
            "high": float(upper),
        }

    valid_counts = [len(values) for values in samples.values() if values]
    result["bootstrap"] = {
        "reps_requested": int(bootstrap_reps),
        "reps_valid_min": int(min(valid_counts)) if valid_counts else 0,
        "ci95": ci95,
    }
    return result
