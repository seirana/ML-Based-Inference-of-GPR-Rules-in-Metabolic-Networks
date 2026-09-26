"""Backward-compatible imports for historical scripts.

New code should import from :mod:`gpr_ml` directly.
"""

from gpr_ml.core import (
    FEATURE_COLS,
    build_feature_table,
    build_pairs,
    build_reference_indices,
    classification_metrics,
    ensure_dir,
    hit_at_k,
    indices_from_split,
    jaccard,
    load_split,
    make_reaction_split,
    pair_features,
    save_json,
    save_split,
    score_model,
)

__all__ = [
    "FEATURE_COLS",
    "build_feature_table",
    "build_pairs",
    "build_reference_indices",
    "classification_metrics",
    "ensure_dir",
    "hit_at_k",
    "indices_from_split",
    "jaccard",
    "load_split",
    "make_reaction_split",
    "pair_features",
    "save_json",
    "save_split",
    "score_model",
]
