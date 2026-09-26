import json

import numpy as np
import pandas as pd
import pytest

from gpr_ml.core import (
    FEATURE_COLS,
    build_feature_table,
    build_pairs,
    build_reference_indices,
    classification_metrics,
    gene_fingerprint,
    indices_from_split,
    load_split,
    make_reaction_split,
    pair_features,
    save_split,
)


def synthetic_reactions():
    return pd.DataFrame(
        [
            {
                "reaction_id": "R1",
                "reaction_name": "r1",
                "subsystem": "S1",
                "genes": ["g1"],
                "metabolites": ["a", "b"],
            },
            {
                "reaction_id": "R2",
                "reaction_name": "r2",
                "subsystem": "S2",
                "genes": ["g1", "g2"],
                "metabolites": ["b", "c"],
            },
            {
                "reaction_id": "R3",
                "reaction_name": "r3",
                "subsystem": "S2",
                "genes": ["g2", "g3"],
                "metabolites": ["c", "d"],
            },
            {
                "reaction_id": "R4",
                "reaction_name": "r4",
                "subsystem": "S3",
                "genes": ["g4"],
                "metabolites": ["e", "f"],
            },
        ]
    )


def test_reaction_split_is_disjoint_and_persisted(tmp_path):
    reactions = synthetic_reactions()

    train, test = make_reaction_split(
        reactions,
        seed=13,
        test_size=0.5,
    )

    assert set(train["reaction_id"]).isdisjoint(
        set(test["reaction_id"])
    )
    assert len(train) + len(test) == len(reactions)

    path = tmp_path / "split.json"
    save_split(
        train,
        test,
        path,
        seed=13,
        test_size=0.5,
    )
    loaded_train, loaded_test = load_split(path)

    assert loaded_train == train["reaction_id"].astype(str).tolist()
    assert loaded_test == test["reaction_id"].astype(str).tolist()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 13
    assert payload["test_size"] == 0.5


def test_indices_from_split_rejects_overlap():
    frame = pd.DataFrame(
        {
            "reaction_id": ["R1", "R2"],
            "label": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="disjoint"):
        indices_from_split(
            frame,
            ["R1"],
            ["R1", "R2"],
        )


def test_gene_fingerprint_excludes_scored_training_reaction():
    reactions = synthetic_reactions()
    indices = build_reference_indices(
        reactions,
        ["R1", "R2"],
    )

    metabolites, subsystems = gene_fingerprint(
        "g1",
        current_reaction_id="R1",
        indices=indices,
    )

    assert metabolites == {"b", "c"}
    assert subsystems == {"S2"}


def test_pair_features_are_leave_one_reaction_out():
    reactions = synthetic_reactions()
    indices = build_reference_indices(
        reactions,
        ["R1", "R2"],
    )
    reaction = reactions.loc[
        reactions["reaction_id"] == "R1"
    ].iloc[0]

    features = pair_features(
        reaction,
        "g1",
        indices=indices,
    )

    assert features["overlap_mets"] == 1
    assert features["jacc_mets"] == pytest.approx(1 / 3)
    assert features["subsystem_match"] == 0


def test_build_pairs_excludes_unseen_test_positive_genes():
    reactions = synthetic_reactions()

    pairs, metadata = build_pairs(
        reactions,
        train_reactions=["R1", "R2"],
        test_reactions=["R3"],
        neg_per_pos=1,
        seed=7,
    )

    test_positive_genes = set(
        pairs.loc[
            (pairs["reaction_id"] == "R3")
            & (pairs["label"] == 1),
            "gene_id",
        ]
    )

    assert test_positive_genes == {"g2"}
    assert metadata["unseen_test_positive_genes"] == ["g3"]
    assert metadata["n_test_positive_links_excluded"] == 1


def test_build_pairs_is_deterministic():
    reactions = synthetic_reactions()

    first, _ = build_pairs(
        reactions,
        train_reactions=["R1", "R2", "R4"],
        test_reactions=["R3"],
        neg_per_pos=1,
        seed=99,
    )
    second, _ = build_pairs(
        reactions,
        train_reactions=["R1", "R2", "R4"],
        test_reactions=["R3"],
        neg_per_pos=1,
        seed=99,
    )

    pd.testing.assert_frame_equal(first, second)


def test_feature_table_has_expected_columns_and_no_missing_values():
    reactions = synthetic_reactions()
    pairs, _ = build_pairs(
        reactions,
        train_reactions=["R1", "R2", "R4"],
        test_reactions=["R3"],
        neg_per_pos=1,
        seed=2,
    )

    features = build_feature_table(
        pairs,
        reactions,
        train_reactions=["R1", "R2", "R4"],
    )

    assert list(features.columns) == [
        "reaction_id",
        "gene_id",
        "label",
        *FEATURE_COLS,
    ]
    assert not features[FEATURE_COLS].isna().any().any()


def test_classification_metrics_bootstrap_is_reaction_grouped():
    frame = pd.DataFrame(
        {
            "reaction_id": [
                "R1",
                "R1",
                "R2",
                "R2",
                "R3",
                "R3",
            ],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )
    scores = np.array(
        [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        dtype=float,
    )

    metrics = classification_metrics(
        frame,
        scores,
        bootstrap_reps=50,
        seed=5,
    )

    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["hit_at_5"] == pytest.approx(1.0)
    assert metrics["bootstrap"]["reps_requested"] == 50
    assert "average_precision" in metrics["bootstrap"]["ci95"]
