#!/usr/bin/env python3
"""Train XGBoost using the stored test split and a train-only validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from gpr_ml import (
    FEATURE_COLS,
    classification_metrics,
    ensure_dir,
    indices_from_split,
    load_split,
    save_json,
)


def grouped_validation_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    seed: int,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError(
            "validation_size must be between 0 and 1"
        )

    train_frame = frame.iloc[
        train_idx
    ].reset_index()
    groups = train_frame[
        "reaction_id"
    ].astype(str).to_numpy()
    labels = train_frame[
        "label"
    ].to_numpy(dtype=int)
    dummy = np.zeros(
        (len(train_frame), 1),
        dtype=np.float32,
    )

    for offset in range(25):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed + offset,
        )
        local_train, local_val = next(
            splitter.split(
                dummy,
                labels,
                groups=groups,
            )
        )
        if (
            len(np.unique(labels[local_train])) == 2
            and len(np.unique(labels[local_val])) == 2
        ):
            original_indices = train_frame[
                "index"
            ].to_numpy(dtype=int)
            return (
                original_indices[local_train],
                original_indices[local_val],
            )

    raise ValueError(
        "Could not create a grouped validation split "
        "containing both classes."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost for GPR association "
            "prediction with grouped validation."
        )
    )
    parser.add_argument(
        "--procdir",
        default=Path("data/processed"),
        type=Path,
    )
    parser.add_argument(
        "--outdir",
        default=Path("reports/metrics"),
        type=Path,
    )
    parser.add_argument(
        "--model_out",
        default=Path(
            "reports/models/xgb.joblib"
        ),
        type=Path,
    )
    parser.add_argument(
        "--seed",
        default=13,
        type=int,
    )
    parser.add_argument(
        "--validation_size",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--bootstrap_reps",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--n_estimators",
        default=1200,
        type=int,
    )
    parser.add_argument(
        "--max_depth",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.03,
        type=float,
    )
    parser.add_argument(
        "--subsample",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--colsample_bytree",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--min_child_weight",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--reg_lambda",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--early_stopping_rounds",
        default=50,
        type=int,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError(
            "bootstrap_reps must be non-negative"
        )

    ensure_dir(args.model_out.parent)
    outdir = ensure_dir(args.outdir)

    frame = pd.read_parquet(
        args.procdir / "features.parquet"
    )
    train_reactions, test_reactions = (
        load_split(
            args.procdir
            / "split_reactions.json"
        )
    )
    train_idx, test_idx = indices_from_split(
        frame,
        train_reactions,
        test_reactions,
    )
    fit_idx, validation_idx = (
        grouped_validation_split(
            frame,
            train_idx,
            seed=args.seed,
            validation_size=(
                args.validation_size
            ),
        )
    )

    x = frame[FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    y = frame["label"].to_numpy(
        dtype=int
    )

    y_fit = y[fit_idx]
    n_positive = int(y_fit.sum())
    n_negative = int(
        (y_fit == 0).sum()
    )
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "XGBoost fit split must contain both classes."
        )
    scale_pos_weight = (
        n_negative / n_positive
    )

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=(
            args.colsample_bytree
        ),
        min_child_weight=(
            args.min_child_weight
        ),
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=(
            scale_pos_weight
        ),
        n_jobs=-1,
        random_state=args.seed,
        early_stopping_rounds=(
            args.early_stopping_rounds
        ),
        tree_method="hist",
    )

    model.fit(
        x[fit_idx],
        y[fit_idx],
        eval_set=[
            (
                x[validation_idx],
                y[validation_idx],
            )
        ],
        verbose=False,
    )

    scores = model.predict_proba(
        x[test_idx]
    )[:, 1]
    test_frame = frame.iloc[
        test_idx
    ].copy()

    metrics = classification_metrics(
        test_frame,
        scores,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    metrics.update(
        {
            "model": "xgboost",
            "feature_cols": FEATURE_COLS,
            "seed": int(args.seed),
            "validation_size": float(
                args.validation_size
            ),
            "best_iteration": int(
                getattr(
                    model,
                    "best_iteration",
                    -1,
                )
            ),
            "scale_pos_weight": float(
                scale_pos_weight
            ),
            "params": {
                "n_estimators": (
                    args.n_estimators
                ),
                "max_depth": args.max_depth,
                "learning_rate": (
                    args.learning_rate
                ),
                "subsample": args.subsample,
                "colsample_bytree": (
                    args.colsample_bytree
                ),
                "min_child_weight": (
                    args.min_child_weight
                ),
                "reg_lambda": (
                    args.reg_lambda
                ),
                "early_stopping_rounds": (
                    args.early_stopping_rounds
                ),
            },
        }
    )

    save_json(
        metrics,
        outdir / "xgb_metrics.json",
    )
    joblib.dump(
        model,
        args.model_out,
    )

    importances = {
        name: float(value)
        for name, value
        in zip(
            FEATURE_COLS,
            model.feature_importances_,
            strict=True,
        )
    }
    save_json(
        {
            "feature_importances": (
                importances
            )
        },
        outdir
        / "xgb_feature_importances.json",
    )

    print(metrics)
    print(
        f"Saved model: {args.model_out}"
    )


if __name__ == "__main__":
    main()
