#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from utils import ensure_dir, save_json, seed_everything

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("xgboost is not installed. Run: pip install xgboost") from e


FEATURE_COLS = [
    "jacc_mets",
    "overlap_mets",
    "n_mets_rxn",
    "n_mets_gene_fp",
    "subsystem_match",
    "n_subsys_gene_fp",
]


def hit_at_k(df: pd.DataFrame, k: int) -> float:
    hits = []
    for rid, sub in df.groupby("reaction_id"):
        if (sub["label"] == 1).sum() == 0:
            continue
        topk = sub.sort_values("score", ascending=False).head(k)
        hits.append(int((topk["label"] == 1).any()))
    return float(np.mean(hits)) if hits else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--outdir", default="reports/metrics", type=str)
    ap.add_argument("--model_out", default="reports/models/xgb.joblib", type=str)
    ap.add_argument("--seed", default=13, type=int)

    # sensible defaults (you can tune later)
    ap.add_argument("--n_estimators", default=1200, type=int)
    ap.add_argument("--max_depth", default=6, type=int)
    ap.add_argument("--learning_rate", default=0.03, type=float)
    ap.add_argument("--subsample", default=0.9, type=float)
    ap.add_argument("--colsample_bytree", default=0.9, type=float)
    ap.add_argument("--min_child_weight", default=1.0, type=float)
    ap.add_argument("--reg_lambda", default=1.0, type=float)
    ap.add_argument("--early_stopping_rounds", default=50, type=int)

    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(Path(args.model_out).parent)
    outdir = ensure_dir(args.outdir)

    df = pd.read_parquet(Path(args.procdir) / "features.parquet")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["reaction_id"].values

    # Reaction-wise split (same as LogReg)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # Handle imbalance
    n_pos = int(ytr.sum())
    n_neg = int((ytr == 0).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1))

    # Use a validation split from TRAIN for early stopping (still reaction-wise)
    # We'll just do a second GroupShuffleSplit on train reactions.
    train_df = df.iloc[train_idx].copy()
    Xtr_full = Xtr
    ytr_full = ytr
    gtr_full = train_df["reaction_id"].values

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + 1)
    tr2_idx, val_idx = next(gss2.split(Xtr_full, ytr_full, groups=gtr_full))

    Xtr2, Xval = Xtr_full[tr2_idx], Xtr_full[val_idx]
    ytr2, yval = ytr_full[tr2_idx], ytr_full[val_idx]

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=args.seed,
    )

    clf.fit(
        Xtr2,
        ytr2,
        eval_set=[(Xval, yval)],
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    proba = clf.predict_proba(Xte)[:, 1]
    ap_score = average_precision_score(yte, proba)
    roc = roc_auc_score(yte, proba)

    test_df = df.iloc[test_idx].copy()
    test_df["score"] = proba

    metrics = {
        "model": "xgboost",
        "average_precision": float(ap_score),
        "roc_auc": float(roc),
        "hit_at_5": hit_at_k(test_df, 5),
        "hit_at_10": hit_at_k(test_df, 10),
        "hit_at_20": hit_at_k(test_df, 20),
        "best_iteration": int(getattr(clf, "best_iteration", -1)),
        "n_test_pairs": int(len(test_df)),
        "n_test_pos": int(test_df["label"].sum()),
        "scale_pos_weight": float(scale_pos_weight),
        "feature_cols": FEATURE_COLS,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
    }

    save_json(metrics, Path(outdir) / "xgb_metrics.json")
    joblib.dump(clf, args.model_out)

    # Feature importances (gain-based is available via booster; keep simple here)
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_.tolist()))
    save_json({"feature_importances": importances}, Path(outdir) / "xgb_feature_importances.json")

    print(metrics)
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
