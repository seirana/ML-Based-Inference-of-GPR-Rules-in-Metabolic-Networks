#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

'''
Evaluate a trained gene–reaction association model on a reaction-wise held-out test set.

The script loads engineered pair-level features (features.parquet), constructs a grouped
train/test split by reaction_id (to avoid leakage across the same reaction), loads a
pretrained model from a .joblib file, and computes PR-AUC (average precision), ROC-AUC,
and reaction-level hit@k metrics (whether at least one true gene is ranked in the top-k
candidates per reaction). Results are saved as a JSON report for reproducibility.
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from utils import ensure_dir, save_json, seed_everything


FEATURE_COLS: List[str] = [
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


def score_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Some models might expose decision_function; keep it safe:
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # map to (0,1) for comparability; sigmoid
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--model_path", required=True, type=str, help="Path to trained .joblib model")
    ap.add_argument("--outdir", default="reports/metrics", type=str)
    ap.add_argument("--seed", default=13, type=int)
    ap.add_argument("--test_size", default=0.2, type=float)
    args = ap.parse_args()

    seed_everything(args.seed)
    procdir = Path(args.procdir)
    outdir = ensure_dir(args.outdir)

    df = pd.read_parquet(procdir / "features.parquet")
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["reaction_id"].values

    # Reaction-wise split (must match training logic)
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    _, test_idx = next(gss.split(X, y, groups=groups))

    Xte = X[test_idx]
    yte = y[test_idx]

    model = joblib.load(args.model_path)
    scores = score_model(model, Xte)

    ap_score = average_precision_score(yte, scores)
    roc = roc_auc_score(yte, scores)

    test_df = df.iloc[test_idx].copy()
    test_df["score"] = scores

    metrics = {
        "model_path": str(args.model_path),
        "average_precision": float(ap_score),
        "roc_auc": float(roc),
        "hit_at_5": hit_at_k(test_df, 5),
        "hit_at_10": hit_at_k(test_df, 10),
        "hit_at_20": hit_at_k(test_df, 20),
        "n_test_pairs": int(len(test_df)),
        "n_test_pos": int(test_df["label"].sum()),
        "feature_cols": FEATURE_COLS,
        "seed": int(args.seed),
        "test_size": float(args.test_size),
    }

    model_name = Path(args.model_path).stem
    out_path = Path(outdir) / f"eval_{model_name}.json"
    save_json(metrics, out_path)

    print(metrics)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
