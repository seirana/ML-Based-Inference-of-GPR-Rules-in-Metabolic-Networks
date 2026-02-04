#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""
'''
Train and evaluate a Logistic Regression baseline for predicting gene–reaction associations.

The script loads engineered features for labeled (reaction, gene) pairs, applies a predefined
reaction-wise train/test split (to avoid leakage across the same reaction), fits a class-balanced
Logistic Regression model, and evaluates performance using PR-AUC, ROC-AUC, and reaction-level
hit@k (whether a true gene is ranked in the top-k predictions per reaction). It saves metrics,
the trained model, and the learned coefficients for interpretability.
'''
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from utils import ensure_dir, save_json, seed_everything, load_split, indices_from_split

FEATURE_COLS = [
    "jacc_mets",
    "overlap_mets",
    "n_mets_rxn",
    "n_mets_gene_fp",
    "subsystem_match",
    "n_subsys_gene_fp",
]


def recall_at_k(df: pd.DataFrame, k: int) -> float:
    # df contains test rows for many reactions, with columns: reaction_id, label, score
    recalls = []
    for rid, sub in df.groupby("reaction_id"):
        pos = sub[sub["label"] == 1]
        if len(pos) == 0:
            continue
        topk = sub.sort_values("score", ascending=False).head(k)
        hit = int((topk["label"] == 1).any())
        # reaction-level hit@k; you can also do fraction of positives recovered, but hit@k is simplest
        recalls.append(hit)
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procdir", default="data/processed", type=str)
    ap.add_argument("--outdir", default="reports/metrics", type=str)
    ap.add_argument("--model_out", default="reports/models/logreg.joblib", type=str)
    ap.add_argument("--seed", default=13, type=int)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(Path(args.model_out).parent)
    outdir = ensure_dir(args.outdir)

    df = pd.read_parquet(Path(args.procdir) / "features.parquet")
    X = df[FEATURE_COLS].values
    y = df["label"].values

    train_r, test_r = load_split(Path(args.procdir) / "split_reactions.json")
    train_idx, test_idx = indices_from_split(df, train_r, test_r)


    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    ap_score = average_precision_score(yte, proba)
    roc = roc_auc_score(yte, proba)

    test_df = df.iloc[test_idx].copy()
    test_df["score"] = proba

    metrics = {
        "model": "logreg",
        "average_precision": float(ap_score),
        "roc_auc": float(roc),
        "hit_at_5": recall_at_k(test_df, 5),
        "hit_at_10": recall_at_k(test_df, 10),
        "hit_at_20": recall_at_k(test_df, 20),
        "n_test_pairs": int(len(test_df)),
        "n_test_pos": int(test_df["label"].sum()),
    }

    save_json(metrics, Path(outdir) / "logreg_metrics.json")
    joblib.dump(clf, args.model_out)

    # Save coefficients for interpretability
    coef = dict(zip(FEATURE_COLS, clf.coef_[0].tolist()))
    save_json({"intercept": float(clf.intercept_[0]), "coef": coef}, Path(outdir) / "logreg_coefficients.json")

    print(metrics)
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
