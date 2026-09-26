#!/usr/bin/env python3
"""Portable orchestrator for the complete GPR-ML pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn
import xgboost

from gpr_ml import ensure_dir, save_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None
    value = result.stdout.strip()
    return value or None


def run_step(
    label: str,
    command: Sequence[str],
) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    subprocess.run(
        list(command),
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete leakage-aware GPR-ML "
            "training and ranking pipeline."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "data/raw/iJO1366.xml"
        ),
    )
    parser.add_argument(
        "--procdir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--neg_per_pos",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--bootstrap_reps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--skip_xgb",
        action="store_true",
        help=(
            "Run parsing, features, heuristic, and "
            "logistic regression without XGBoost."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}"
        )

    ensure_dir(args.procdir)
    metrics_dir = ensure_dir(
        args.reports / "metrics"
    )
    models_dir = ensure_dir(
        args.reports / "models"
    )
    candidates_dir = ensure_dir(
        args.reports / "candidates"
    )

    python = sys.executable

    steps: list[
        tuple[str, list[str]]
    ] = [
        (
            "0 parse model",
            [
                python,
                "src/00_parse_model.py",
                "--model",
                str(args.model),
                "--outdir",
                str(args.procdir),
            ],
        ),
        (
            "1 create reaction split",
            [
                python,
                "src/00b_make_split.py",
                "--procdir",
                str(args.procdir),
                "--out",
                str(
                    args.procdir
                    / "split_reactions.json"
                ),
                "--seed",
                str(args.seed),
                "--test_size",
                str(args.test_size),
            ],
        ),
        (
            "2 build pairs",
            [
                python,
                "src/01_build_pairs.py",
                "--procdir",
                str(args.procdir),
                "--outdir",
                str(args.procdir),
                "--neg_per_pos",
                str(args.neg_per_pos),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "3 build features",
            [
                python,
                "src/02_features.py",
                "--procdir",
                str(args.procdir),
                "--outdir",
                str(args.procdir),
            ],
        ),
        (
            "4 evaluate heuristic baseline",
            [
                python,
                "src/03a_evaluate_heuristic.py",
                "--procdir",
                str(args.procdir),
                "--outdir",
                str(metrics_dir),
                "--seed",
                str(args.seed),
                "--bootstrap_reps",
                str(args.bootstrap_reps),
            ],
        ),
        (
            "5 train logistic regression",
            [
                python,
                "src/03_train_logreg.py",
                "--procdir",
                str(args.procdir),
                "--outdir",
                str(metrics_dir),
                "--model_out",
                str(
                    models_dir
                    / "logreg.joblib"
                ),
                "--seed",
                str(args.seed),
                "--bootstrap_reps",
                str(args.bootstrap_reps),
            ],
        ),
    ]

    if not args.skip_xgb:
        steps.extend(
            [
                (
                    "6 train XGBoost",
                    [
                        python,
                        "src/04_train_xgb.py",
                        "--procdir",
                        str(args.procdir),
                        "--outdir",
                        str(metrics_dir),
                        "--model_out",
                        str(
                            models_dir
                            / "xgb.joblib"
                        ),
                        "--seed",
                        str(args.seed),
                        "--bootstrap_reps",
                        str(
                            args.bootstrap_reps
                        ),
                    ],
                ),
                (
                    "7 evaluate XGBoost",
                    [
                        python,
                        "src/05_evaluate.py",
                        "--procdir",
                        str(args.procdir),
                        "--model_path",
                        str(
                            models_dir
                            / "xgb.joblib"
                        ),
                        "--outdir",
                        str(metrics_dir),
                        "--seed",
                        str(args.seed),
                        "--bootstrap_reps",
                        str(
                            args.bootstrap_reps
                        ),
                    ],
                ),
                (
                    "8 rank candidates",
                    [
                        python,
                        "src/06_rank_candidates.py",
                        "--procdir",
                        str(args.procdir),
                        "--model_path",
                        str(
                            models_dir
                            / "xgb.joblib"
                        ),
                        "--outdir",
                        str(candidates_dir),
                        "--topk",
                        str(args.topk),
                        "--seed",
                        str(args.seed),
                    ],
                ),
            ]
        )

    for label, command in steps:
        run_step(label, command)

    metadata = {
        "command": "gprml-pipeline",
        "arguments": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value
            in vars(args).items()
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": (
            sklearn.__version__
        ),
        "xgboost": xgboost.__version__,
        "git_commit": git_commit(),
        "input_model_sha256": file_sha256(
            args.model
        ),
    }
    save_json(
        metadata,
        args.reports
        / "run_metadata.json",
    )

    print(
        "\nPipeline finished successfully."
    )
    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
