#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Configuration
# ----------------------------
MODEL_XML="data/raw/iJO1366.xml"
PROCDIR="data/processed"
REPORTS="reports"
SEED=13
TEST_SIZE=0.2
NEG_PER_POS=10
TOPK=10

echo "========================================"
echo " Running Gene–Reaction ML Pipeline"
echo "========================================"

# ----------------------------
# 0. Parse metabolic model
# ----------------------------
echo "[0] Parsing SBML model"
python src/00_parse_model.py \
  --model "$MODEL_XML" \
  --outdir "$PROCDIR"

# ----------------------------
# 1. Train/test split (reaction-wise)
# ----------------------------
echo "[1] Creating reaction-wise split"
python src/00b_make_split.py \
  --procdir "$PROCDIR" \
  --out "$PROCDIR/split_reactions.json" \
  --seed "$SEED" \
  --test_size "$TEST_SIZE"

# ----------------------------
# 2. Build reaction–gene pairs
# ----------------------------
echo "[2] Building positive and hard-negative pairs"
python src/01_build_pairs.py \
  --procdir "$PROCDIR" \
  --outdir "$PROCDIR" \
  --neg_per_pos "$NEG_PER_POS" \
  --seed "$SEED"

# ----------------------------
# 3. Feature engineering
# ----------------------------
echo "[3] Computing features"
python src/02_features.py \
  --procdir "$PROCDIR" \
  --outdir "$PROCDIR"

# ----------------------------
# 4. Train Logistic Regression
# ----------------------------
echo "[4] Training Logistic Regression"
python src/03_train_logreg.py \
  --procdir "$PROCDIR"

# ----------------------------
# 5. Train XGBoost
# ----------------------------
echo "[5] Training XGBoost"
python src/04_train_xgb.py \
  --procdir "$PROCDIR"

# ----------------------------
# 6. Evaluate models
# ----------------------------
echo "[6] Evaluating XGBoost"
python src/05_evaluate.py \
  --procdir "$PROCDIR" \
  --model_path "$REPORTS/models/xgb.joblib"

echo "[6] Evaluating Logistic Regression"
python src/05_evaluate.py \
  --procdir "$PROCDIR" \
  --model_path "$REPORTS/models/logreg.joblib"

# ----------------------------
# 7. Rank candidate genes
# ----------------------------
echo "[7] Ranking candidate genes (top-$TOPK)"
python src/06_rank_candidates.py \
  --procdir "$PROCDIR" \
  --model_path "$REPORTS/models/xgb.joblib" \
  --outdir "$REPORTS/candidates" \
  --topk "$TOPK"

echo "========================================"
echo " Pipeline finished successfully"
echo "========================================"

