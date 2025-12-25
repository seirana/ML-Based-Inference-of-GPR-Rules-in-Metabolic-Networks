
# 🧬 GPR-ML: Machine Learning Based Inference of Gene-Protein-Reaction Rules in Metabolic Networks

> **Predict missing and incorrect GPR rules in genome-scale metabolic models using interpretable machine learning.**

**GPR-ML** is a lightweight, interpretable machine-learning framework that learns **gene–reaction relationships directly from metabolic network structure** and predicts **missing candidate genes for metabolic reactions**.

It is designed as a clean, reproducible baseline for **automatic GPR rule completion and correction**.

---

## 🔬 Why this matters

Gene–Protein–Reaction (GPR) rules define which genes activate metabolic reactions.
They are manually curated, incomplete, and error-prone, yet every flux simulation depends on them.

**GPR-ML** provides:

• a fully data-driven method
• no omics input required
• interpretable features
• realistic reaction-wise generalization
• ranked candidate genes per reaction

---

## 🚀 What GPR-ML does

For every reaction (r) and gene (g):

Predict whether g belongs to the GPR of r

It produces:

• reaction–gene link predictions
• candidate missing genes
• model explanations
• recall@K rankings

---

## 🧠 Models

| Model               | Purpose                |
| ------------------- | ---------------------- |
| Logistic Regression | Interpretable baseline |
| XGBoost             | High-performance model |

---

## 🧩 Features (network-derived, biology-aware)

| Feature                         | Meaning                    |
| ------------------------------- | -------------------------- |
| Metabolite Jaccard similarity   | biochemical compatibility  |
| Metabolite overlap count        | shared biochemical context |
| Reaction size                   | complexity control         |
| Gene metabolic fingerprint size | enzyme versatility         |
| Subsystem compatibility         | pathway consistency        |

---

## 🧬 Dataset

Currently supported:

• *E. coli* **iJO1366** (BiGG Models)

---

## 📁 Repository structure

```
data/
  raw/
  processed/
reports/
  metrics/
  models/
  candidates/
src/
  00_parse_model.py
  01_build_pairs.py
  02_features.py
  03_train_logreg.py
  04_train_xgb.py
  05_evaluate.py
  06_rank_candidates.py
  utils.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/seirana/ML-Based-Inference-of-GPR-Rules-in-Metabolic-Networks.git
cd gpr-ml
pip install -r requirements.txt
```

---

## ▶️ Quickstart (E. coli iJO1366)

Download `iJO1366.xml` and place it in:

```
data/raw/iJO1366.xml
```

Then run:

```bash
python src/00_parse_model.py --model data/raw/iJO1366.xml
python src/00_parse_model.py --model data/raw/iJO1366.xml --outdir data/processed
python src/00b_make_split.py --procdir data/processed --out data/processed/split_reactions.json --seed 13 --test_size 0.2
python src/01_build_pairs.py --procdir data/processed --outdir data/processed --neg_per_pos 10
python src/02_features.py --procdir data/processed --outdir data/processed
python src/03_train_logreg.py --procdir data/processed
python src/04_train_xgb.py --procdir data/processed
python src/05_evaluate.py --procdir data/processed --model_path reports/models/xgb.joblib
python src/05_evaluate.py --procdir data/processed --model_path reports/models/logreg.joblib
python src/06_rank_candidates.py --procdir data/processed --model_path reports/models/xgb.joblib --outdir reports/candidates --topk 10
```

---

## 📤 Outputs

| File                                        | Description             |
| ------------------------------------------- | ----------------------- |
| `reports/metrics/*.json`                    | model performance       |
| `reports/models/*.joblib`                   | trained models          |
| `reports/candidates/top_candidates_xgb.csv` | predicted missing genes |

---

## 🧪 Use cases

• GPR curation assistance
• Model quality control
• Automated GEM refinement
• Comparative metabolic modeling
