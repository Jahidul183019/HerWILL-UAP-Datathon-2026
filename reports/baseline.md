# First baseline results

The word + character TF-IDF / balanced LinearSVC baseline achieved **0.511373 Macro F1** on a fixed holdout of 9,563 posts. A majority-class dummy model scored **0.221275** on the same holdout. Accuracy was 0.557670.

| Class | Precision | Recall | F1 | Validation posts |
| --- | ---: | ---: | ---: | ---: |
| 0 — Explicitly toxic / hateful | 0.6343 | 0.6540 | 0.6440 | 4,751 |
| 1 — Subtly toxic | 0.4905 | 0.4919 | 0.4912 | 3,633 |
| 2 — Non-toxic / neutral | 0.4295 | 0.3723 | 0.3989 | 1,179 |

## Validation design

The training partition contains 38,254 posts. Validation uses the first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; this is one holdout evaluation, not a five-fold averaged result. Groups are texts normalized with Unicode NFC, lowercase conversion, and collapsed whitespace. There is no normalized text overlap between the fit and validation partitions. Vectorizers learn their vocabularies and IDF weights exclusively from the fit partition.

The training set has 669 duplicate rows by normalized text and 63 normalized text groups with conflicting labels. All rows and original labels are retained; conflicting groups remain together. The test set contains 298 texts also present in training after normalization. Test text and predictions do not influence model selection or validation, and no lookup-based prediction overrides are used.

## Model and reproducibility

- Word TF-IDF: whitespace-delimited tokens, 1–2 grams, at most 80,000 features.
- Character TF-IDF: 3–5 grams, at most 150,000 features.
- Both feature sets use `min_df=2`, sublinear term frequency, and default L2 normalization.
- LinearSVC: `C=1.0`, balanced class weights, `dual="auto"`, 5,000 maximum iterations, seed 42.
- No external dataset or pretrained model is used. Settings were fixed before evaluating this baseline.
- Actual run environment: Python 3.14.3; exact dependency versions and input SHA-256 hashes are in `baseline_metrics.json`.

Run `python3 src/train_baseline.py` from the repository to reproduce the evaluation and final refit. The script treats classifier convergence warnings as errors. A fresh pipeline is trained on all 47,817 labeled rows after evaluation and produces `submissions/submission1_tfidf_svc.csv` with 11,955 predictions. Local per-row validation outputs and model artifacts are ignored by Git.

## Interpretation

Explicit toxicity is the strongest class. Neutral recall (0.3723) is the weakest result; subtle and explicit toxicity are also frequently confused. The single holdout does not estimate variation across folds, and normalized grouping does not catch every near-duplicate or shared source.

## Public leaderboard feedback

Team `DU_NameDiyeKiAsheJay` achieved a public Macro F1 of **0.50248** on its first entry, placing second when recorded. This was 0.008893 below the local validation score of 0.511373. The two scores use different data subsets; their proximity alone does not establish validation reliability.

The leading score at that point was 0.54629. Public scoring uses approximately 40% of test data; final leaderboard results use the other 60%. Model selection remains based on local validation.
