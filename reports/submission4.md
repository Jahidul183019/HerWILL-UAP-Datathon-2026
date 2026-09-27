# Submission 4: TF-IDF NB-SVM ensemble

Kaggle accepted `submission4_tfidf_nbsvm_views.csv` with a public Macro F1 of **0.54759**, an increase of **0.00158** over submission 3.

| Submission | Validation Macro F1 | Public Macro F1 |
| --- | ---: | ---: |
| 1 | 0.51137 | 0.50248 |
| 2 | 0.52756 | 0.53744 |
| 3 | 0.53442 | 0.54601 |
| 4 | 0.53766 | 0.54759 |

## Model

The selected ensemble combines three NB-SVM models with different TF-IDF features:

| Features | Ensemble weight |
| --- | ---: |
| Word 1–3 grams + character-within-word 2–6 grams | 0.334240 |
| Word 1–2 grams + character 2–5 grams | 0.334166 |
| Character-within-word 2–6 grams | 0.331594 |

All three use `C=0.1`, `alpha=1.0`, and balanced binary class weights. Each model's scores are scaled by their standard deviation on its fitting rows. The weighted score average receives class offsets `[0, -0.05, -0.05]` before selecting the largest score. Training labels are unchanged, and no external data is used.

Five feature configurations and 125 ensemble/offset combinations were tested on the inner grouped split. The selected settings were then evaluated on the same outer holdout as earlier submissions. Final models were refitted on all 47,817 training rows.

## Validation and verification

The outer Macro F1 increased from 0.534419 to 0.537658. Subtle-toxicity F1 improved from 0.51606 to 0.53474, while explicit-toxicity and neutral F1 declined slightly.

A paired bootstrap of normalized text groups, with 2,000 resamples and seed 42, gave an approximate 95% interval of [-0.00337, 0.01003] for the gain over submission 3. This interval includes zero and does not account for repeated experiment development on the same holdout. The small validation and public-score gains do not guarantee a gain on the private test portion.

All 11,955 test predictions and all validation predictions were reproduced from their saved models in a separate process. Input hashes, CSV schema, ID order, labels, holdout alignment, and duplicate-group separation were verified.

## Files and status

- Training script: `src/train_submission4.py`.
- Detailed metrics: `reports/submission4_metrics.json`.
- Final model: `models/submission4_tfidf_nbsvm_views.joblib`.
- Validation model: `models/submission4_validation_model.joblib`.
- Inner candidates: `models/submission4_inner_candidates.joblib`.
- Upload file: `submissions/submission4_tfidf_nbsvm_views.csv`.

The training script checks all output paths before starting. Use a unique `--run-name` for a rerun, and `--reference-validation` to specify a reproduced submission 3 validation CSV. The training and prediction logic is unchanged; the original script used for the recorded run is retained locally in `models/submission4_training_original.py`. See `docs/team-setup.md` for the full reproduction sequence.

Kaggle status is Complete. The final-selection checkbox was unchecked when recorded. Private evaluation uses the remaining 60% of test data.
