# Submission 2: TF-IDF + NB-SVM

Validation Macro F1 improved from **0.51137 to 0.52756** on the same 9,563 posts. The largest gain was in subtle toxicity.

| Class | Submission 1 F1 | Submission 2 F1 |
| --- | ---: | ---: |
| Explicitly toxic / hateful | 0.64400 | 0.65089 |
| Subtly toxic | 0.49120 | 0.52533 |
| Non-toxic / neutral | 0.39891 | 0.40646 |
| Macro average | 0.51137 | 0.52756 |

## Changes

Word features still use 1–2 grams; the vocabulary cap increased to 150,000. Character features now cover 2–5 grams, with a cap of 250,000. Both use TF-IDF, sublinear term frequency, L2 normalization, and a minimum document frequency of two.

For each class, NB-SVM estimates which features are more common in that class than in the rest of the training data. It uses the log ratio of smoothed, normalized binary document counts to reweight TF-IDF features, then fits a balanced binary linear SVM. Predictions use the class with the largest decision score. Ratios use additive smoothing of 1; the selected SVM uses `C=0.25`, a 10,000-iteration limit, and seed 42.

The feature-reweighting approach follows [Wang and Manning (2012)](https://aclanthology.org/P12-2018/). The implementation here uses one-vs-rest classifiers and TF-IDF inputs. It uses no external training data or pretrained models.

## Selection and validation

The original outer training partition was split again: 28,690 posts for fitting and 9,564 for selecting the model. This inner split uses the first fold of four-fold stratified group splitting, seed 43. Normalized duplicates remain grouped at both levels. Vectorizers and class-specific ratios are fitted only on each split's training rows.

| Candidate | C | Inner Macro F1 |
| --- | ---: | ---: |
| LinearSVC | 0.25 | 0.51681 |
| LinearSVC | 1.00 | 0.50286 |
| NB-SVM | 0.25 | 0.51704 |
| NB-SVM | 1.00 | 0.49711 |

NB-SVM at `C=0.25` was selected before evaluating it on the original outer holdout. Its inner lead over LinearSVC is small; this does not establish that NB-SVM is generally better. The selected model was fitted on the 38,254 outer training rows and evaluated on the same 9,563 posts used for submission 1. These are single-split results, not averaged cross-validation scores.

## Reproduce and submit

Install `requirements.txt`. Keep `src/train_baseline.py`, `src/train_submission2.py`, and `src/nbsvm.py` together. Run the baseline first if `data/processed/baseline_validation.csv` is missing, then run:

```bash
python3 src/train_submission2.py
```

The script selects a configuration, checks its holdout score against the baseline, refits on all 47,817 training rows, and writes `submissions/submission2_tfidf_nbsvm.csv`. Submission 1 is preserved. Full metrics, versions, and input hashes are in `submission2_metrics.json`.

To reload the saved model, add `src/` to the Python import path so `nbsvm` is available, and normalize text with `train_baseline.normalize` before prediction.

## Public leaderboard result

After its second entry, team `DU_NameDiyeKiAsheJay` improved its best public Macro F1 from **0.50248 to 0.53744**. The team remained second when recorded, trailing the leading score of 0.54629 by 0.00885.

| Result | Submission 1 | Submission 2 | Change |
| --- | ---: | ---: | ---: |
| Local validation Macro F1 | 0.51137 | 0.52756 | +0.01619 |
| Public leaderboard Macro F1 | 0.50248 | 0.53744 | +0.03496 |

Public and validation scores improved, but the public leaderboard covers approximately 40% of test rows. Final leaderboard results depend on the other 60%; the private score and rank remain unknown. Continue using local validation to select models.
