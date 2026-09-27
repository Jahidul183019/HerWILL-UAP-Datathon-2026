# Submission 3

This experiment builds on submission 2 and the proposed Desktop training script. It keeps the same grouped validation split and tests numerical text features, NB-SVM variants, label cleaning, and an ensemble with logistic regression.

## Results

Validation Macro F1 increased from **0.52756 to 0.53442**, a gain of **0.00686** on the same 9,563 posts.

| Class | Submission 2 F1 | Submission 3 F1 |
| --- | ---: | ---: |
| Explicitly toxic / hateful | 0.65089 | 0.66231 |
| Subtly toxic | 0.52533 | 0.51606 |
| Non-toxic / neutral | 0.40646 | 0.42489 |
| Macro average | 0.52756 | 0.53442 |

The selected configuration is a single NB-SVM with numerical text features, `C=0.1`, and `alpha=1.0`. After score standardization, class 1 receives an offset of -0.1; other classes receive no offset. No label cleaning is used in the selected model. Gains in explicit toxicity and neutral classification outweigh a decline in subtle-toxicity F1. Neutral recall improved from 0.36302 to 0.41985.

The selected configuration uses one model. The upload file is **`submissions/submission3_tfidf_nbsvm_meta.csv`**, an identical copy of the pipeline output `submission3_ensemble.csv`. Both files are retained.

## Kaggle result

Kaggle accepted `submission3_tfidf_nbsvm_meta.csv` with a public Macro F1 of **0.54601**.

| Submission | Validation Macro F1 | Public Macro F1 |
| --- | ---: | ---: |
| 1 | 0.51137 | 0.50248 |
| 2 | 0.52756 | 0.53744 |
| 3 | 0.53442 | 0.54601 |

The public score increased by 0.00857 over submission 2. The submission record confirms successful processing. Its final-selection checkbox was unchecked when recorded. Final results depend on the private 60% of test data.

## Method

TF-IDF settings match submission 2: word 1–2 grams, character 2–5 grams, vocabulary caps of 150,000 and 250,000, and `min_df=2`. The additional features describe length, capitalization, digits, punctuation, emoji, Bangla characters, repetition, vocabulary diversity, and whitespace. Counts and lengths use log transforms where appropriate. Numerical features are standardized on training rows and weighted by 0.2.

NB weights apply only to text features. Numerical features enter the classifier separately; they are not treated as word-presence counts. The reference candidate uses the original submission 2 classifier without numerical features.

The label-cleaning candidate changes labels only where a normalized text group has a strict majority. Ties remain unchanged. These changes occur in memory within the fitting partition; the original data and evaluation labels are preserved.

## Model selection

Six candidates are compared on the same inner split used for submission 2: 28,690 fitting rows and 9,564 validation rows. The top one, two, and three candidates are evaluated with equal weights and, for multiple models, weights proportional to their inner Macro F1.

| Candidate | Inner Macro F1 before class offsets |
| --- | ---: |
| Submission 2 reference | 0.51704 |
| NB-SVM + numerical features, C=0.25 | 0.51561 |
| NB-SVM + numerical features, C=0.1 | 0.52624 |
| NB-SVM + numerical features, alpha=0.5 | 0.51188 |
| NB-SVM + numerical features, strict-majority labels | 0.51701 |
| Logistic regression + numerical features | 0.51714 |

The selected configuration scored 0.52904 on inner validation after the class-score adjustment. The search considered 60 combinations of member counts, weighting schemes, and offsets; all are recorded in `submission3_metrics.json`.

Each model's decision scores are divided by a scalar standard deviation measured on its fitting rows. The scores are not calibrated probabilities. A fixed grid of class offsets is also tested on inner validation: class 0 stays at zero, class 1 uses -0.1, 0, or 0.1, and class 2 uses 0, 0.1, 0.2, or 0.3. These offsets adjust the class decision boundaries for Macro F1.

The configuration is fixed before evaluation on the original 9,563-row outer holdout. Duplicate text groups stay together throughout, and feature extraction is fitted separately within each training partition. The outer holdout has been used in earlier experiments, so it remains a development estimate rather than an untouched final test.

## Reproduction

Install the pinned `requirements.txt` dependencies, retain the scripts in `src/`, and run:

```bash
python3 src/train_submission3.py
```

The script requires `data/processed/submission2_validation.csv` for an ID-aligned comparison with submission 2. If missing, reproduce submissions 1 and 2 first. Existing output paths cause the script to stop; use `--run-name submission3_repeat1` for a separate rerun.

Artifacts include all fitted inner candidates, the outer validation model, validation predictions, source snapshots, source and input hashes, search results, and environment versions. A final model and CSV are generated only when outer Macro F1 exceeds submission 2. Final inference uses the selected weights and offsets unchanged.

To regenerate the predictions from the saved final model in a separate process:

```bash
python3 src/predict_submission3.py
```

This writes a separate reproduction CSV and checks it against the original. Original data, previous model files, and previous submission CSVs are checked by SHA-256 before and after training. No Kaggle upload is performed by either script.

## Verification

All 11,955 predictions were reproduced from the saved model in a separate Python process. CSV columns, IDs, row order, and labels were checked. The upload copy matches the pipeline output byte for byte, and hashes confirmed all protected earlier artifacts were unchanged. Training completed in 220.85 seconds in the recorded environment. The original Desktop script was not modified.
