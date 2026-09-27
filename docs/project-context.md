# Project status

Updated 2026-09-27.

## Evaluation

Public leaderboard: approximately 40% of test rows. Private leaderboard: the other 60%, used for final leaderboard results. Subset membership is unknown. This is separate from the local validation split. Prefer robust local validation when comparing models; public leaderboard gains may not generalize to the private portion.

## Current baseline

- Model: word and character TF-IDF with balanced LinearSVC; `src/train_baseline.py`.
- Validation Macro F1: 0.5113727016 on 9,563 posts, versus a majority-class baseline of 0.2212752084.
- Validation uses one held-out fold of a seeded five-fold stratified group split; normalized duplicates stay together. This is not an average over five folds.
- Final model was refitted on all 47,817 training rows and generated 11,955 test predictions in `submissions/submission1_tfidf_svc.csv`.
- Saved model reload and all submission predictions, IDs, column names, and ordering were verified.
- The team's first submission achieved a public Macro F1 of **0.50248**, placing second when recorded. The leading score was 0.54629.
- Public results cover approximately 40% of the test set and do not establish performance or rank on the private 60%.

## Second submission

- Selected NB-SVM (`C=0.25`) using four candidate configurations on an inner grouped validation split.
- Macro F1 on the same original outer holdout: **0.5275587794**, up **0.0161860778** from the first model. This remains a single-holdout estimate.
- Training: `src/train_submission2.py`; classifier implementation: `src/nbsvm.py`; report: `reports/submission2.md`.
- Output: `submissions/submission2_tfidf_nbsvm.csv`.
- After two entries, team `DU_NameDiyeKiAsheJay` achieved a best public Macro F1 of **0.53744**, improving by **0.03496** over its first entry.
- Last recorded rank: **2**. The leading score was 0.54629, a gap of 0.00885. Private results are pending.

Leaderboard scores are recorded here in submission order. Uploaded filenames have not been checked against Kaggle's submission history.

## Repository and restrictions

- GitHub repository: `Jahidul183019/HerWILL-UAP-Datathon-2026` (private when last checked).
- Original dataset archive is in `data/archives/`; extracted CSVs are in `data/raw/`.
- Competition data, model artifacts, and prediction CSVs remain local and are ignored by Git. Competition rules prohibit distributing the dataset outside the competition.
- Both Kaggle predictions and best-result source code through the organizers' form are required. Code can be submitted through the form only once.

See `competition.md` for references and `../reports/baseline.md` for detailed baseline results. Check Git status for the current local/remote synchronization state.
