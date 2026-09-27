# Project status

Updated 2026-09-27.

Track our validation scores, Kaggle scores, and submission status. Do not track leaderboard positions or competitor scores.

## Evaluation

Public leaderboard: approximately 40% of test rows. Private leaderboard: the other 60%, used for final leaderboard results. Subset membership is unknown. This is separate from the local validation split. Prefer robust local validation when comparing models; public leaderboard gains may not generalize to the private portion.

## Current baseline

- Model: word and character TF-IDF with balanced LinearSVC; `src/train_baseline.py`.
- Validation Macro F1: 0.5113727016 on 9,563 posts, versus a majority-class baseline of 0.2212752084.
- Validation uses one held-out fold of a seeded five-fold stratified group split; normalized duplicates stay together. This is not an average over five folds.
- Final model was refitted on all 47,817 training rows and generated 11,955 test predictions in `submissions/submission1_tfidf_svc.csv`.
- Saved model reload and all submission predictions, IDs, column names, and ordering were verified.
- The team's first submission achieved a public Macro F1 of **0.50248**.
- Public results cover approximately 40% of the test set and do not establish performance on the private 60%.

## Second submission

- Selected NB-SVM (`C=0.25`) using four candidate configurations on an inner grouped validation split.
- Macro F1 on the same original outer holdout: **0.5275587794**, up **0.0161860778** from the first model. This remains a single-holdout estimate.
- Training: `src/train_submission2.py`; classifier implementation: `src/nbsvm.py`; report: `reports/submission2.md`.
- Output: `submissions/submission2_tfidf_nbsvm.csv`.
- After two entries, team `DU_NameDiyeKiAsheJay` achieved a best public Macro F1 of **0.53744**, improving by **0.03496** over its first entry.
- Private results are pending.

The first two leaderboard scores are recorded in submission order; their uploaded filenames have not been verified. The third submission record confirms both its filename and score.

## Third submission

- Pipeline: `src/train_submission3.py`; model and feature code: `src/submission3_model.py`; inference: `src/predict_submission3.py`.
- Six candidates and 60 score-combination configurations were evaluated on the inner grouped split. The outer holdout was used after selection.
- Selected: NB-SVM with 15 numerical text features, `C=0.1`, `alpha=1.0`, score offsets `[0, -0.1, 0]`. One model, no label cleaning.
- Validation Macro F1: **0.5344191161**, an increase of **0.0068603367** over submission 2. Subtle-toxicity F1 declined while explicit and neutral F1 improved.
- Model bundle: `models/submission3/final_model.joblib`. Upload CSV: `submissions/submission3_tfidf_nbsvm_meta.csv`, a byte-identical copy of the retained `submissions/submission3_ensemble.csv` pipeline output.
- All 11,955 predictions were independently reproduced in a separate process; hashes confirmed earlier protected artifacts were unchanged. The Desktop proposal remains unchanged.
- Kaggle status: **Complete** for `submission3_tfidf_nbsvm_meta.csv`, with public Macro F1 **0.54601**. Improvement over submission 2: **0.00857**; improvement over submission 1: **0.04353**.
- Final-selection checkbox was unchecked in the recorded submission view. The private score is unconfirmed.
- Preserve all previous data, code, models, and submissions. The new training script rejects existing output paths and records hashes to verify earlier artifacts remain unchanged.

## Fourth submission

- Script: `src/train_submission4.py`; report: `reports/submission4.md`; metrics: `reports/submission4_metrics.json`.
- Selected three NB-SVM feature configurations using the inner grouped split, with `C=0.1`, `alpha=1.0`, approximately equal weights, and score offsets `[0, -0.05, -0.05]`.
- Validation Macro F1: **0.5376578865**, up **0.0032387704** from submission 3. The gain is small; the paired group-bootstrap interval includes zero.
- Kaggle status: **Complete** for `submission4_tfidf_nbsvm_views.csv`, with public Macro F1 **0.54759**, up **0.00158** from submission 3. Its final-selection checkbox was unchecked when recorded.
- Final and validation model predictions were independently reproduced. The final CSV contains 11,955 rows.
- The script now checks all output paths before training and supports `--run-name` and `--reference-validation` for isolated reruns. Its modeling logic is unchanged. The original executed script is retained locally in `models/submission4_training_original.py`.

## Fifth submission experiment

- XLM-R pipeline: `src/train_submission5.py`; setup and evaluation plan: `reports/submission5.md`.
- MPS is available outside the sandbox, and a short local GPU benchmark succeeded. Full training is assigned to Kaggle GPU.
- Inner validation selects the epoch count before outer evaluation. Final refit and CSV generation require outer Macro F1 above submission 4's 0.5376578865. The reused holdout remains a development estimate.
- Kaggle training version 2 completed successfully. Inner validation selected epoch 3 at Macro F1 0.5500206584. The directly comparable outer holdout score is 0.5590271875, a gain of 0.0213693011 over submission 4. The generated `submissions/submission5_xlmr.csv` passed model-reload, ID, ordering, label and hash checks. Its Kaggle competition score is pending manual upload.
- Local training launched on 2026-09-27 at 21:34 Bangladesh time, then was stopped at the request to switch to Kaggle GPU. The process group was terminated and local files are preserved; the stop record is `models/submission5_launch/stopped.json`.
- Kaggle phone and identity verification are complete, and the CLI is authenticated. Version 2 of `jahidul183019/submission-5-xlm-r` completed on a T4 GPU. It keeps the selected checkpoint weights while retaining every epoch's metrics and validation logits. The launch configuration is under `kaggle/submission5-script/`. The local `.ipynb` draft was removed.
- Earlier scripts, models, data, metrics and predictions are preserved. Training runs independently of Codex and makes no OpenAI API calls.

## Repository and restrictions

- Teammate setup, reproduction commands, and next-model guidance: `docs/team-setup.md`.

- GitHub repository: `Jahidul183019/HerWILL-UAP-Datathon-2026` (private when last checked).
- Original dataset archive is in `data/archives/`; extracted CSVs are in `data/raw/`.
- The three original CSVs in `data/raw/` are tracked in the private repository for authorized team access. Archives, processed data, model artifacts, and prediction CSVs remain ignored. Competition rules prohibit distributing the dataset outside the competition.
- Both Kaggle predictions and best-result source code through the organizers' form are required. Code can be submitted through the form only once.

See `competition.md` for references and `../reports/baseline.md` for detailed baseline results. Check Git status for the current local/remote synchronization state.
